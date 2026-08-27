"""R2/S3: presigned PUT cho upload, presigned GET cho keyframe/clip."""
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import requests
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

# Load môi trường từ .env nếu có
load_dotenv()


class ProgressPercentage:
    """Class hỗ trợ callback cập nhật thanh tiến trình tqdm cho boto3 upload."""

    def __init__(self, filename, filesize, pbar):
        self._filename = filename
        self._filesize = filesize
        self._seen_so_far = 0
        self._lock = threading.Lock()
        self._pbar = pbar

    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            self._pbar.update(bytes_amount)


def neighbor_slice_bounds(sorted_frames: list, from_frame: int, to_frame: int,
                          before: int, after: int) -> tuple:
    """(start, end) để cắt `sorted_frames[start:end]` -- N khung trước from_frame đến
    N khung sau to_frame, theo VỊ TRÍ trong danh sách (không phải số học frame, vì
    keyframe không cách đều nhau).

    to_frame == from_frame tái hiện đúng hành vi 1-mỏ neo cũ của get_neighbor_frames.
    Thứ tự from/to không quan trọng (chain temporal luôn đúng thứ tự, nhưng lấy
    min/max ở đây không tốn gì thêm).
    """
    def _index(frame: int) -> int:
        try:
            return sorted_frames.index(frame)
        except ValueError:
            raise ValueError(f"Không tìm thấy frame {frame} trong danh sách S3") from None

    lo, hi = sorted((_index(from_frame), _index(to_frame)))
    start = max(0, lo - before)
    end = min(len(sorted_frames), hi + after + 1)
    return start, end


class AWSStorageHelper:
    """Class tổng hợp tất cả các thao tác tương tác với AWS S3 và CloudFront CDN."""

    def __init__(
        self,
        bucket_name: str | None = None,
        cloudfront_domain: str | None = None,
        region: str | None = None,
        aws_access_key: str | None = None,
        aws_secret_key: str | None = None,
        max_workers: int = 10,
    ):
        # 1. Khởi tạo cấu hình S3
        self.bucket = bucket_name or os.environ.get("AWS_BUCKET_NAME", "aic-bucket-hcmus")
        self.region = region or os.environ.get("AWS_REGION", "ap-southeast-2")
        self.cloudfront_domain = cloudfront_domain or os.environ.get("CLOUDFRONT_DOMAIN")

        access_key = aws_access_key or os.environ.get("AWS_ACCESS_KEY")
        secret_key = aws_secret_key or os.environ.get("AWS_SECRET_KEY")

        if access_key and secret_key:
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=self.region,
            )
        else:
            self.s3_client = boto3.client("s3", region_name=self.region)

        self.max_workers = max_workers

        # 2. Khởi tạo Requests Session với cơ chế Retry cho CloudFront Download
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    # ==========================================
    # 🔗 URL BUILDERS
    # ==========================================

    def get_s3_public_url(self, object_name: str) -> str:
        """Tạo đường dẫn public trực tiếp từ S3 Bucket."""
        return f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{object_name}"

    def get_cloudfront_url(self, object_name: str) -> str:
        """Tạo đường dẫn công khai qua CloudFront CDN."""
        if not self.cloudfront_domain:
            raise ValueError("CloudFront domain chưa được cấu hình!")
        domain = self.cloudfront_domain.replace("https://", "").strip("/")
        return f"https://{domain}/{object_name}"

    # ==========================================
    # 📤 UPLOAD FUNCTIONS (Direct to S3)
    # ==========================================

    def upload_file(
        self, file_path: str, object_name: str | None = None, storage_class: str = "STANDARD"
    ) -> str | None:
        """Upload 1 file đơn lẻ lên S3 kèm thanh tiến trình."""
        try:
            if object_name is None:
                object_name = os.path.basename(file_path)

            filesize = os.path.getsize(file_path)
            with tqdm(total=filesize, unit="B", unit_scale=True, desc=f"Uploading {os.path.basename(file_path)}") as pbar:
                self.s3_client.upload_file(
                    file_path,
                    self.bucket,
                    object_name,
                    ExtraArgs={"StorageClass": storage_class},
                    Callback=ProgressPercentage(file_path, filesize, pbar),
                )

            url = self.get_s3_public_url(object_name)
            print(f"\n✅ Uploaded {file_path} → {url}")
            return url
        except FileNotFoundError:
            print("❌ Không tìm thấy file cục bộ.")
        except NoCredentialsError:
            print("❌ Không tìm thấy AWS credentials.")
        except Exception as e:  # noqa: BLE001
            print(f"❌ Lỗi Upload: {e}")

    def upload_large_file(
        self, file_path: str, object_name: str | None = None, part_size: int = 50 * 1024 * 1024
    ) -> str | None:
        """Multipart upload cho file dung lượng lớn."""
        try:
            if object_name is None:
                object_name = os.path.basename(file_path)

            config = TransferConfig(
                multipart_threshold=part_size,
                multipart_chunksize=part_size,
                max_concurrency=10,
                use_threads=True,
            )

            filesize = os.path.getsize(file_path)
            with tqdm(total=filesize, unit="B", unit_scale=True, desc=f"Uploading Large {os.path.basename(file_path)}") as pbar:
                self.s3_client.upload_file(
                    file_path,
                    self.bucket,
                    object_name,
                    Config=config,
                    Callback=ProgressPercentage(file_path, filesize, pbar),
                )

            url = self.get_s3_public_url(object_name)
            print(f"\n🚀 File lớn đã upload thành công: {file_path} → {url}")
            return url
        except Exception as e:  # noqa: BLE001
            print(f"❌ Lỗi Upload file lớn: {e}")

    def upload_many(
        self, file_mappings: list, storage_class: str = "STANDARD", max_workers: int | None = None
    ) -> list:
        """
        Upload danh sách nhiều file song song.
        file_mappings: danh sách các tuple [(local_path, s3_object_name), ...]
        """
        workers = max_workers or self.max_workers
        results = []
        total_size = sum(os.path.getsize(fp) for fp, _ in file_mappings)

        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Total Progress") as global_pbar, ThreadPoolExecutor(
            max_workers=workers
        ) as executor:
            future_to_file = {}
            for file_path, object_name in file_mappings:
                filesize = os.path.getsize(file_path)
                s3_key = object_name or os.path.basename(file_path)
                future = executor.submit(
                    self.s3_client.upload_file,
                    file_path,
                    self.bucket,
                    s3_key,
                    ExtraArgs={"StorageClass": storage_class},
                    Callback=ProgressPercentage(file_path, filesize, global_pbar),
                )
                future_to_file[future] = (file_path, s3_key)

            for future in as_completed(future_to_file):
                file_path, s3_key = future_to_file[future]
                try:
                    future.result()
                    results.append(self.get_s3_public_url(s3_key))
                except Exception as e:  # noqa: BLE001
                    print(f"❌ Upload thất bại {file_path}: {e}")
                    results.append(None)

        return results

    def upload_folder(
        self,
        local_folder: str,
        s3_prefix: str = "",
        storage_class: str = "STANDARD",
        max_workers: int | None = None,
    ) -> list:
        """Upload toàn bộ thư mục hoặc một file cục bộ lên S3."""
        file_mappings = []

        if os.path.isfile(local_folder):
            file_name = os.path.basename(local_folder)
            object_name = os.path.join(s3_prefix, file_name).replace("\\", "/") if s3_prefix else file_name
            file_mappings.append((local_folder, object_name))
        else:
            for root, _, files in os.walk(local_folder):
                for file in files:
                    local_path = os.path.join(root, file)
                    rel_path = os.path.relpath(local_path, local_folder)
                    object_name = os.path.join(s3_prefix, rel_path).replace("\\", "/")
                    file_mappings.append((local_path, object_name))

        if not file_mappings:
            print("⚠️ Thư mục trống hoặc không tìm thấy file:", local_folder)
            return []

        print(f"📂 Tìm thấy {len(file_mappings)} file cần upload từ {local_folder}")
        return self.upload_many(file_mappings, storage_class=storage_class, max_workers=max_workers)

    # ==========================================
    # 📥 DOWNLOAD FUNCTIONS (Via CloudFront CDN or S3)
    # ==========================================

    def download_file_s3(self, object_name: str, dest_path: str):
        """Download file trực tiếp từ S3 (không qua CloudFront)."""
        try:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            self.s3_client.download_file(self.bucket, object_name, dest_path)
            print(f"✅ Downloaded s3://{self.bucket}/{object_name} → {dest_path}")
        except ClientError as e:
            print(f"❌ Lỗi Download S3: {e}")
            

    def download_file_cloudfront(self, object_key: str, dest_path: str, pbar=None) -> tuple:
        """Download 1 file thông qua CloudFront CDN với cơ chế Auto-Retry."""
        url = self.get_cloudfront_url(object_key)
        try:
            response = self.session.get(url, stream=True, timeout=15)
            response.raise_for_status()

            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=64 * 1024):  # 64KB
                    if chunk:
                        f.write(chunk)
                        if pbar:
                            pbar.update(len(chunk))

            return True, url
        except requests.exceptions.RequestException as e:
            return False, f"❌ Lỗi tải {url}: {e}"

    def download_folder_cloudfront(self, prefix: str, dest_folder: str, max_workers: int | None = None):
        """Download toàn bộ folder qua CloudFront bằng luồng song song (Lấy metadata từ S3)."""
        workers = max_workers or self.max_workers
        files = self.list_files(prefix)

        if not files:
            print(f"📂 Không tìm thấy file nào với prefix: {prefix}")
            return

        total_size = sum(f["Size"] for f in files)

        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading via CloudFront") as pbar, ThreadPoolExecutor(
            max_workers=workers
        ) as executor:
            futures = []
            for f in files:
                key = f["Key"]
                relative_path = key[len(prefix):] if prefix and key.startswith(prefix) else key
                local_path = os.path.join(dest_folder, relative_path)
                futures.append(executor.submit(self.download_file_cloudfront, key, local_path, pbar))

            for future in as_completed(futures):
                success, message = future.result()
                if not success:
                    print(message)

        print("✅ Hoàn tất tải toàn bộ folder qua CloudFront!")

    # ==========================================
    # 🔍 S3 UTILITIES & HELPERS
    # ==========================================

    def list_files(self, prefix: str = "") -> list:
        """Liệt kê tất cả file trong bucket (xử lý phân trang tự động nếu >1000 items)."""
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            files = []

            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        files.append({"Key": obj["Key"], "Size": obj["Size"]})

            print(f"📂 Tìm thấy {len(files)} files (Tổng dung lượng: {sum(f['Size'] for f in files)} bytes).")
            return files
        except ClientError as e:
            print(f"❌ Lỗi List Files: {e}")
            return []

    def delete_file(self, object_name: str):
        """Xóa 1 file khỏi S3 Bucket."""
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=object_name)
            print(f"🗑️ Đã xóa s3://{self.bucket}/{object_name}")
        except ClientError as e:
            print(f"❌ Lỗi Xóa File: {e}")

    def generate_presigned_url(self, object_name: str, expiry: int = 3600) -> str | None:
        """Tạo đường dẫn xem tạm thời (dùng nếu Bucket đặt Private)."""
        try:
            url = self.s3_client.generate_presigned_url(
                "get_object", Params={"Bucket": self.bucket, "Key": object_name}, ExpiresIn=expiry
            )
            print(f"🔗 Presigned URL (Hạn dùng {expiry}s): {url}")
            return url
        except ClientError as e:
            print(f"❌ Lỗi tạo Presigned URL: {e}")
            return None

    def get_neighbor_frames(self, current_key: str, before: int = 25, after: int = 25,
                            to_key: str | None = None) -> list:
        """N khung trước `current_key` .. N khung sau `to_key`, kèm mọi khung ở giữa.

        `to_key=None` (mặc định) = 1 mỏ neo, hành vi cũ. Hai mỏ neo dành cho ô "xem
        frames giữa 2 sự kiện" của temporal search, nơi cả hai hit luôn cùng video —
        nên khác video là lỗi của caller, raise chứ không im lặng.

        Trả CẢ mỏ neo để UI đặt nó ở giữa và đánh dấu `is_current`.

        KHÔNG paginate cả thư mục video: key S3 zero-pad theo frame nên thứ tự
        lexicographic cũng là thứ tự frame, dùng `StartAfter` đi thẳng tới vùng cần.

        ĐO THẬT trên bucket aic-bucket-2026 (từ VN, video 2352 object), xen kẽ 9 lần:

            StartAfter, 1 mỏ neo    p50 860ms   3 request (MaxKeys 1000/1000/26)
            StartAfter, 2 mỏ neo    p50 1174ms  3 request
            paginate cả video       p50 896ms   3 trang

        Tức trên corpus HIỆN TẠI hai cách BẰNG NHAU, không phải nhanh hơn. Lý do: video
        lớn nhất mới 2809 object (trung bình 1237) = đúng 3 trang, mà vòng `before` dưới
        đây cũng đọc 2 trang 1000. Thời gian là round-trip S3 (~280ms x 3), không phải
        số object. Giữ cách này vì nó không đắt hơn và chịu được video lớn hơn, chứ
        KHÔNG phải vì nó đang nhanh hơn.
        """
        prefix, current_frame_num, width, ext = self._parse_frame_key(current_key)
        if to_key:
            to_prefix, to_frame_num, _, _ = self._parse_frame_key(to_key)
            if to_prefix != prefix:
                raise ValueError("current_key và to_key phải cùng video")
        else:
            to_frame_num = current_frame_num
        # Không giả định thứ tự: chain temporal luôn đúng thứ tự nhưng min/max không tốn gì.
        lo_frame, hi_frame = sorted((current_frame_num, to_frame_num))

        def entries(response) -> list:
            found = []
            for obj in response.get("Contents", []):
                key = obj["Key"]
                item = re.fullmatch(r".*/(\d+)\.webp", key)
                if item:
                    found.append((int(item.group(1)), key))
            return found

        def key_at(frame: int) -> str:
            return f"{prefix}{frame:0{width}d}{ext}"

        def page(start_after: str, max_keys: int) -> list:
            return entries(self.s3_client.list_objects_v2(
                Bucket=self.bucket, Prefix=prefix,
                StartAfter=start_after, MaxKeys=max_keys))

        # frame -> key. Gom vào dict vì các đoạn dưới có thể chồng lấn nhau.
        found: dict = {}

        # ── Đoạn TRƯỚC lo_frame: nới cửa sổ số frame tới khi đủ `before` ──
        # Keyframe không cách đều nhau (7/shot, shot dài ngắn khác nhau) nên không suy
        # ra được khoảng cần đọc; 8x đủ cho sampling thường, nới gấp 4 nếu video thưa.
        if before:
            span = max(128, before * 8)
            for _ in range(8):
                anchor = max(0, lo_frame - span)
                got = [x for x in page(prefix if anchor == 0 else key_at(anchor), 1000)
                       if x[0] < lo_frame]
                found.update(got)
                if len(got) >= before or anchor == 0:
                    break
                span *= 4

        # ── Từ lo_frame trở đi ──
        # StartAfter loại chính nó nên phải lùi 1: key zero-pad cố định width nên
        # key_at(lo-1) luôn nằm ngay trước key_at(lo) theo lexicographic, dù frame
        # lo-1 không tồn tại thật.
        start_after = prefix if lo_frame == 0 else key_at(lo_frame - 1)
        if lo_frame == hi_frame:
            # Đường 1 mỏ neo: đúng 1 request nhỏ, y như trước khi có temporal search.
            found.update(page(start_after, max(after, 1) + 1))
        else:
            # Đường 2 mỏ neo: đọc tới khi vượt hi_frame và đủ `after` khung sau nó.
            # Chặn 20 vòng: khoảng cách bị TRAKE_MAX_GAP_SEC giới hạn nên thực tế 1-2
            # vòng là hết, vòng lặp có chặn chỉ để một key lạ không treo request.
            for _ in range(20):
                got = page(start_after, 1000)
                if not got:
                    break
                found.update(got)
                if sum(1 for f, _ in got if f > hi_frame) >= max(after, 1):
                    break
                start_after = got[-1][1]

        ordered = sorted(found.items())
        # Cắt theo VỊ TRÍ, không theo số học frame — xem docstring neighbor_slice_bounds.
        # Hàm này raise nếu mỏ neo không có trên S3, browse.py map thành HTTP 400.
        start, end = neighbor_slice_bounds([f for f, _ in ordered],
                                           lo_frame, hi_frame, before, after)
        return [k for _, k in ordered[start:end]]

    @staticmethod
    def _parse_frame_key(key: str) -> tuple:
        """(prefix, frame_num, width, ext). `width` cần để dựng lại key làm mốc
        StartAfter, `ext` để không hardcode .webp ở hai chỗ."""
        match = re.match(r"(.*/)(\d+)(\.webp)", key)
        if not match:
            raise ValueError(f"Định dạng key không hợp lệ: {key}")
        prefix, frame_str, ext = match.groups()
        return prefix, int(frame_str), len(frame_str), ext


# ==========================================
# 🧪 SỬ DỤNG VÍ DỤ (EXAMPLE USAGE)
# ==========================================
if __name__ == "__main__":
    # Khởi tạo instance
    aws = AWSStorageHelper(
        bucket_name="aic-bucket-2026",
        cloudfront_domain="",
    )

    # 1. Upload 1 file lên S3
    aws.upload_file("D:\\Pictures\\kyyeu.jpg", object_name="test_upload_2/kyyeu.jpg")

    # 2. Download 1 file từ S3 về máy
    aws.download_file_s3("test_upload_2/kyyeu.jpg", "D:\\Pictures\\kyyeu2.jpg")
