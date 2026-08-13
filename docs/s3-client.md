# Hướng dẫn dùng S3 client

File đang dùng: [services/be/src/app/clients/s3.py](../services/be/src/app/clients/s3.py)

> CloudFront hiện chưa hoạt động vì biến `CLOUDFRONT_DOMAIN` đang rỗng. Vì vậy cách dùng ổn định nhất là upload/download trực tiếp qua S3.

## 1. Setup env

Trong file `.env` ở root project:

```bash
AWS_ACCESS_KEY=your_access_key
AWS_SECRET_KEY=your_secret_key
AWS_REGION=ap-southeast-1
AWS_BUCKET_NAME=aic-bucket-2026
CLOUDFRONT_DOMAIN=
```

Khởi tạo client:

```python
from app.clients.s3 import AWSStorageHelper

aws = AWSStorageHelper(
    bucket_name="aic-bucket-2026",
    region="ap-southeast-1",
    cloudfront_domain="",
)
```

---

## 2. Upload ảnh lên S3

### Upload 1 file

```python
url = aws.upload_file(
    file_path="D:/Pictures/kyyeu.jpg",
    object_name="test_upload_2/kyyeu.jpg",
)
print(url)
```

Trả về URL dạng:

```text
https://aic-bucket-2026.s3.ap-southeast-1.amazonaws.com/test_upload_2/kyyeu.jpg
```

### Upload folder

```python
aws.upload_folder(
    local_folder="D:/Pictures/test_images",
    s3_prefix="gallery/demo",
)
```

### Upload file đơn lẻ qua `upload_folder`

```python
aws.upload_folder(
    local_folder="D:/Pictures/kyyeu.jpg",
    s3_prefix="test_upload_2",
)
```

Hàm này đã được sửa để nhận cả file đơn lẻ. Trước đó, nếu truyền ảnh trực tiếp thì code cũ chỉ `os.walk()` nên báo "Thư mục trống hoặc không tìm thấy file".

### Upload nhiều file

```python
files = [
    ("D:/Pictures/a.jpg", "images/a.jpg"),
    ("D:/Pictures/b.jpg", "images/b.jpg"),
]
print(aws.upload_many(files, max_workers=8))
```

---

## 3. Download từ S3

```python
aws.download_file_s3(
    object_name="test_upload_2/kyyeu.jpg",
    dest_path="D:/Downloads/kyyeu.jpg",
)
```

Đây là cách download ổn định nhất hiện nay.

---

## 4. Các hàm khác

```python
aws.list_files(prefix="test_upload_2/")
aws.delete_file("test_upload_2/kyyeu.jpg")
url = aws.generate_presigned_url("test_upload_2/kyyeu.jpg", expiry=3600)
```

- `list_files`: liệt kê object trong bucket
- `delete_file`: xóa file
- `generate_presigned_url`: tạo URL tạm thời cho object private

---

## 5. CloudFront

Các hàm CloudFront như:

```python
aws.get_cloudfront_url("test_upload_2/kyyeu.jpg")
aws.download_file_cloudfront("test_upload_2/kyyeu.jpg", "D:/Downloads/x.jpg")
```

sẽ lỗi nếu `CLOUDFRONT_DOMAIN` chưa được set:

```text
ValueError: CloudFront domain chưa được cấu hình!
```

Cần set:

```bash
CLOUDFRONT_DOMAIN=d123abcxyz.cloudfront.net
```

Nếu chưa có domain CloudFront, hãy dùng S3 trực tiếp.

---

## 6. Ví dụ nhanh

```python
from app.clients.s3 import AWSStorageHelper

aws = AWSStorageHelper(bucket_name="aic-bucket-2026", region="ap-southeast-1")
aws.upload_file("D:/Pictures/kyyeu.jpg", object_name="test_upload_2/kyyeu.jpg")
aws.download_file_s3("test_upload_2/kyyeu.jpg", "D:/Pictures/kyyeu2.jpg")
```

Đây là cách dùng chuẩn cho việc upload/download ảnh trong project hiện tại.
