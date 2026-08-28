"""Load prompt.yaml as the system prompt, load model/key from .env, run one query through litellm.

Edit `query` below and re-run: .venv/Scripts/python.exe run_prompt.py
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import litellm

sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1252, can't
                                          # print Vietnamese diacritics without this

REPO_ROOT = Path(__file__).parent

load_dotenv(REPO_ROOT / ".env")

# Prompt sống trong package BE (nó phải ship trong container). Harness này đọc CHÍNH file
# đó, không phải bản sao — hai bản trôi lệch nhau là cách âm thầm nhất để "chạy tay thì
# đúng, chạy trong BE thì sai".
system_prompt = (REPO_ROOT / "services" / "be" / "src" / "app" / "services"
                 / "segment_prompt.txt").read_text(encoding="utf-8")

model = os.environ["LLM_MODEL"]
api_key = os.environ.get("LLM_API_KEY") or None  # None -> litellm falls back to the
                                                  # provider's own env var (e.g. CEREBRAS_API_KEY)

# ---- edit this ----
queries = [
    "Có thể thấy trong cảnh quay có 4 tài xế xe ôm công nghệ trong trạm xăng, trong đó 3 người đứng đợi còn 1 người lái xe từ trái sang phải khung hình. Trước đó là cảnh một người đậy nắp bình xăng xe máy của họ.",
    "Cảnh quay một nhóm hơn 5 người xếp thành hàng tập thể dục, cùng thực hiện động tác hai tay chạm mũi chân. Trong nhóm chỉ có một người đeo kính và ba người đội nón có màu đỏ.",
    "Đoạn phim bắt đầu bằng một bản đồ, trên đó một loại công trình thủy lợi lần lượt xuất hiện bốn lần. Sau đó chuyển sang cảnh một con đập được quay từ trên cao, tiếp đến là cảnh cận con đập dưới trời mưa.",
    "Hình ảnh một con cá được đặt lên cân, sau đó có cảnh một con cá khác cùng loại bị một người cầm đuôi. Con số hiển thị cuối cùng trên cân là bao nhiêu?",
    "Một đàn sư tử đang nghỉ ngơi và leo trèo trên các bục gỗ trong khu nuôi dưỡng, phía trước có bảng thông tin của London Zoo phục vụ công tác theo dõi và bảo tồn động vật.. Sau đó có cảnh hai nhân viên mặc áo xanh lá đang cân và ghi nhận số liệu của một con vật trong khuôn viên sở thú.",
    "Đoạn clip bắt đầu bằng việc đậu hà lan được bỏ vào với mực đang được xào trên chảo, bên cạnh là đĩa hành tây và ớt đỏ thái lát chuẩn bị cho vào món ăn. Đoạn clip kết thúc với khung quay chậm (slow motion) cảnh lắc chảo trên bếp lửa.",
    "Mẩu tin bắt đầu với hình ảnh nột người đàn ông mặc vest xanh đậm, sơ mi trắng và cà vạt, đang ngồi trên một chiếc ghế lớn. Ông cầm bằng hai tay một khối đá quý thô khá lớn, đưa lên gần mặt để quan sát. Bên phải là một phụ nữ mặc trang phục công sở màu đen và khăn trùm đầu màu hồng tím, đang đứng cạnh và mỉm cười. Tiếp theo có hình ảnh toàn cảnh từ trên cao của một mỏ đá quý lộ thiên quy mô lớn với hố khai thác sâu nhiều tầng và hệ thống đường vận chuyển bao quanh.",
    "Đoạn clip bắt đầu bằng cảnh cà rốt cắt hình ngôi sao đang được luộc trong nồi nước sôi, đặt trong rổ lưới kim loại và được đảo bằng đôi đũa gỗ. Đoạn clip kết thúc bằng hình ảnh đĩa rau củ luộc và đồ chiên được trình bày đẹp mắt, gồm đậu bắp, súp lơ, cà rốt hình ngôi sao, bí xanh, chén nước chấm màu hồng ở giữa và đôi đũa màu hồng nhạt đặt bên phải",
    "Người đầu bếp lần lượt đặt các miếng nguyên liệu dạng thanh và những lát cắt hình hoa vào một đĩa đang được hấp trong nồi. Các nguyên liệu được dùng đũa sắp xếp xen kẽ xung quanh phần thức ăn đã có sẵn trên đĩa. Sau đó, đầu bếp dùng muỗng lấy thêm một loại nguyên liệu mềm từ tô thủy tinh. Phần nguyên liệu này được đặt vào giữa đĩa, xung quanh là các miếng dạng thanh và hình hoa đã được sắp xếp trước đó.",
    "Đoạn phim ghi lại cảnh những chiếc xe ô tô lội nước, chiếc xe màu vàng, màu đỏ và màu đen lần lượt chuẩn bị đi qua cầu. Con số được ghi trên biển báo bên trái của cây cầu là bao nhiêu?",
    "Hành động cắt chùm nho bằng kéo từ giàn nho bằng một chiếc kéo màu đen. Có thể thấy có một sợi dây màu xanh dương được buộc vào cuống của chùm nho này trước khi nó được cắt."   
]
# -------------------

for query in queries:
    resp = litellm.completion(
        model=model,
        api_key=api_key,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
    )

    print(f"query: {query}")
    print(f"model: {model}")
    print(f"finish_reason: {resp.choices[0].finish_reason}")
    print(f"usage: {resp.usage}")
    print("---")
    print(resp.choices[0].message.content)
    print("=" * 80)
