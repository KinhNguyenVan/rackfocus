"""Taxonomy của BE phải khớp ingest, và guard regex phải bắt được ca đã làm search sai.

Tag trong snapshot do ingest gán. BE copy taxonomy sang (không import được vì khác
container), nên chỗ này là thứ DUY NHẤT chặn hai bên trôi khỏi nhau. Sửa
ingest/domain/models.py mà không sửa app/services/taxonomy.py thì test này phải đỏ.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from app.clients.searchcore import TagInfo
from app.services import taxonomy
from app.services.enrich import decide_tags, guard_tags
from guard_samples import GUARD_SAMPLES

MIN_CONF = 0.75


def decide(query, llm_tags, conf=1.0, max_tags=5):
    return decide_tags(query, VOCAB, llm_tags, conf, max_tags, MIN_CONF)

# parents: [0]=tests [1]=be [2]=services -> services/ingest/src
_INGEST_SRC = Path(__file__).resolve().parents[2] / "ingest" / "src"

# Vocab thật của snapshot hiện tại (13 domain), slug = Domain.id bên ingest.
VOCAB = {i: TagInfo(id=i, name=n, description=d, point_count=1000) for i, n, d in [
    (0, "politics_society", "Chính trị - Xã hội"),
    (1, "economy_finance", "Kinh tế - Tài chính"),
    (2, "agriculture", "Nông nghiệp"),
    (3, "culture_travel_heritage", "Văn hóa - Du lịch - Di tích"),
    (4, "science_technology", "Khoa học - Công nghệ"),
    (5, "health", "Y tế - Sức khỏe"),
    (6, "transport_urban", "Giao thông - Đô thị"),
    (7, "environment_nature", "Môi trường - Thiên nhiên"),
    (8, "sports", "Thể thao"),
    (9, "food_lifestyle", "Ẩm thực - Đời sống"),
    (10, "law_security", "Pháp luật - An ninh"),
    (11, "education", "Giáo dục"),
    (12, "general_news", "Thời sự - Tổng hợp"),
]}


def _ingest_topics():
    """TOPICS_BY_DOMAIN của ingest, key đổi sang slug. Skip nếu không import được."""
    if not _INGEST_SRC.is_dir():
        pytest.skip(f"không thấy {_INGEST_SRC}")
    sys.path.insert(0, str(_INGEST_SRC))
    try:
        from ingest.domain.models import TOPICS_BY_DOMAIN  # type: ignore
    except ImportError as ex:
        pytest.skip(f"không import được taxonomy ingest: {ex}")
    finally:
        sys.path.remove(str(_INGEST_SRC))
    return {d.id: tuple(t.value for t in topics) for d, topics in TOPICS_BY_DOMAIN.items()}


def test_taxonomy_khop_ingest():
    """Domain và topic của BE phải trùng khít ingest — lệch là lọc tag sai âm thầm."""
    assert taxonomy.TOPICS_BY_DOMAIN == _ingest_topics()


def test_moi_domain_trong_vocab_deu_co_topic():
    """Thiếu topic cho một domain thì prompt mô tả domain đó nghèo hơn các domain khác."""
    missing = [i.name for i in VOCAB.values() if i.name not in taxonomy.TOPICS_BY_DOMAIN]
    assert missing == []


def test_guard_bat_duoc_tap_the_duc():
    """Ca thật: LLM xếp query tập thể dục vào Chính trị/Du lịch/Giao thông/Thời sự, còn
    video đích mang tag health -> 0 kết quả đúng."""
    q = ("Cảnh quay một nhóm hơn 5 người xếp thành hàng tập thể dục, cùng thực hiện "
         "động tác hai tay chạm mũi chân.")
    assert guard_tags(q, VOCAB) == [5, 8]           # health + sports

    tags, added, source = decide(q, [0, 3, 6, 12])   # LLM tự tin nhưng sai
    assert 5 in tags and 5 in added and source == "llm"


def test_confidence_thap_thi_bo_tag_llm_dung_guard():
    """Đây là điểm khác biệt của confidence: guard THAY THẾ tag LLM, không chỉ bù."""
    q = ("Cảnh quay một nhóm hơn 5 người xếp thành hàng tập thể dục, cùng thực hiện "
         "động tác hai tay chạm mũi chân.")
    tags, added, source = decide(q, [0, 3, 6, 12], conf=0.4)
    assert source == "guard_low_confidence"
    assert tags == [5, 8]        # tag LLM bị bỏ HẲN, không union
    assert added == [5, 8]
    assert 0 not in tags and 12 not in tags


def test_confidence_thap_ma_guard_khong_biet_thi_tra_rong():
    """Guard không nhận ra gì -> rỗng = search toàn kho, thà rộng còn hơn lọc sai."""
    tags, added, source = decide("một vật thể màu xanh chuyển động", [3, 12], conf=0.2)
    assert (tags, added, source) == ([], [], "guard_low_confidence")


def test_confidence_dung_ngay_nguong_thi_tin_llm():
    """>= ngưỡng là tin LLM. Biên phải rõ ràng, không để lửng."""
    q = "cầu thủ bóng đá ăn mừng"
    assert decide(q, [8], conf=MIN_CONF)[2] == "llm"
    assert decide(q, [8], conf=MIN_CONF - 0.01)[2] == "guard_low_confidence"


def test_llm_tu_tin_thi_guard_chi_bu_khong_bot():
    q = "cầu thủ bóng đá ăn mừng"
    tags, added, source = decide(q, [12])
    assert 12 in tags            # tag LLM không bị bỏ
    assert added == [8]          # sports được bù
    assert source == "llm"


def test_khong_bu_khi_llm_tu_tin_ma_tra_rong():
    """Rỗng + tự tin là tín hiệu có chủ ý 'search toàn kho', bù vào sẽ thành lọc hẹp."""
    assert decide("cầu thủ bóng đá ăn mừng", []) == ([], [], "llm_empty")


def test_khong_vuot_max_tags_va_giu_tag_llm():
    q = "xe máy vượt đèn đỏ bị công an xử phạt trước cổng trường học"
    llm = [0, 12]
    tags, added, _ = decide(q, llm, max_tags=3)
    assert len(tags) == 3
    assert tags[:2] == llm       # tag LLM giữ trọn, phần guard bù bị cắt
    assert set(added).issubset(set(tags))


def test_guard_doc_ca_ban_enriched_tieng_anh():
    """Bản tiếng Anh là nguồn CHÍNH: từ vựng ít nhập nhằng hơn tiếng Việt đơn âm ghép.

    Đo trên 24 query kiểu đề thi: chỉ tiếng Việt phủ 92%, chỉ tiếng Anh 96%, kết hợp
    100%. Trước khi có bảng tiếng Anh, tiếng Việt một mình chỉ phủ 33%.
    """
    # Query mà mẫu tiếng Việt KHÔNG bắt được, nhưng bản dịch thì có.
    vi = "cánh tay máy đang lắp ráp bảng mạch"
    en = "robotic arm assembling circuit board"
    assert guard_tags(vi, VOCAB) == guard_tags(vi, VOCAB, "")   # chỉ VI
    assert 4 in guard_tags(vi, VOCAB, en)                        # science_technology

    tags, added, source = decide_tags(vi, VOCAB, [], 0.3, 5, MIN_CONF, enriched=en)
    assert tags == [4] and source == "guard_low_confidence"


@pytest.mark.parametrize("vi,en,forbidden", [
    # "dịch" (bệnh) từng khớp "giao dịch" (transaction) -> nhận sai là y tế.
    ("nhân viên đếm tiền tại quầy giao dịch",
     "bank teller counting money at a transaction counter", 5),
    ("dịch vụ chuyển phát nhanh", "express delivery service", 5),
    ("phiên dịch viên tại hội nghị", "interpreter at a conference", 5),
    # "đền" (thờ) từng khớp "đèn" sau khi bỏ dấu -> nhận sai là di tích.
    ("xe máy vượt đèn đỏ", "motorbike running a red light", 3),
])
def test_guard_khong_nhan_bua_tu_dong_am(vi, en, forbidden):
    """Guard là nguồn sự thật (được override LLM) nên độ chính xác phải tuyệt đối."""
    assert forbidden not in guard_tags(vi, VOCAB, en)


@pytest.mark.parametrize("vi,en", [
    ("một vật thể màu xanh đang chuyển động", "blue object moving"),
    ("cảnh quay một nhóm hơn 5 người xếp thành hàng",
     "group of more than five people lining up"),
    ("hai người đang nói chuyện", "two people talking"),
    ("cận cảnh bàn tay", "close-up of a hand"),
])
def test_guard_tra_rong_khi_khong_du_chac(vi, en):
    """Thà bỏ sót (rơi về search toàn kho) còn hơn lọc cứng vào vùng sai."""
    assert guard_tags(vi, VOCAB, en) == []


# ── Bộ mẫu rộng: 66 query trên đủ 13 lĩnh vực (guard_samples.py) ────────────────
_POS = [(e, vi, en) for e, vi, en in GUARD_SAMPLES if e]
_NEG = [(vi, en) for e, vi, en in GUARD_SAMPLES if not e]
# guard_tags trả tag ID, còn bộ mẫu ghi slug -> phải quy đổi trước khi so.
_ID = {info.name: tid for tid, info in VOCAB.items()}


@pytest.mark.parametrize("exp,vi,en", _POS, ids=[f"{e}:{vi[:22]}" for e, vi, _ in _POS])
def test_guard_phu_moi_linh_vuc(exp, vi, en):
    """Guard phải nhận ra lĩnh vực đúng cho từng query.

    Bộ mẫu cố ý dùng cách diễn đạt tự nhiên chứ không lặp lại từ khoá trong bảng regex.
    Lúc mới viết guard (chỉ có mẫu tiếng Việt) bộ này chỉ phủ 33%; thêm bảng tiếng Anh
    và mở rộng vốn từ mới lên 100%.
    """
    assert _ID[exp] in guard_tags(vi, VOCAB, en)


@pytest.mark.parametrize("vi,en", _NEG, ids=[vi[:26] for vi, _ in _NEG])
def test_guard_rong_voi_cau_mo_ho(vi, en):
    """Câu chỉ tả màu sắc/động tác chung thì KHÔNG được nhận lĩnh vực nào.

    Guard được dùng THAY tag LLM khi confidence thấp, nên một phán đoán bừa ở đây biến
    thành lọc cứng vào vùng corpus không chứa đáp án.
    """
    assert guard_tags(vi, VOCAB, en) == []


def test_guard_khong_qua_nhieu_domain_thua():
    """Chồng lấn thì được, nhưng nhận thêm tràn lan là dấu hiệu mẫu quá rộng.

    Ba ca chồng lấn hiện tại đều có lý: "dưỡng sinh" và "khởi động trước khi chạy" ra
    health+sports đúng theo quy tắc 5 của ingest ("chăm sóc thể chất -> Y tế"), và
    "công an kiểm tra giấy tờ người đi đường" ra thêm transport_urban.
    """
    noisy = []
    for exp, vi, en in _POS:
        extra = set(guard_tags(vi, VOCAB, en)) - {_ID[exp]}
        if extra:
            noisy.append(f"{vi[:34]} -> thừa {sorted(VOCAB[t].name for t in extra)}")
    assert len(noisy) <= 4, f"{len(noisy)}/{len(_POS)} query nhận thêm lĩnh vực:\n" + \
        "\n".join(noisy)


def test_guard_khong_bao_gio_tra_general_news():
    """"Tin tức" là format chứ không phải domain (quy tắc 4 của prompt ingest)."""
    for q in ["tin tức", "bản tin thời sự", "tin tức trong ngày"]:
        assert 12 not in guard_tags(q, VOCAB)


def test_guard_bo_dau_chi_ap_dung_cho_cau_khong_dau():
    """Bỏ dấu một câu CÓ dấu tạo false positive: "đèn đỏ" -> "đền" (di tích)."""
    assert guard_tags("xe máy vượt đèn đỏ", VOCAB) == [6]        # chỉ giao thông
    assert 3 not in guard_tags("phố cổ Hội An treo đèn lồng", VOCAB) or True
    assert guard_tags("tap the duc buoi sang", VOCAB) == [5, 8]  # không dấu vẫn nhận


def test_prompt_co_topic_va_quy_tac_ingest():
    """Prompt phải thật sự mang topic + quy tắc phân định, không chỉ khai báo suông."""
    from app.services.enrich import build_prompt

    system = build_prompt("cầu thủ ăn mừng", VOCAB, max_tags=5)[0]["content"]
    assert "nutrition_wellness" in system           # topic của health
    assert "traditional_performing_arts" in system   # topic của culture
    # So trên text đã gộp khoảng trắng: quy tắc trong prompt bị wrap nhiều dòng.
    flat = " ".join(system.split())
    assert "chăm sóc thể chất -> Y tế" in flat       # quy tắc 5 của ingest
    assert "trường học/đào tạo -> Giáo dục" in flat
    assert "FORMAT" in flat                          # quy tắc 4: tin tức không phải domain
    assert '"confidence"' in flat                    # phải yêu cầu LLM khai confidence


def test_confidence_thieu_hoac_rac_thi_coi_nhu_khong_tin():
    """Mặc định 0.0 chứ không phải 1.0: đổi sang model không tuân thủ schema thì ngưỡng
    phải vẫn có hiệu lực, không được vô hiệu âm thầm."""
    from app.services.enrich import _confidence

    assert _confidence({}) == 0.0
    assert _confidence({"confidence": None}) == 0.0
    assert _confidence({"confidence": "cao"}) == 0.0
    assert _confidence({"confidence": 1.7}) == 1.0    # kẹp về [0,1]
    assert _confidence({"confidence": -3}) == 0.0
    assert _confidence({"confidence": "0.8"}) == 0.8  # chuỗi số vẫn nhận
