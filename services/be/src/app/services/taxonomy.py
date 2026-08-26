"""Taxonomy domain/topic + guard regex, ĐỒNG BỘ với ingest.

Tag trong snapshot là `domain_id` do `ingest/domain/` gán lúc offline. Nếu prompt
query-time ở BE mô tả domain khác cách ingest mô tả, hai bên phân loại lệch nhau và
lọc tag trở thành CỨNG-mà-SAI: frame đúng nằm ngoài tag đã chọn là không thể với tới.

Ví dụ thật đã gặp: video tập thể dục L30_V046 được ingest gán `health`, vì prompt
ingest có quy tắc "bệnh/chăm sóc thể chất -> Y tế". Prompt BE trước đây không có quy
tắc đó nên không có gì bảo đảm nó cũng chọn `health`.

Vì vậy file này COPY taxonomy + quy tắc phân định từ
`services/ingest/src/ingest/domain/{models,enricher}.py`. Không import trực tiếp được
vì ingest là service/container riêng, không có trong image của BE.

QUAN TRỌNG: sửa taxonomy bên ingest thì PHẢI sửa cả đây.
`tests/test_taxonomy.py` so hai bên và fail nếu lệch — đừng bỏ test đó.
"""
from __future__ import annotations

import re
import unicodedata

# ── Topic theo domain — khớp TOPICS_BY_DOMAIN của ingest ─────────────────────────
# Key là domain slug (Domain.id bên ingest = tên enum lowercase), cũng chính là
# `TagInfo.name` mà core trả về trong tag_vocab -> nối được với tag_id.
TOPICS_BY_DOMAIN: dict[str, tuple[str, ...]] = {
    "politics_society": ("public_policy", "public_administration", "labor_social_welfare",
                         "diplomacy", "community_social_issues", "other"),
    "economy_finance": ("banking_interest_rates", "real_estate", "trade_exports",
                        "business_markets", "consumer_prices", "other"),
    "agriculture": ("crop_farming", "livestock_aquaculture", "agricultural_exports",
                    "rural_development", "other"),
    "culture_travel_heritage": ("heritage_historical_sites", "tourism_destinations",
                                "arts_entertainment", "festivals_traditions",
                                "traditional_performing_arts", "other"),
    "science_technology": ("artificial_intelligence", "robotics_automation", "biotechnology",
                           "space_research", "digital_technology", "other"),
    "health": ("disease_prevention", "healthcare_services",
               "pharmaceuticals_medical_devices", "public_health", "nutrition_wellness",
               "other"),
    "transport_urban": ("road_accident", "traffic_violation", "public_transport",
                        "transport_infrastructure", "traffic_congestion", "other"),
    "environment_nature": ("extreme_weather", "natural_disaster", "conservation_wildlife",
                           "pollution_waste", "climate_environment", "other"),
    "sports": ("football", "cycling", "combat_sports", "athletics", "other_sports",
               "other"),
    "food_lifestyle": ("food_cuisine", "consumer_lifestyle", "family_daily_life",
                       "fashion_beauty", "other"),
    "law_security": ("crime_investigation", "court_justice", "law_enforcement",
                     "public_security", "other"),
    "education": ("schools_education", "exams_admissions", "skills_training",
                  "education_policy", "other"),
    "general_news": ("breaking_news", "mixed_news_digest", "human_interest", "other"),
}

# ── Quy tắc phân định — trích từ build_system_prompt() của ingest ────────────────
# Chỉ giữ những quy tắc quyết định CHỌN DOMAIN NÀO (quy tắc 4, 5, 7). Bỏ các quy tắc
# về cách cắt đoạn theo scene (1, 2, 3, 6) vì query-time không phân đoạn gì.
DISAMBIGUATION_RULES = """- "Tin tức" là FORMAT, không phải domain. Chỉ dùng "Thời sự - Tổng hợp" cho bản tin
  hỗn hợp hoặc khi thật sự không có domain cụ thể nào phù hợp.
- Ưu tiên khi giao thoa: mùa vụ/canh tác -> Nông nghiệp; giá/ngân hàng/doanh nghiệp ->
  Kinh tế; hạ tầng/di chuyển -> Giao thông; tội phạm/xử phạt -> Pháp luật; bệnh/chăm
  sóc thể chất -> Y tế; trường học/đào tạo -> Giáo dục.
- Múa lân biểu diễn, tập luyện hoặc truyền thống -> "Văn hóa - Du lịch - Di tích".
  Chỉ dùng "Thể thao" khi nội dung nhấn mạnh giải đấu có chấm điểm hoặc xếp hạng."""


def _strip_diacritics(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip().casefold()


# ─────────────────────────────────────────────────────────────────────────────────
# Guard regex
#
# Guard KHÔNG phải bộ lọc hallucination mà là NGUỒN SỰ THẬT: khi LLM khai confidence
# thấp, tag của guard được dùng THAY cho tag của LLM (xem enrich.decide_tags). Vì vậy
# nguyên tắc là ĐỘ CHÍNH XÁC TUYỆT ĐỐI: chỉ nhận từ mà nếu xuất hiện thì gần như chắc
# chắn thuộc domain đó. Thà bỏ sót (rơi về search toàn kho, chậm nhưng đúng) còn hơn
# nhận sai (lọc cứng vào vùng không chứa đáp án).
#
# Khớp trên HAI nguồn:
#   - bản enriched TIẾNG ANH (chính): LLM chuẩn hoá về từ vựng ngắn gọn, ít nhập nhằng.
#     Đo trên 24 query kiểu đề thi: tiếng Anh cho "transaction counter", "tractor",
#     "robotic arm", "blood pressure", "traffic jam", "courtroom", "blackboard" —
#     toàn từ một nghĩa.
#   - query gốc TIẾNG VIỆT (dự phòng): giữ để guard còn hoạt động khi enrich lỗi
#     (lúc đó enriched_text chính là query gốc) hoặc khi LLM viết lại thiếu.
#
# Tiếng Việt khó hơn hẳn vì đơn âm ghép: từ ngắn nằm lọt trong từ khác. Đã trúng thật
# hai lần — "dịch" (bệnh) khớp "giao dịch", "đền" (thờ) khớp "đèn" sau khi bỏ dấu. Nên
# mẫu tiếng Việt chỉ dùng cụm >= 2 âm tiết, hoặc từ không có đồng âm trong ngữ cảnh khác.
# ─────────────────────────────────────────────────────────────────────────────────

# general_news CỐ Ý không có mẫu nào ở cả hai bảng: theo quy tắc 4 của ingest, "tin tức"
# là format chứ không phải domain, để guard thêm nó vào là làm tập ứng viên phình vô ích.

_EN_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # health — "exercise/workout" vào CẢ health lẫn sports theo quy tắc 5 của ingest
    # ("chăm sóc thể chất -> Y tế"), đây chính là ca L30_V046.
    (r"\bexercis(e|es|ing)\b|\bwork(ing )?out\b|\bcalisthenics\b|\bdư[oỡ]ng sinh\b"
     r"|\bstretch(ing|es)?\b|\bwarm[- ]?up\b|\baerobics\b|\bmorning gymnastics\b",
     ("health", "sports")),
    (r"\bdoctor\b|\bnurse\b|\bhospital\b|\bpatient\b|\bblood pressure\b|\bstethoscope\b"
     r"|\bvaccin(e|ation)\b|\binjection\b|\bsurger(y|ies)\b|\bpharmac(y|ist)\b|\bclinic\b"
     r"|\bambulance\b|\bmedical (mask|staff|equipment)\b|\bwheelchair\b|\bIV drip\b"
     r"|\bhealth ?care\b|\bepidemic\b|\bnutrition\b",
     ("health",)),
    (r"\bathlete\b|\bfootball\b|\bsoccer\b|\bstadium\b|\bfinish line\b|\bmedal\b"
     r"|\bmarathon\b|\bfighter(s)?\b|\bbox(ing|er)\b|\breferee\b|\bjersey\b|\btournament\b"
     r"|\bmatch (score|referee)\b|\bswimming pool\b|\bcycling race\b|\bmartial arts\b"
     r"|\bgoalkeeper\b|\bpodium\b|\bcyclist(s)?\b|\bfinish(ing)? line\b",
     ("sports",)),
    (r"\bteacher\b|\bstudent(s)?\b|\bpupil(s)?\b|\bclassroom\b|\bblackboard\b|\bchalk\b"
     r"|\bschool\b|\bexam(s|ination)?\b|\blecture\b|\buniversity\b|\bdesk(s)? in a row\b"
     r"|\bgraduation\b|\bschoolyard\b",
     ("education",)),
    (r"\bpagoda\b|\btemple\b|\bshrine\b|\btourist(s)?\b|\bfestival\b|\blantern(s)?\b"
     r"|\bsedan chair\b|\b[aá]o d[aà]i\b|\b[aá]o the\b|\bkh[aă]n x[eế]p\b|\bincense\b"
     r"|\bancient (tower|town|house)\b|\bmonument\b|\bmausoleum\b|\bheritage\b"
     r"|\blion dance\b|\bl[aâ]n (dance|dancing)\b|\bsedan\b"
     r"|\btraditional (dance|costume|music|instrument)\b|\bcalligraphy\b",
     ("culture_travel_heritage",)),
    (r"\btraffic jam\b|\boverpass\b|\bhighway\b|\bmotor(bike|cycle)\b|\btruck\b|\bbus\b"
     r"|\blicense plate\b|\broundabout\b|\bintersection\b|\btraffic light\b|\bhelmet\b"
     r"|\btoll (booth|gate)\b|\bcrosswalk\b|\brush hour\b|\bparking lot\b"
     r"|\b(car|truck|bus|motorbike) (crash|accident|overturned)\b|\bpedestrian(s)?\b"
     r"|\bpassenger stop\b|\bzebra crossing\b",
     ("transport_urban",)),
    (r"\btractor\b|\bharvest(ing|er)?\b|\bpaddy\b|\brice field\b|\bsickle(s)?\b"
     r"|\blivestock\b|\bbuffalo\b|\bplough|\bplow\b|\bfertilizer\b|\borchard\b"
     r"|\baquaculture\b|\bshrimp (farm|pond|processing)\b|\bsorting shrimp\b"
     r"|\bfarmer(s)?\b|\bgreenhouse\b|\bcattle\b|\bcow(s)?\b|\bherd of\b"
     r"|\bpoultry\b|\bseedling(s)?\b|\bpesticide\b|\bdirt road\b",
     ("agriculture",)),
    (r"\bbank teller\b|\bbanknote(s)?\b|\bcounting money\b|\btransaction counter\b"
     r"|\bstock (price|market|chart)\b|\binterest rate\b|\breal estate\b|\bATM\b"
     r"|\bcashier\b|\binvoice\b|\bcurrency\b|\bgold bar(s)?\b|\btrading floor\b"
     r"|\bfactory (production|assembly) line\b|\bexport container\b",
     ("economy_finance",)),
    (r"\bpolice\b|\bcourtroom\b|\bhandcuff(s|ed)?\b|\bprison\b|\barrest(ed|ing)?\b"
     r"|\btrial\b|\bjudge\b|\bprosecutor\b|\bv[aà]nh m[oó]ng ng[uự]a\b|\bpolice officer\b"
     r"|\bcrime scene\b|\bevidence bag\b|\bsmuggl(ing|ed|er|ers)\b|\bnarcotics\b"
     r"|\bdrug (paraphernalia|bust|seizure|trafficking)\b|\blaw enforcement\b"
     r"|\bchecking documents\b",
     ("law_security",)),
    (r"\bflood(ing|ed|s)?\b|\bstorm\b|\btyphoon\b|\blandslide\b|\bforest\b|\bwildlife\b"
     r"|\bmacaque(s)?\b|\bmonkey(s)?\b|\bpollution\b|\bgarbage\b|\bwaste dump\b"
     r"|\bmangrove\b|\bcoral\b|\bwaterfall\b|\bcliff\b|\bdrought\b|\bwildfire\b"
     r"|\bendangered\b|\bnature reserve\b",
     ("environment_nature",)),
    (r"\brobot(ic)?\b|\bcircuit board\b|\bsemiconductor\b|\bsatellite\b|\blaborator(y|ies)\b"
     r"|\blab coat\b|\bvirtual reality\b|\bVR headset\b|\bartificial intelligence\b"
     r"|\bmicroscope\b|\bserver (rack|room)\b|\bdata cent(er|re)\b|\b3D print(er|ing)\b"
     r"|\btest tube(s)?\b|\bsolar panel(s)?\b",
     ("science_technology",)),
    (r"\bph[oở]\b|\bnoodle(s)?\b|\bstreet vendor\b|\brestaurant\b|\bcook(ing|s)?\b"
     r"|\bkitchen\b|\bmarket stall\b|\bchef\b|\bstreet food\b|\bb[aá]nh m[iì]\b"
     r"|\bfashion show\b|\bhair salon\b|\bwet market\b|\bdining table\b|\bfood tray\b",
     ("food_lifestyle",)),
    (r"\bparliament\b|\bnational assembly\b|\bward office\b|\bgovernment official(s)?\b"
     r"|\bdiplomat(ic)?\b|\bpress conference\b|\bsigning ceremony\b|\bdelegation\b"
     r"|\bpolic(y|ies) (meeting|announcement)\b|\bcommune (office|committee)\b"
     r"|\bpaperwork\b|\bcivil servant(s)?\b|\bvot(e|es|ing)\b|\bballot\b"
     r"|\brepresentative(s)? (pressing|voting)\b|\bassembly hall\b",
     ("politics_society",)),
)

_VI_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (r"\bth[eể] d[uụ]c\b|\bt[aậ]p luy[eệ]n\b|\bkh[oở]i đ[oộ]ng\b|\bdư[oỡ]ng sinh\b"
     r"|\bth[eể] d[uụ]c bu[oổ]i s[aá]ng\b|\bgi[aã]n c[oơ]\b",
     ("health", "sports")),
    # "dịch" một âm CỐ Ý không có mẫu: nó lọt trong "giao dịch", "dịch vụ", "phiên dịch".
    (r"\bb[eệ]nh vi[eệ]n\b|\bb[aá]c s[iĩ]\b|\by t[eế]\b|\bs[uứ]c kh[oỏ]e\b|\bb[eệ]nh nh[aâ]n\b"
     r"|\bd[iị]ch b[eệ]nh\b|\bti[eê]m ch[uủ]ng\b|\bhuy[eế]t [aá]p\b|\bph[oò]ng kh[aá]m\b"
     r"|\bxe c[aấ]p c[uứ]u\b|\bdinh dư[oỡ]ng\b|\bđi[eề]u dư[oỡ]ng\b|\bnh[aà] thu[oố]c\b",
     ("health",)),
    (r"\bth[eể] thao\b|\bb[oó]ng đ[aá]\b|\bc[aầ]u th[uủ]\b|\bthi đ[aấ]u\b|\bgi[aả]i đ[aấ]u\b"
     r"|\bv[aậ]n đ[oộ]ng vi[eê]n\b|\bhuy chương\b|\bs[aâ]n v[aậ]n đ[oộ]ng\b|\bv[oõ] s[iĩ]\b"
     r"|\btr[oọ]ng t[aà]i\b|\bmarathon\b|\bv[eề] đ[ií]ch\b|\bđua xe\b|\btay đua\b"
     r"|\bđ[aạ]p xe\b|\bth[uủ] m[oô]n\b|\bb[uụ]c nh[aậ]n\b",
     ("sports",)),
    (r"\bh[oọ]c sinh\b|\bgi[aá]o vi[eê]n\b|\bl[oớ]p h[oọ]c\b|\bsinh vi[eê]n\b|\bnh[aà] trư[oờ]ng\b"
     r"|\btuy[eể]n sinh\b|\bđ[aà]o t[aạ]o\b|\bk[yỳ] thi\b|\bkhai gi[aả]ng\b|\bb[aả]ng đen\b"
     r"|\bth[aầ]y gi[aá]o\b|\bc[oô] gi[aá]o\b|\bs[aâ]n trư[oờ]ng\b|\bch[aà]o c[oờ]\b",
     ("education",)),
    (r"\bdu l[iị]ch\b|\bdi t[ií]ch\b|\bch[uù]a\b|\bđ[eề]n th[oờ]\b|\bmi[eế]u\b"
     r"|\bl[eễ] h[oộ]i\b|\bv[aă]n h[oó]a\b|\bkh[aá]ch du l[iị]ch\b|\btruy[eề]n th[oố]ng\b"
     r"|\b[aá]o d[aà]i\b|\bph[oố] c[oổ]\b|\bth[aắ]ng c[aả]nh\b|\bm[uú]a l[aâ]n\b"
     r"|\br[uướ]c ki[eệ]u\b|\bkh[aă]n x[eế]p\b|\bđ[eè]n l[oồ]ng\b|\bth[aá]p c[oổ]\b"
     r"|\bcon l[aâ]n\b|\bnh[aả]y m[uú]a\b|\bth[aắ]p hương\b|\b[aá]o the\b",
     ("culture_travel_heritage",)),
    (r"\bgiao th[oô]ng\b|\btai n[aạ]n\b|\b[uù]n t[aắ]c\b|\bcao t[oố]c\b|\bxe bu[yý]t\b"
     r"|\bđ[oô] th[iị]\b|\bb[eế]n xe\b|\bxe m[aá]y\b|\bbi[eể]n s[oố]\b|\bc[aầ]u vư[oợ]t\b"
     r"|\bxe t[aả]i\b|\bm[uũ] b[aả]o hi[eể]m\b|\bđ[eè]n giao th[oô]ng\b|\bgi[oờ] cao đi[eể]m\b"
     r"|\bngư[oờ]i đi b[oộ]\b|\bv[aạ]ch k[eẻ] đư[oờ]ng\b|\bsang đư[oờ]ng\b",
     ("transport_urban",)),
    (r"\bn[oô]ng nghi[eệ]p\b|\bm[uù]a v[uụ]\b|\bcanh t[aá]c\b|\bthu ho[aạ]ch\b|\bru[oộ]ng\b"
     r"|\bn[oô]ng d[aâ]n\b|\bch[aă]n nu[oô]i\b|\bthu[yỷ] s[aả]n\b|\bph[aâ]n b[oó]n\b"
     r"|\bm[aá]y c[aà]y\b|\bg[aặ]t\b|\bl[uú]a\b|\bc[aâ]y tr[oồ]ng\b|\bv[uườ]n c[aâ]y\b"
     r"|\bđ[aà]n b[oò]\b|\bph[aâ]n lo[aạ]i t[oô]m\b|\bnu[oô]i t[oô]m\b|\bthu[oố]c tr[uừ] s[aâ]u\b",
     ("agriculture",)),
    (r"\bng[aâ]n h[aà]ng\b|\bl[aã]i su[aấ]t\b|\bdoanh nghi[eệ]p\b|\bxu[aấ]t kh[aẩ]u\b"
     r"|\bch[uứ]ng kho[aá]n\b|\bb[aấ]t đ[oộ]ng s[aả]n\b|\bgi[aá] c[aả]\b|\bđ[aầ]u tư\b"
     r"|\bc[oổ] phi[eế]u\b|\bgiao d[iị]ch\b|\bti[eề]n m[aặ]t\b|\bthu[eế]\b|\bch[oợ] ch[uứ]ng\b",
     ("economy_finance",)),
    (r"\bt[oộ]i ph[aạ]m\b|\bx[uử] ph[aạ]t\b|\bc[oô]ng an\b|\bb[aắ]t gi[uữ]\b|\bđi[eề]u tra\b"
     r"|\bma t[uú]y\b|\bma tu[yý]\b|\bbu[oô]n l[aậ]u\b|\ban ninh\b|\bc[aả]nh s[aá]t\b"
     r"|\bx[eé]t x[uử]\b|\btang v[aậ]t\b|\bd[aẫ]n gi[aả]i\b"
     r"|\bv[aà]nh m[oó]ng ng[uự]a\b|\bc[oò]ng tay\b|\bt[oò]a [aá]n\b|\bnh[aà] t[uù]\b",
     ("law_security",)),
    (r"\bm[oô]i trư[oờ]ng\b|\bthi[eê]n nhi[eê]n\b|\br[uừ]ng\b|\bb[aã]o\b|\bl[uũ] l[uụ]t\b"
     r"|\b[oô] nhi[eễ]m\b|\bkh[ií] h[aậ]u\b|\bđ[oộ]ng v[aậ]t hoang d[aã]\b|\bs[aạ]t l[oở]\b"
     r"|\bng[aậ]p l[uụ]t\b|\bth[aá]c nư[oớ]c\b|\br[aá]c th[aả]i\b|\bkh[oô] h[aạ]n\b"
     r"|\bch[aá]y r[uừ]ng\b|\bv[aá]ch đ[aá]\b",
     ("environment_nature",)),
    (r"\bkhoa h[oọ]c\b|\bc[oô]ng ngh[eệ]\b|\brobot\b|\bchuy[eể]n đ[oổ]i s[oố]\b"
     r"|\bv[eệ] tinh\b|\bnghi[eê]n c[uứ]u\b|\btr[ií] tu[eệ] nh[aâ]n t[aạ]o\b"
     r"|\bb[aả]ng m[aạ]ch\b|\bph[oò]ng th[ií] nghi[eệ]m\b|\bth[uự]c t[eế] [aả]o\b"
     r"|\bk[ií]nh hi[eể]n vi\b|\bpin m[aặ]t tr[oờ]i\b",
     ("science_technology",)),
    (r"\b[aẩ]m th[uự]c\b|\bn[aấ]u [aă]n\b|\bnh[aà] h[aà]ng\b|\bth[oờ]i trang\b"
     r"|\bl[aà]m đ[eẹ]p\b|\bph[oở]\b|\bb[aá]nh m[iì]\b|\bh[aà]ng rong\b|\bqu[aá]n [aă]n\b"
     r"|\bch[oợ] (quê|đ[eê]m|c[oó]c)\b|\bm[aâ]m c[oơ]m\b|\bđ[aầ]u b[eế]p\b",
     ("food_lifestyle",)),
    (r"\bch[ií]nh tr[iị]\b|\bch[ií]nh s[aá]ch\b|\bh[oộ]i ngh[iị]\b|\bl[aã]nh đ[aạ]o\b"
     r"|\bqu[oố]c h[oộ]i\b|\bngo[aạ]i giao\b|\blao đ[oộ]ng\b|\ban sinh\b|\b[uủ]y ban\b"
     r"|\bh[oọ]p b[aá]o\b|\bk[yý] k[eế]t\b|\bl[aà]m gi[aấ]y t[oờ]\b|\bc[aá]n b[oộ]\b"
     r"|\bđ[aạ]i bi[eể]u\b|\bbi[eể]u quy[eế]t\b|\bh[oộ]i trư[oờ]ng\b|\bph[uư][oờ]ng\b",
     ("politics_society",)),
)

_EN_COMPILED = tuple((re.compile(p, re.IGNORECASE), doms) for p, doms in _EN_PATTERNS)
_VI_COMPILED = tuple((re.compile(p), doms) for p, doms in _VI_PATTERNS)


def _match(compiled, text: str) -> set[str]:
    if not text:
        return set()
    normalized = _norm(text)
    candidates = [normalized]
    stripped = _strip_diacritics(normalized)
    if stripped == normalized:
        # Chỉ thử bản bỏ dấu khi câu VỐN đã không dấu. Bỏ dấu một câu CÓ dấu là tự tạo
        # false positive: "đèn đỏ" thành "đền" -> nhận sai là di tích.
        candidates.append(stripped)
    found: set[str] = set()
    for pattern, domains in compiled:
        if any(pattern.search(c) for c in candidates):
            found.update(domains)
    return found


def domains_from_text(vi_text: str, en_text: str = "") -> set[str]:
    """Domain slug guard nhận ra. Hợp của hai nguồn, không bao giờ trả general_news.

    `en_text` là bản enriched tiếng Anh (rỗng nếu chưa có). Nó là nguồn CHÍNH vì từ vựng
    tiếng Anh ít nhập nhằng hơn; `vi_text` giữ vai trò dự phòng cho lúc enrich lỗi hoặc
    viết lại thiếu.

    Có chủ ý là guard đọc cả bản LLM viết lại, dù guard tồn tại để kiểm LLM: hai tín
    hiệu KHÁC nhau. Viết lại/dịch là việc LLM làm rất chắc (đo 24/24 câu đúng), còn
    PHÂN LOẠI lĩnh vực mới là chỗ nó lung lay. Suy tag một cách tất định từ bản dịch
    đáng tin là tín hiệu độc lập với phán đoán lĩnh vực của chính nó.
    """
    return _match(_VI_COMPILED, vi_text) | _match(_EN_COMPILED, en_text)


def topics_listing(domain_slugs: list[str]) -> str:
    """Dòng "<slug>: topic1, topic2, ..." cho prompt, chỉ những domain có trong vocab."""
    lines = []
    for slug in domain_slugs:
        topics = TOPICS_BY_DOMAIN.get(slug)
        if topics:
            lines.append(f"  {slug}: {', '.join(topics)}")
    return "\n".join(lines)
