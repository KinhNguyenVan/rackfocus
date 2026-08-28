import { useState } from "react";
import { prepareTemporal } from "../api/client";
import type { TemporalPrepareResponse } from "../api/types";

// Thứ tự bấm = thứ tự sự kiện. Quy tắc viết đủ để không phải đoán:
//   - chưa chọn gì, bấm A          -> A thành 1
//   - đã có 1, bấm B               -> B thành 2
//   - bấm lại đoạn đang chọn       -> bỏ chọn nó; bỏ 1 thì 2 TỤT LÊN thành 1
//   - đã đủ 2 mà bấm đoạn thứ ba   -> BỎ QUA, không thay thế ngầm
// Không thay thế ngầm vì làm thế là vứt mất lựa chọn user vừa cân nhắc; muốn đổi thì bỏ
// chọn tường minh trước. Tách thành hàm thuần để đọc/kiểm mà không phải dựng React.
export function toggleSelection(selected: number[], order: number): number[] {
	const at = selected.indexOf(order);
	if (at !== -1) return selected.filter((o) => o !== order);
	if (selected.length >= 2) return selected;
	return [...selected, order];
}

type Props = {
	onSearch: (event1: string, event2: string, tags: number[]) => void;
	onRunAsKis: (text: string) => void;
};

// Bước 1 của temporal khi BẬT LLM: gõ một câu tiếng Việt -> LLM tách thành N câu tiếng
// Anh cho CLIP -> user sửa, chọn 2 theo thứ tự, tick tag -> mới search.
// Tắt LLM thì App dùng TemporalQueryBuilder (2 ô nhập tay) thay cho component này.
export function TemporalPrepare({ onSearch, onRunAsKis }: Props) {
	const [query, setQuery] = useState("");
	const [prep, setPrep] = useState<TemporalPrepareResponse | null>(null);
	// Chữ đang hiển thị, theo order. Tách khỏi `prep` vì user sửa được: `prep` giữ bản
	// LLM trả về, `texts` giữ bản thực sự đem đi search.
	const [texts, setTexts] = useState<Record<number, string>>({});
	const [selected, setSelected] = useState<number[]>([]);
	const [tags, setTags] = useState<number[]>([]);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);

	const analyze = () => {
		const q = query.trim();
		if (!q) return;
		setLoading(true);
		setError(null);
		prepareTemporal(q)
			.then((data) => {
				setPrep(data);
				setTexts(Object.fromEntries(data.segments.map((s) => [s.order, s.english_clip_query])));
				setSelected([]);
				setTags(data.tags);
			})
			.catch((cause: unknown) => {
				setPrep(null);
				setError(cause instanceof Error ? cause.message : "Prepare failed");
			})
			.finally(() => setLoading(false));
	};

	const segments = prep?.segments ?? [];
	const ready = selected.length === 2;

	return (
		<div className="mb-3">
			<label className="form-label small fw-semibold">Câu truy vấn (tiếng Việt)</label>
			<div className="input-group mb-2">
				<textarea
					className="form-control"
					rows={2}
					placeholder="VD: Cá được đặt lên cân, sau đó một người cầm đuôi con cá khác."
					value={query}
					onChange={(e) => setQuery(e.target.value)}
				/>
				<button
					className="btn btn-outline-primary"
					type="button"
					disabled={loading || !query.trim()}
					onClick={analyze}
				>
					{loading ? "Đang phân tích..." : "Phân tích"}
				</button>
			</div>

			{error && <div className="alert alert-danger p-2 small">{error}</div>}

			{prep && prep.warnings.includes("llm_failed_segment") && (
				<div className="alert alert-warning p-2 small">
					LLM tách đoạn lỗi — đang hiện nguyên câu gốc.
				</div>
			)}

			{segments.length === 1 && (
				<div className="alert alert-info p-2 small">
					Chỉ tìm thấy 1 sự kiện — không tạo được chuỗi thời gian.
					<button
						className="btn btn-sm btn-outline-primary ms-2"
						type="button"
						onClick={() => onRunAsKis(texts[segments[0].order] ?? "")}
					>
						Chạy như KIS thường
					</button>
				</div>
			)}

			{segments.length > 1 && (
				<>
					<label className="form-label small fw-semibold">
						Câu tiếng Anh cho CLIP — bấm số để chọn 2, theo thứ tự:
					</label>
					{segments.map((s) => {
						const at = selected.indexOf(s.order);
						return (
							<div className="d-flex align-items-start gap-2 mb-2" key={s.order}>
								<button
									type="button"
									className={`btn btn-sm ${at === -1 ? "btn-outline-secondary" : "btn-primary"}`}
									style={{ width: "2.5rem", flexShrink: 0 }}
									onClick={() => setSelected((cur) => toggleSelection(cur, s.order))}
									title={at === -1 ? "Chọn làm sự kiện" : "Bỏ chọn"}
								>
									{at === -1 ? "·" : at + 1}
								</button>
								{/* `order` = thứ tự thời gian LLM suy ra, chỉ để đọc. Badge bên trái mới
								    là thứ tự sự kiện user chọn — hai con số khác nhau, đừng gộp hiển thị. */}
								<span className="text-muted small align-self-center" style={{ minWidth: "1.5rem" }}>
									{s.order}.
								</span>
								<textarea
									className="form-control form-control-sm"
									rows={2}
									value={texts[s.order] ?? ""}
									onChange={(e) =>
										setTexts((cur) => ({ ...cur, [s.order]: e.target.value }))
									}
								/>
							</div>
						);
					})}

					<label className="form-label small fw-semibold mt-2">
						Lọc theo lĩnh vực (LLM tự tin {prep!.confidence.toFixed(2)} · nguồn: {prep!.tag_source || "?"}):
					</label>
					<div className="d-flex flex-wrap gap-2 mb-2">
						{Object.entries(prep!.tag_names).map(([id, name]) => {
							const tid = Number(id);
							return (
								<div className="form-check" key={id}>
									<input
										className="form-check-input"
										type="checkbox"
										checked={tags.includes(tid)}
										onChange={() =>
											setTags((cur) =>
												cur.includes(tid) ? cur.filter((t) => t !== tid) : [...cur, tid],
											)
										}
									/>
									<label className="form-check-label small">
										{id} {name}
									</label>
								</div>
							);
						})}
					</div>
					<div className="text-muted small mb-2">
						Bỏ tick hết = search toàn kho (chậm hơn nhưng không bỏ sót).
					</div>

					<button
						className="btn btn-primary"
						type="button"
						disabled={!ready}
						onClick={() => onSearch(texts[selected[0]], texts[selected[1]], tags)}
					>
						Tìm chuỗi {ready ? "" : `(còn thiếu ${2 - selected.length})`}
					</button>
				</>
			)}
		</div>
	);
}
