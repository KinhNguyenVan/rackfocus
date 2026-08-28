import type { TemporalChain } from "../api/types";

type Props = {
	chain: TemporalChain;
	onUseChain: (chain: TemporalChain) => void;
	onViewFrames: (chain: TemporalChain) => void;
};

// Một chain: video, điểm, khoảng cách thời gian, 2 thumbnail theo đúng thứ tự.
// "Dùng chuỗi này" đẩy cả 2 hit vào sidebar Selected có sẵn -- exportResult() nhánh
// "trake" đã kỳ vọng frame cùng 1 video trong `selected`, nên không cần đổi gì ở đó.
export function TemporalChainCard({ chain, onUseChain, onViewFrames }: Props) {
	const [hit1, hit2] = chain.hits;
	return (
		<div className="card mb-2">
			<div className="card-body p-2">
				<div className="d-flex justify-content-between align-items-center mb-1">
					<small className="fw-semibold">{chain.video_name}</small>
					<small className="text-muted">
						score={chain.score.toFixed(3)} · span={chain.span_sec.toFixed(1)}s
					</small>
				</div>
				<div className="d-flex gap-2">
					{[hit1, hit2].map((hit, i) => (
						<div key={i} className="text-center" style={{ flex: 1 }}>
							<img
								src={hit.keyframe_url || "https://placehold.co/200x120"}
								className="img-fluid rounded"
								alt={`event ${i + 1}`}
							/>
							<small className="d-block text-muted">
								#{i + 1} · {hit.keyframe_time.toFixed(1)}s
							</small>
						</div>
					))}
				</div>
				<div className="d-flex gap-2 mt-2">
					<button
						className="btn btn-sm btn-outline-primary flex-fill"
						onClick={() => onUseChain(chain)}
					>
						Dùng chuỗi này
					</button>
					<button
						className="btn btn-sm btn-outline-secondary flex-fill"
						onClick={() => onViewFrames(chain)}
					>
						Xem frames giữa
					</button>
				</div>
			</div>
		</div>
	);
}
