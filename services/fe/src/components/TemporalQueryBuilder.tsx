type Props = {
	event1: string;
	event2: string;
	onEvent1Change: (value: string) => void;
	onEvent2Change: (value: string) => void;
};

// 2 ô nhập sự kiện có thứ tự cho TRAKE — thay ô query đơn của KIS khi bật temporal mode.
// Gap tối thiểu/tối đa giữa 2 sự kiện là cấu hình phía core (TRAKE_MIN_GAP_SEC/
// TRAKE_MAX_GAP_SEC), không phơi ra UI.
export function TemporalQueryBuilder({ event1, event2, onEvent1Change, onEvent2Change }: Props) {
	return (
		<div className="mb-3">
			<div className="mb-2">
				<label className="form-label small fw-semibold">Sự kiện 1 (xảy ra trước)</label>
				<input
					type="text"
					className="form-control"
					placeholder="VD: người đàn ông cầm micro phát biểu"
					value={event1}
					onChange={(e) => onEvent1Change(e.target.value)}
				/>
			</div>
			<div>
				<label className="form-label small fw-semibold">Sự kiện 2 (xảy ra sau)</label>
				<input
					type="text"
					className="form-control"
					placeholder="VD: khán giả đứng dậy vỗ tay"
					value={event2}
					onChange={(e) => onEvent2Change(e.target.value)}
				/>
			</div>
		</div>
	);
}
