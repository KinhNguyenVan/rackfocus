import type { TranscriptSuggestItem } from "../api/types";

type Props = {
  items: TranscriptSuggestItem[];
  loading: boolean;
  error: string | null;
  onPick: (item: TranscriptSuggestItem) => void;
};

// snippet BE trả có keyword bọc trong [ ] (FTS5 snippet()). Tách thành đoạn thường + <mark>
// để highlight, không dùng dangerouslySetInnerHTML (tránh injection từ transcript).
function renderSnippet(snippet: string) {
  const parts = snippet.split(/\[([^\]]*)\]/g);
  return parts.map((part, i) =>
    i % 2 === 1 ? (
      <mark key={i} className="px-0">
        {part}
      </mark>
    ) : (
      <span key={i}>{part}</span>
    ),
  );
}

function fmtTime(sec: number): string {
  const s = Math.max(0, Math.floor(sec));
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, "0")}`;
}

export function TranscriptSuggest({ items, loading, error, onPick }: Props) {
  return (
    <div
      className="position-absolute start-0 end-0 bg-white border rounded shadow-lg mt-1"
      style={{ zIndex: 2000, maxHeight: "60vh", overflowY: "auto" }}
      role="listbox"
    >
      {error && <div className="p-2 small text-danger">{error}</div>}
      {!error && loading && items.length === 0 && (
        <div className="p-2 small text-muted">Đang tìm…</div>
      )}
      {!error && !loading && items.length === 0 && (
        <div className="p-2 small text-muted">Không có transcript khớp</div>
      )}
      {items.map((item) => (
        <button
          key={`${item.video_name}#${item.scene_idx}`}
          type="button"
          className="d-flex align-items-start gap-2 w-100 text-start border-0 bg-transparent p-2 suggest-row"
          role="option"
          onClick={() => onPick(item)}
        >
          {item.keyframe_url ? (
            <img
              src={item.keyframe_url}
              alt=""
              width={72}
              height={40}
              className="rounded flex-shrink-0"
              style={{ objectFit: "cover" }}
            />
          ) : (
            <div
              className="rounded flex-shrink-0 bg-light"
              style={{ width: 72, height: 40 }}
            />
          )}
          <div className="flex-grow-1 min-w-0">
            <div className="small text-truncate">{renderSnippet(item.snippet)}</div>
            <div className="text-muted" style={{ fontSize: "0.72rem" }}>
              {item.video_name} · scene {item.scene_idx} · {fmtTime(item.start_sec)}
            </div>
          </div>
        </button>
      ))}
    </div>
  );
}
