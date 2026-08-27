import { useEffect, useRef } from "react";

type VideoPlayerProps = {
  src: string;
  seekSeconds: number;
  title: string;
  onClose: () => void;
};

export function VideoPlayer({ src, seekSeconds, title, onClose }: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  const seekToMatchedFrame = () => {
    const video = videoRef.current;
    if (!video) return;
    const upperBound = Number.isFinite(video.duration)
      ? Math.max(0, video.duration - 0.01)
      : seekSeconds;
    video.currentTime = Math.min(Math.max(0, seekSeconds), upperBound);
    // Dừng đúng frame khớp. Autoplay sẽ lập tức chạy khỏi frame trước khi user kịp
    // kiểm chứng; sau khi mở, user chủ động bấm Play bằng controls.
    video.pause();
  };

  return (
    <div
      className="position-fixed top-0 start-0 w-100 h-100 d-flex align-items-center justify-content-center"
      style={{ zIndex: 3000, background: "rgba(0, 0, 0, 0.78)" }}
      role="dialog"
      aria-modal="true"
      aria-label={title}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="bg-dark text-white rounded shadow-lg p-3" style={{ width: "min(960px, 94vw)" }}>
        <div className="d-flex justify-content-between align-items-center mb-2">
          <div>
            <div className="fw-semibold">{title}</div>
            <small className="text-white-50">
              Scene clip · seek {seekSeconds.toFixed(3)}s tính từ đầu clip
            </small>
          </div>
          <button className="btn btn-sm btn-outline-light" onClick={onClose} aria-label="Đóng">
            ×
          </button>
        </div>
        <video
          ref={videoRef}
          src={src}
          className="w-100 rounded bg-black"
          style={{ maxHeight: "75vh" }}
          controls
          playsInline
          onLoadedMetadata={seekToMatchedFrame}
        />
        <div className="small text-white-50 mt-2">
          Bucket hiện chưa có full video; player dùng scene chứa frame khớp để giữ mapping chính xác.
        </div>
      </div>
    </div>
  );
}
