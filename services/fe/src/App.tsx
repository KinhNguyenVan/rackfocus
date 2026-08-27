import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { FormEvent, useEffect, useMemo, useState } from "react";
import { neighbors as fetchNeighbors } from "./api/client";
import type { NeighborFrame, SearchHit, TemporalChain } from "./api/types";
import { VideoPlayer } from "./components/VideoPlayer";
import { useSearch } from "./hooks/useSearch";
import { useTemporalSearch } from "./hooks/useTemporalSearch";
import { TemporalQueryBuilder } from "./components/TemporalQueryBuilder";
import { TemporalChainCard } from "./components/TemporalChainCard";
import "./styles/global.css";

type Task = "kis" | "qa" | "trake";
// SearchHit đã có video_name/frame/keyframe_url thật (services/be/src/app/api/search.py)
// -- url/video là alias tiện dùng trong UI, point_id là khoá duy nhất thật (không có
// scene_id toàn cục nào cả, scene_idx chỉ là index trong 1 video).
type Result = SearchHit & {
  url: string;
  video: string;
};

// Hash chuỗi ổn định trong phiên (djb2) -- dùng làm point_id giả cho frame lân cận
// (không có point_id thật vì không đi qua snapshot), chỉ cần duy nhất/lặp lại được,
// không cần khớp cách core băm point_id thật (blake2b).
function hashString(s: string): number {
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = (h * 33) ^ s.charCodeAt(i);
  return h >>> 0;
}
const topics = [
  "tin tức",
  "Tin tức+múa lân",
  "Tin tức+xe đạp",
  "Tin tức+dạy học online",
  "Tin tức+chương trình nấu ăn",
];

export default function App() {
  const [route, setRoute] = useState(window.location.pathname);

  // Tự động inject CDNs để đảm bảo cả Bootstrap và Tailwind đều hoạt động
  useEffect(() => {
    if (!document.getElementById("bs-css")) {
      const bsLink = document.createElement("link");
      bsLink.id = "bs-css";
      bsLink.rel = "stylesheet";
      bsLink.href =
        "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css";
      document.head.appendChild(bsLink);
    }
    if (!document.getElementById("bs-icons")) {
      const iconsLink = document.createElement("link");
      iconsLink.id = "bs-icons";
      iconsLink.rel = "stylesheet";
      iconsLink.href =
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.2/font/bootstrap-icons.min.css";
      document.head.appendChild(iconsLink);
    }
    if (!document.getElementById("tw-script")) {
      const twScript = document.createElement("script");
      twScript.id = "tw-script";
      twScript.src = "https://cdn.tailwindcss.com";
      document.head.appendChild(twScript);
    }
  }, []);

  useEffect(() => {
    const onPop = () => setRoute(window.location.pathname);
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const go = (path: string) => {
    window.history.pushState({}, "", path);
    setRoute(path);
  };

  return route === "/submission" ? (
    <SubmissionPage goSearch={() => go("/")} />
  ) : (
    <SearchPage goSubmission={() => go("/submission")} />
  );
}

// -------------------------------------------------------------
// SEARCH PAGE (Sử dụng cấu trúc DOM Bootstrap chuẩn từ index.html)
// -------------------------------------------------------------
function SearchPage({ goSubmission }: { goSubmission: () => void }) {
  const [query, setQuery] = useState("");
  const [submitted, setSubmitted] = useState("");
  const [flag, setFlag] = useState("");
  const [service, setService] = useState("image");
  const [chosenTopics, setChosenTopics] = useState<string[]>([]);
  // Tắt = bỏ qua bước LLM chọn tag ở BE (services/be/src/app/api/search.py::_enrich
  // đã có sẵn use_llm), core search toàn bộ corpus không lọc theo tag.
  const [useLlm, setUseLlm] = useState(true);
  // "rerank" (mặc định) = core tự chọn: HNSW+SQ8 coarse -> rerank exact trên top
  // rerank_candidates (hoặc EXACT_SUBSET nếu tag đủ hẹp). "exact" = ép brute-force
  // toàn candidate/corpus, bỏ qua HNSW hoàn toàn -- chậm hơn, không xấp xỉ.
  const [exactMode, setExactMode] = useState(false);
  const [searchMode, setSearchMode] = useState<"kis" | "temporal">("kis");
  const [event1, setEvent1] = useState("");
  const [event2, setEvent2] = useState("");
  const [submittedEvent1, setSubmittedEvent1] = useState("");
  const [submittedEvent2, setSubmittedEvent2] = useState("");
  const [selected, setSelected] = useState<Result[]>([]);
  const [task, setTask] = useState<Task>("kis");
  const [qaAnswer, setQaAnswer] = useState("");
  const [output, setOutput] = useState("");
  const [framesOpen, setFramesOpen] = useState(false);
  const [neighborFrames, setNeighborFrames] = useState<NeighborFrame[]>([]);
  const [neighborVideo, setNeighborVideo] = useState("");
  const [framesLabel, setFramesLabel] = useState("");
  const [neighborsLoading, setNeighborsLoading] = useState(false);
  const [neighborsError, setNeighborsError] = useState<string | null>(null);
  const [playback, setPlayback] = useState<Result | null>(null);
  const [contextMenu, setContextMenu] = useState<{
    visible: boolean;
    x: number;
    y: number;
    target: Result | null;
  }>({
    visible: false,
    x: 0,
    y: 0,
    target: null,
  });

  const loadFrames = (
    video: string,
    fromUrl: string,
    before: number,
    after: number,
    toUrl?: string,
  ) => {
    setFramesOpen(true);
    setNeighborVideo(video);
    setFramesLabel(toUrl ? `${before} trước · giữa 2 sự kiện · ${after} sau` : `${before} trước/sau`);
    setNeighborsLoading(true);
    setNeighborsError(null);
    fetchNeighbors(fromUrl, before, after, toUrl)
      .then((data) => {
        // BE suy video_name từ S3 key nên chính xác hơn tên đoán từ hit.
        setNeighborVideo(data.video_name);
        setNeighborFrames(data.frames ?? []);
      })
      .catch((cause: unknown) => {
        setNeighborFrames([]);
        setNeighborsError(cause instanceof Error ? cause.message : "Load frames failed");
      })
      .finally(() => setNeighborsLoading(false));
  };

  // Nhận Result thay vì (url, video): context menu của main giữ cả object trong
  // contextMenu.target, và loadFrames cần đúng hai field này.
  const loadNeighbors = (result: Result) =>
    loadFrames(result.video, result.keyframe_url, 25, 25);

  // "Xem frames giữa" trên TemporalChainCard: 5 khung trước sự kiện 1, tất cả khung
  // giữa 2 sự kiện, 5 khung sau sự kiện 2 -- cả 2 hit luôn cùng video (TemporalChain).
  const loadChainFrames = (chain: TemporalChain) => {
    const [hit1, hit2] = chain.hits;
    loadFrames(chain.video_name, hit1.keyframe_url, 5, 5, hit2.keyframe_url);
  };

  // Frame lân cận không đi qua vector index nên không có point_id thật. Timeline và
  // scene metadata vẫn là dữ liệu thật, được BE nối từ scenes.json trên S3.
  const neighborToResult = (video: string, nf: NeighborFrame): Result => ({
    point_id: hashString(`${video}:${nf.frame}`),
    row: -1,
    score: 0,
    rank: -1,
    video_name: video,
    frame: nf.frame,
    keyframe_time: nf.keyframe_time,
    start_sec: nf.start_sec,
    end_sec: nf.end_sec,
    keyframe_url: nf.url,
    clip_url: nf.clip_url,
    scene_idx: nf.scene_idx,
    has_speech: false,
    url: nf.url,
    video,
  });

  // top_k=300: khớp thiết kế "rerank lấy top 1k rồi rerank exact top 300 / exact brute-
  // force thẳng top 300" -- BE max_top_k cũng phải nới lên (config.py) không thì bị cắt.
  const { hits, totalMs, strategy, loading, error } = useSearch(
    submitted,
    300,
    useLlm,
    exactMode,
  );

  const {
    chains,
    totalMs: temporalTotalMs,
    warnings: temporalWarnings,
    loading: temporalLoading,
    error: temporalError,
  } = useTemporalSearch(submittedEvent1, submittedEvent2, useLlm, exactMode);

  const results = useMemo(
    () =>
      hits.map((hit) => ({
        ...hit,
        video: hit.video_name,
        url:
          hit.keyframe_url ||
          `https://placehold.co/640x360/e9eef0/354d58?text=${hit.video_name}%23${hit.frame}`,
      })),
    [hits],
  );

  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem("searchState") ?? "{}");
      setQuery(saved.query ?? "");
      setFlag(saved.flagValue ?? "");
      setService(saved.serviceValue ?? "image");
      setChosenTopics(saved.selectedTopics ?? []);
      setUseLlm(saved.useLlm ?? true);
      setExactMode(saved.exactMode ?? false);
      setSearchMode(saved.searchMode ?? "kis");
    } catch {}
  }, []);

  useEffect(() => {
    localStorage.setItem(
      "searchState",
      JSON.stringify({
        query,
        flagValue: flag,
        serviceValue: service,
        selectedTopics: chosenTopics,
        useLlm,
        exactMode,
        searchMode,
      }),
    );
  }, [query, flag, service, chosenTopics, useLlm, exactMode, searchMode]);

  useEffect(() => {
    const handleClick = () =>
      setContextMenu((current) => ({ ...current, visible: false }));
    document.addEventListener("click", handleClick);
    return () => document.removeEventListener("click", handleClick);
  }, []);

  const toggle = (result: Result) => {
    setSelected((items) =>
      items.some((item) => item.point_id === result.point_id)
        ? items.filter((item) => item.point_id !== result.point_id)
        : [...items, result],
    );
  };

  const useChain = (chain: TemporalChain) => {
    const results: Result[] = chain.hits.map((hit) => ({
      ...hit,
      video: hit.video_name,
      url:
        hit.keyframe_url ||
        `https://placehold.co/640x360/e9eef0/354d58?text=${hit.video_name}%23${hit.frame}`,
    }));
    setSelected(results);
  };

  const clear = () => {
    if (
      !window.confirm(
        "Clear saved search state? This will remove your saved query and selections.",
      )
    )
      return;
    localStorage.removeItem("searchState");
    setQuery("");
    setFlag("");
    setService("image");
    setChosenTopics([]);
    setUseLlm(true);
    setExactMode(false);
    setSubmitted("");
    setSelected([]);
    setOutput("");
  };

  function search(event: FormEvent) {
    event.preventDefault();
    if (searchMode === "temporal") {
      if (event1.trim() && event2.trim()) {
        setSelected([]);
        setSubmittedEvent1(event1.trim());
        setSubmittedEvent2(event2.trim());
      }
      return;
    }
    if (query.trim()) {
      setSelected([]);
      setSubmitted(query.trim());
    }
  }

  // ms thật từ keyframe_time (giây, BE tính từ payload) -- chính xác hơn suy ra từ
  // frame*33 vì đó giả định cứng 30fps, không đúng với mọi video.
  const toMs = (item: Result) => Math.round(item.keyframe_time * 1000);

  function exportResult() {
    if (!selected.length)
      return alert(`${task.toUpperCase()}: Cần chọn ít nhất 1 frame.`);
    const first = selected[0];
    if (task === "kis") {
      const last = selected[1] ?? first;
      if (first.video !== last.video)
        return alert("KIS: 2 frame phải thuộc cùng 1 video.");
      setOutput(
        JSON.stringify(
          {
            answerSets: [
              {
                answers: [
                  {
                    mediaItemName: first.video,
                    start: toMs(first),
                    end: toMs(last),
                  },
                ],
              },
            ],
          },
          null,
          2,
        ),
      );
    }
    if (task === "qa") {
      if (selected.length !== 1 || !qaAnswer)
        return alert("QA: Cần chọn đúng 1 frame và nhập answer.");
      setOutput(
        JSON.stringify(
          {
            answerSets: [
              {
                answers: [
                  { text: `QA-${qaAnswer}-${first.video}-${toMs(first)}` },
                ],
              },
            ],
          },
          null,
          2,
        ),
      );
    }
    if (task === "trake") {
      if (!selected.every((item) => item.video === first.video))
        return alert("Trake: Tất cả frame phải cùng 1 video.");
      setOutput(
        JSON.stringify(
          {
            answerSets: [
              {
                answers: [
                  {
                    text: `TR-${first.video}-${selected.map((item) => toMs(item)).join(",")}`,
                  },
                ],
              },
            ],
          },
          null,
          2,
        ),
      );
    }
  }

  const send = () => {
    if (!output)
      return alert("Vui lòng xuất kết quả trước khi gửi sang trang nộp bài.");
    localStorage.setItem("preparedSubmissionBody", output);
    goSubmission();
  };

  const handleContextMenu = (e: React.MouseEvent, result: Result) => {
    e.preventDefault();
    setContextMenu({ visible: true, x: e.pageX, y: e.pageY, target: result });
  };

  return (
    <div id="pageWrapper" className="d-flex">
      {/* Left Sidebar */}
      <div
        id="leftSidebar"
        className="position-fixed top-0 start-0 bg-light shadow h-100 p-2"
        style={{ width: "200px", overflowY: "auto", zIndex: 1040 }}
      >
        <h6>Selected</h6>
        <div id="selectedList" className="row row-cols-1 g-2">
          {selected.map((item, index) => (
            <div className="col" key={item.point_id}>
              <div className="card">
                <img
                  src={item.url}
                  className="card-img-top"
                  style={{ height: "60px", objectFit: "cover" }}
                  alt="selected"
                />
                <div className="card-body p-1 text-center">
                  <input
                    type="checkbox"
                    className="form-check-input selectedSidebarCb me-1"
                    checked
                    readOnly
                    onClick={() => toggle(item)}
                  />
                  <small>
                    {index + 1}. {item.video}#{item.frame}
                  </small>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Context menu của một hit thật; giữ cả Result để mọi action dùng cùng metadata. */}
      {contextMenu.visible && contextMenu.target && (
        <div
          id="contextMenu"
          className="position-absolute bg-white border rounded shadow-sm"
          style={{
            top: contextMenu.y,
            left: contextMenu.x,
            zIndex: 2000,
            display: "block",
          }}
        >
          <button
            className="dropdown-item"
            onClick={() => loadNeighbors(contextMenu.target!)}
          >
            📽️ Xem 25 frames trước/sau
          </button>
          <button
            className="dropdown-item"
            onClick={() => navigator.clipboard.writeText(contextMenu.target!.keyframe_url)}
          >
            🔗 Copy URL
          </button>
          <button
            className="dropdown-item"
            disabled={!contextMenu.target.clip_url}
            onClick={() => setPlayback(contextMenu.target)}
          >
            ▶️ Mở video tại frame khớp
          </button>
        </div>
      )}

      {/* Main Content */}
      <div
        id="mainContent"
        className="flex-grow-1"
        style={{ marginLeft: "200px", zIndex: 1 }}
      >
        <div className="container2">
          <nav className="navbar bg-body-tertiary mb-3">
            <div className="container d-flex justify-content-between">
              <span className="navbar-brand mb-0 h1">
                Web ML content-based image retrieval demo
              </span>
              <div>
                <button
                  className="btn btn-sm btn-outline-secondary me-2"
                  onClick={() => window.scrollTo(0, 0)}
                >
                  Tìm kiếm
                </button>
                <button
                  className="btn btn-sm btn-primary"
                  onClick={goSubmission}
                >
                  Trang nộp bài
                </button>
              </div>
            </div>
          </nav>

          <div className="container">
            <div className="row g-3">
              <div className="col-12 col-md-6">
                <div className="card h-100 shadow-sm">
                  <div className="card-body">
                    <h5 className="card-title">
                      <i className="bi bi-image"></i> Image Search
                    </h5>
                    <div className="btn-group btn-group-sm mb-2" role="group">
                      <input
                        type="radio"
                        className="btn-check"
                        name="searchTopMode"
                        id="modeKis"
                        checked={searchMode === "kis"}
                        onChange={() => setSearchMode("kis")}
                      />
                      <label className="btn btn-outline-secondary" htmlFor="modeKis">
                        KIS (1 câu)
                      </label>
                      <input
                        type="radio"
                        className="btn-check"
                        name="searchTopMode"
                        id="modeTemporal"
                        checked={searchMode === "temporal"}
                        onChange={() => {
                          setSearchMode("temporal");
                          setUseLlm(false);
                        }}
                      />
                      <label className="btn btn-outline-secondary" htmlFor="modeTemporal">
                        Temporal (2 sự kiện)
                      </label>
                    </div>
                    <form onSubmit={search}>
                      {searchMode === "temporal" ? (
                        <TemporalQueryBuilder
                          event1={event1}
                          event2={event2}
                          onEvent1Change={setEvent1}
                          onEvent2Change={setEvent2}
                        />
                      ) : (
                        <div className="input-group mb-3">
                          <input
                            type="text"
                            className="form-control"
                            placeholder="Search by text..."
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                          />
                          <input
                            type="text"
                            className="form-control"
                            placeholder="Enter flag..."
                            value={flag}
                            onChange={(e) => setFlag(e.target.value)}
                          />
                          <select
                            className="form-select"
                            style={{ maxWidth: "120px" }}
                            value={service}
                            onChange={(e) => setService(e.target.value)}
                          >
                            <option value="image">Image</option>
                            <option value="caption">Caption</option>
                            <option value="ocr">OCR</option>
                          </select>
                        </div>
                      )}
                      <button className="btn btn-outline-secondary" type="submit">
                        <i className="bi bi-search"></i>
                      </button>
                      <button
                        className="btn btn-sm btn-outline-danger ms-2"
                        type="button"
                        onClick={clear}
                        title="Clear saved search state"
                      >
                        Clear
                      </button>
                    </form>
                    <div className="form-check form-switch mb-3">
                      <input
                        className="form-check-input"
                        type="checkbox"
                        role="switch"
                        id="useLlmSwitch"
                        checked={useLlm}
                        onChange={(e) => setUseLlm(e.target.checked)}
                      />
                      <label
                        className="form-check-label small"
                        htmlFor="useLlmSwitch"
                      >
                        {useLlm
                          ? "Lọc theo LLM (tag) — bật"
                          : "Search toàn bộ, không lọc — tắt LLM"}
                      </label>
                    </div>
                    <div className="mb-3">
                      <label className="form-label small fw-semibold d-block">
                        Chế độ search:
                      </label>
                      <div className="btn-group btn-group-sm" role="group">
                        <input
                          type="radio"
                          className="btn-check"
                          name="searchMode"
                          id="modeRerank"
                          checked={!exactMode}
                          onChange={() => setExactMode(false)}
                        />
                        <label
                          className="btn btn-outline-secondary"
                          htmlFor="modeRerank"
                          title="HNSW+SQ8 coarse -> rerank exact trên top rerank_candidates"
                        >
                          rerank
                        </label>
                        <input
                          type="radio"
                          className="btn-check"
                          name="searchMode"
                          id="modeExact"
                          checked={exactMode}
                          onChange={() => setExactMode(true)}
                        />
                        <label
                          className="btn btn-outline-secondary"
                          htmlFor="modeExact"
                          title="Brute-force toàn candidate/corpus, bỏ qua HNSW -- chậm hơn, không xấp xỉ"
                        >
                          exact
                        </label>
                      </div>
                      {strategy && (
                        <small className="text-muted ms-2">
                          strategy thật: {strategy}
                        </small>
                      )}
                    </div>
                    <div className="mb-3">
                      <label className="form-label small fw-semibold">
                        Topics:
                      </label>
                      <div className="d-flex flex-wrap gap-2">
                        {topics.map((topic) => (
                          <div className="form-check" key={topic}>
                            <input
                              className="form-check-input topic-checkbox"
                              type="checkbox"
                              checked={chosenTopics.includes(topic)}
                              onChange={() =>
                                setChosenTopics((items) =>
                                  items.includes(topic)
                                    ? items.filter((i) => i !== topic)
                                    : [...items, topic],
                                )
                              }
                            />
                            <label className="form-check-label small">
                              {topic}
                            </label>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="col-12 col-md-6">
                <div className="card h-100 shadow-sm">
                  <div className="card-body">
                    <h5 className="card-title">
                      <i className="bi bi-ui-checks"></i> Xuất thông báo kết quả
                    </h5>
                    <div className="mb-2">
                      <label className="form-label">Kiểu câu hỏi</label>
                      <select
                        className="form-select"
                        value={task}
                        onChange={(e) => setTask(e.target.value as Task)}
                      >
                        <option value="kis">kis</option>
                        <option value="qa">qa</option>
                        <option value="trake">trake</option>
                      </select>
                    </div>
                    {task === "qa" && (
                      <div className="mb-2" id="qaAnswerGroup">
                        <label className="form-label">
                          Answer (chỉ cho QA)
                        </label>
                        <input
                          type="text"
                          className="form-control"
                          placeholder="Nhập answer..."
                          value={qaAnswer}
                          onChange={(e) => setQaAnswer(e.target.value)}
                        />
                      </div>
                    )}
                    <div className="d-grid gap-2 mb-2">
                      <button
                        className="btn btn-primary"
                        onClick={exportResult}
                      >
                        <i className="bi bi-check2-circle"></i> Xuất kết quả
                      </button>
                      <button
                        className="btn btn-outline-primary"
                        onClick={send}
                      >
                        <i className="bi bi-box-arrow-in-right"></i> Gửi sang
                        trang nộp bài
                      </button>
                    </div>
                    <label className="form-label">Kết quả</label>
                    <textarea
                      className="form-control"
                      rows={8}
                      readOnly
                      value={output}
                    ></textarea>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-3 mb-4">
              <div className="card">
                <div className="card-body">
                  <h5 className="card-title">
                    <i className="bi bi-database"></i> Images results
                  </h5>
                  {searchMode === "temporal" ? (
                    <>
                      {temporalLoading && (
                        <div className="text-center my-3">
                          <div className="spinner-border text-primary" role="status">
                            <span className="visually-hidden">Loading...</span>
                          </div>
                        </div>
                      )}
                      {temporalError && <p className="text-danger">{temporalError}</p>}
                      {temporalWarnings.length > 0 && (
                        <p className="text-muted small">{temporalWarnings.join(", ")}</p>
                      )}
                      {chains.map((chain, i) => (
                        <TemporalChainCard
                          key={i}
                          chain={chain}
                          onUseChain={useChain}
                          onViewFrames={loadChainFrames}
                        />
                      ))}
                      {temporalTotalMs !== null && (
                        <small className="text-muted d-block mt-3">
                          {chains.length} chains · {temporalTotalMs.toFixed(2)} ms
                        </small>
                      )}
                    </>
                  ) : (
                    <>
                      {loading && (
                        <div className="text-center my-3">
                          <div
                            className="spinner-border text-primary"
                            role="status"
                          >
                            <span className="visually-hidden">Loading...</span>
                          </div>
                        </div>
                      )}
                      {error && <p className="text-danger">{error}</p>}
                      <div
                        className="row row-cols-2 row-cols-sm-3 row-cols-md-5 g-2"
                        id="imageContainer"
                      >
                        {results.map((result) => {
                          const isSelected = selected.some(
                            (item) => item.point_id === result.point_id,
                          );
                          const selectedIndex =
                            selected.findIndex(
                              (item) => item.point_id === result.point_id,
                            ) + 1;
                          return (
                            <div
                              className="col p-2 text-center"
                              key={result.point_id}
                            >
                              <div className="card position-relative">
                                <img
                                  src={result.url}
                                  className="card-img-top main-img"
                                  alt="scene"
                                  onContextMenu={(e) =>
                                    handleContextMenu(e, result)
                                  }
                                />
                                <div className="card-body p-2">
                                  <input
                                    type="checkbox"
                                    className="form-check-input me-1 selectImg"
                                    checked={isSelected}
                                    onChange={() => toggle(result)}
                                  />
                                  <small>
                                    {result.video}#{result.frame}
                                  </small>
                                </div>
                                {isSelected && (
                                  <span className="badge bg-primary position-absolute top-0 start-0 m-1 order-badge">
                                    {selectedIndex}
                                  </span>
                                )}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                      {totalMs !== null && (
                        <small className="text-muted d-block mt-3">
                          {results.length} results · {totalMs.toFixed(2)} ms
                        </small>
                      )}
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Right Sidebar (Frames) -- flexbox, không dùng float: float trước đây làm
          content tràn khỏi width khai báo, panel che gần hết màn hình. */}
      <div
        id="frameSidebar"
        className="position-fixed top-0 end-0 bg-light shadow h-100 d-flex"
        style={{
          width: framesOpen ? "400px" : "40px",
          maxWidth: "90vw",
          transition: "0.2s",
          zIndex: 1050,
          overflow: "hidden",
        }}
      >
        <div
          className="d-flex justify-content-center align-items-center h-100 border-start"
          style={{ width: "40px", flexShrink: 0, cursor: "pointer" }}
          onClick={() => setFramesOpen(!framesOpen)}
        >
          <button className="btn btn-sm btn-outline-secondary">
            {framesOpen ? "»" : "«"}
          </button>
        </div>
        {framesOpen && (
          <div
            id="sidebarContent"
            style={{
              flex: 1,
              minWidth: 0,
              padding: "10px",
              height: "100%",
              overflowY: "auto",
            }}
          >
            <div className="d-flex justify-content-between align-items-center border-bottom mb-2">
              <h6 className="mb-0">
                Frames {neighborVideo && `— ${neighborVideo}`}
              </h6>
            </div>
            {framesLabel && <span className="badge text-bg-primary mb-2">{framesLabel}</span>}
            {neighborsLoading && (
              <div className="text-center my-3">
                <div className="spinner-border spinner-border-sm text-primary" role="status">
                  <span className="visually-hidden">Loading...</span>
                </div>
              </div>
            )}
            {neighborsError && (
              <div className="alert alert-danger p-2 small">{neighborsError}</div>
            )}
            {!neighborsLoading && !neighborsError && neighborFrames.length === 0 && (
              <div className="text-muted small">
                Bấm chuột phải vào 1 ảnh kết quả rồi chọn "Xem 25 frames trước/sau".
              </div>
            )}
            <div className="row row-cols-2 g-1">
              {neighborFrames.map((nf) => {
                const item = neighborToResult(neighborVideo, nf);
                const isSelected = selected.some(
                  (s) => s.point_id === item.point_id,
                );
                return (
                  <div className="col p-1" key={item.point_id}>
                    <div className="card position-relative">
                      <img
                        src={nf.url}
                        className="img-fluid rounded"
                        alt="neighbor frame"
                        onContextMenu={(event) => handleContextMenu(event, item)}
                      />
                      <div className="card-body p-1 text-center">
                        <input
                          type="checkbox"
                          className="form-check-input me-1"
                          checked={isSelected}
                          onChange={() => toggle(item)}
                        />
                        <small className={nf.is_current ? "fw-bold text-primary" : ""}>
                          #{nf.frame}{nf.is_current ? " · current" : ""}
                        </small>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
      {playback && (
        <VideoPlayer
          src={playback.clip_url}
          seekSeconds={Math.max(0, playback.keyframe_time - playback.start_sec)}
          title={`${playback.video} · frame ${playback.frame}`}
          onClose={() => setPlayback(null)}
        />
      )}
    </div>
  );
}

// -------------------------------------------------------------
// SUBMISSION PAGE (Sử dụng cấu trúc DOM Tailwind chuẩn từ AIC25-Submission.html)
// -------------------------------------------------------------
function SubmissionPage({ goSearch }: { goSearch: () => void }) {
  const [session, setSession] = useState(localStorage.getItem("aic_sessionId"));
  const [evaluation, setEvaluation] = useState(
    localStorage.getItem("aic_evaluationId") ?? "",
  );
  const [prepared, setPrepared] = useState(
    localStorage.getItem("preparedSubmissionBody") ?? "",
  );
  const [usePrepared, setUsePrepared] = useState(false);
  const [task, setTask] = useState<Task>("kis");

  const [video, setVideo] = useState("");
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [answer, setAnswer] = useState("");
  const [frames, setFrames] = useState("");

  const [status, setStatus] = useState({ text: "", type: "" });
  const [response, setResponse] = useState<{
    ok: boolean;
    code: string;
    body: string;
  }>();

  const mockLogin = (event?: FormEvent) => {
    event?.preventDefault();
    setSession("mock-session-id");
    setEvaluation("mock-eval-id");
    setStatus({ text: "Mock login thành công.", type: "success" });
    localStorage.setItem("aic_sessionId", "mock-session-id");
    localStorage.setItem("aic_evaluationId", "mock-eval-id");
  };

  const clearPrepared = () => {
    localStorage.removeItem("preparedSubmissionBody");
    setPrepared("");
    setUsePrepared(false);
  };

  function fillPrepared(checked: boolean) {
    setUsePrepared(checked);
    if (!checked) return;
    if (!prepared)
      return setStatus({
        text: "Không có prepared body để dùng.",
        type: "error",
      });
    try {
      const item = JSON.parse(prepared).answerSets[0].answers[0];
      if (item.mediaItemName !== undefined) {
        setTask("kis");
        setVideo(item.mediaItemName);
        setStart(String(item.start));
        setEnd(String(item.end));
      } else if (item.text?.startsWith("QA-")) {
        const parts = item.text.split("-");
        setTask("qa");
        setAnswer(parts.slice(1, -2).join("-"));
        setVideo(parts.at(-2) ?? "");
        setStart(parts.at(-1) ?? "");
      } else if (item.text?.startsWith("TR-")) {
        const text = item.text.slice(3);
        const split = text.indexOf("-");
        setTask("trake");
        setVideo(text.slice(0, split));
        setFrames(text.slice(split + 1));
      }
    } catch {
      setStatus({ text: "Không thể parse prepared body.", type: "error" });
    }
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    if (!session || !evaluation)
      return setStatus({
        text: "Session or Evaluation ID is missing. Please log in again.",
        type: "error",
      });
    const body =
      task === "kis"
        ? {
            answerSets: [
              {
                answers: [
                  {
                    mediaItemName: video,
                    start: Number(start),
                    end: Number(end),
                  },
                ],
              },
            ],
          }
        : task === "qa"
          ? {
              answerSets: [
                { answers: [{ text: `QA-${answer}-${video}-${start}` }] },
              ],
            }
          : { answerSets: [{ answers: [{ text: `TR-${video}-${frames}` }] }] };

    setResponse({
      ok: true,
      code: "200 OK (MOCK)",
      body: JSON.stringify(body, null, 2),
    });
    setStatus({ text: "Submission processed.", type: "success" });
  }

  return (
    <div className="bg-gray-100 font-sans min-h-screen flex items-center justify-center p-4">
      <div className="bg-white shadow-2xl rounded-xl p-6 md:p-8 w-full max-w-3xl space-y-6">
        <div className="flex justify-between items-center">
          <h1 className="text-2xl font-bold text-gray-800">
            AIC 2025 Submission Tool
          </h1>
          <button
            onClick={goSearch}
            className="px-4 py-2 rounded-lg border border-gray-300 text-gray-700 hover:bg-gray-50"
          >
            ← Quay về trang tìm kiếm
          </button>
        </div>

        {/* Step 1: Login */}
        <div className="border border-gray-200 rounded-lg p-5">
          <h2 className="text-2xl font-semibold text-gray-800 mb-4">
            Step 1: Login
          </h2>
          <form onSubmit={mockLogin} className="space-y-4">
            <div>
              <label
                htmlFor="base-url"
                className="block text-sm font-medium text-gray-700 mb-1"
              >
                API Base URL
              </label>
              <input
                type="text"
                id="base-url"
                className="w-full p-3 border border-gray-300 rounded-lg bg-white focus:ring-2 focus:ring-blue-500"
                value="https://eventretrieval.oj.io.vn/api/v2"
                readOnly
              />
            </div>
            <div>
              <label
                htmlFor="username"
                className="block text-sm font-medium text-gray-700 mb-1"
              >
                Username
              </label>
              <input
                type="text"
                id="username"
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                placeholder="Your team's username"
                required
              />
            </div>
            <div>
              <label
                htmlFor="password"
                className="block text-sm font-medium text-gray-700 mb-1"
              >
                Password
              </label>
              <input
                type="password"
                id="password"
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                placeholder="Your team's password"
                required
              />
            </div>
            <button
              type="submit"
              className="w-full bg-blue-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-blue-700 transition duration-300 shadow-lg"
            >
              Login & Fetch Evaluation
            </button>
          </form>
          <div className="mt-3 flex gap-2">
            <button
              onClick={() => mockLogin()}
              className="bg-gray-600 text-white font-semibold py-2 px-4 rounded hover:bg-gray-700 transition"
            >
              Mock Login
            </button>
            <button
              onClick={() => {
                setSession(null);
                setEvaluation("");
                setStatus({ text: "Đã đăng xuất.", type: "success" });
              }}
              className="bg-red-600 text-white font-semibold py-2 px-4 rounded hover:bg-red-700 transition"
            >
              Logout
            </button>
            <button
              onClick={clearPrepared}
              className="bg-gray-200 text-gray-800 font-semibold py-2 px-4 rounded hover:bg-gray-300 transition"
            >
              Clear Prepared Body
            </button>
          </div>
          {status.text && (
            <div
              className={`mt-4 font-semibold ${status.type === "error" ? "text-red-600" : "text-green-600"}`}
            >
              {status.type === "error" ? (
                <strong>Error:</strong>
              ) : (
                <strong>Success:</strong>
              )}{" "}
              {status.text}
            </div>
          )}
        </div>

        {/* Step 2: Active Evaluation */}
        {session && (
          <div className="border border-gray-200 rounded-lg p-5">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">
              Step 2: Active Evaluation
            </h2>
            <div className="space-y-2 font-mono text-sm">
              <p>
                <strong>Evaluation Name:</strong>{" "}
                <span className="text-gray-700">Mock Evaluation</span>
              </p>
              <p>
                <strong>Evaluation ID:</strong>{" "}
                <span className="text-blue-600">{evaluation}</span>
              </p>
            </div>
          </div>
        )}

        {/* Step 3: Submit Answer */}
        {session && (
          <div className="border border-gray-200 rounded-lg p-5">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">
              Step 3: Submit Answer
            </h2>
            <form onSubmit={submit} className="space-y-4">
              <div className="bg-yellow-50 border border-yellow-200 rounded p-3 space-y-2">
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="use-prepared-body"
                    className="h-4 w-4"
                    checked={usePrepared}
                    onChange={(e) => fillPrepared(e.target.checked)}
                  />
                  <label
                    htmlFor="use-prepared-body"
                    className="text-sm text-yellow-800"
                  >
                    Use prepared body (được gửi từ trang tìm kiếm)
                  </label>
                </div>
                <label className="block text-sm font-medium text-yellow-900">
                  Prepared Body
                </label>
                <textarea
                  className="w-full p-2 border border-yellow-200 rounded font-mono text-xs"
                  rows={6}
                  value={prepared}
                  readOnly
                ></textarea>
              </div>

              <div>
                <label
                  htmlFor="submission-type"
                  className="block text-sm font-medium text-gray-700 mb-1"
                >
                  Submission Type
                </label>
                <select
                  id="submission-type"
                  className="w-full p-3 border border-gray-300 rounded-lg bg-white focus:ring-2 focus:ring-blue-500"
                  value={task}
                  onChange={(e) => setTask(e.target.value as Task)}
                >
                  <option value="kis">Known-Item Search (KIS)</option>
                  <option value="qa">Q&amp;A</option>
                  <option value="trake">TRAKE</option>
                </select>
              </div>

              {task === "kis" && (
                <div className="space-y-4">
                  <div>
                    <label
                      htmlFor="kis-video-id"
                      className="block text-sm font-medium text-gray-700 mb-1"
                    >
                      Video ID (mediaItemName)
                    </label>
                    <input
                      type="text"
                      id="kis-video-id"
                      className="w-full p-3 border border-gray-300 rounded-lg font-mono text-sm"
                      placeholder="e.g., v_0001"
                      value={video}
                      onChange={(e) => setVideo(e.target.value)}
                      required
                    />
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label
                        htmlFor="kis-start-time"
                        className="block text-sm font-medium text-gray-700 mb-1"
                      >
                        Start Time (ms)
                      </label>
                      <input
                        type="number"
                        id="kis-start-time"
                        className="w-full p-3 border border-gray-300 rounded-lg font-mono text-sm"
                        placeholder="e.g., 123456"
                        value={start}
                        onChange={(e) => setStart(e.target.value)}
                        required
                      />
                    </div>
                    <div>
                      <label
                        htmlFor="kis-end-time"
                        className="block text-sm font-medium text-gray-700 mb-1"
                      >
                        End Time (ms)
                      </label>
                      <input
                        type="number"
                        id="kis-end-time"
                        className="w-full p-3 border border-gray-300 rounded-lg font-mono text-sm"
                        placeholder="e.g., 123999"
                        value={end}
                        onChange={(e) => setEnd(e.target.value)}
                        required
                      />
                    </div>
                  </div>
                </div>
              )}

              {task === "qa" && (
                <div className="space-y-4">
                  <div>
                    <label
                      htmlFor="qa-answer"
                      className="block text-sm font-medium text-gray-700 mb-1"
                    >
                      Answer (Text)
                    </label>
                    <input
                      type="text"
                      id="qa-answer"
                      className="w-full p-3 border border-gray-300 rounded-lg font-mono text-sm"
                      placeholder="e.g., 12345"
                      value={answer}
                      onChange={(e) => setAnswer(e.target.value)}
                      required
                    />
                  </div>
                  <div>
                    <label
                      htmlFor="qa-video-id"
                      className="block text-sm font-medium text-gray-700 mb-1"
                    >
                      Video ID
                    </label>
                    <input
                      type="text"
                      id="qa-video-id"
                      className="w-full p-3 border border-gray-300 rounded-lg font-mono text-sm"
                      placeholder="e.g., v_0002"
                      value={video}
                      onChange={(e) => setVideo(e.target.value)}
                      required
                    />
                  </div>
                  <div>
                    <label
                      htmlFor="qa-time"
                      className="block text-sm font-medium text-gray-700 mb-1"
                    >
                      Time (ms)
                    </label>
                    <input
                      type="number"
                      id="qa-time"
                      className="w-full p-3 border border-gray-300 rounded-lg font-mono text-sm"
                      placeholder="e.g., 456789"
                      value={start}
                      onChange={(e) => setStart(e.target.value)}
                      required
                    />
                  </div>
                </div>
              )}

              {task === "trake" && (
                <div className="space-y-4">
                  <div>
                    <label
                      htmlFor="trake-video-id"
                      className="block text-sm font-medium text-gray-700 mb-1"
                    >
                      Video ID
                    </label>
                    <input
                      type="text"
                      id="trake-video-id"
                      className="w-full p-3 border border-gray-300 rounded-lg font-mono text-sm"
                      placeholder="e.g., v_0003"
                      value={video}
                      onChange={(e) => setVideo(e.target.value)}
                      required
                    />
                  </div>
                  <div>
                    <label
                      htmlFor="trake-frame-list"
                      className="block text-sm font-medium text-gray-700 mb-1"
                    >
                      List Frame IDs (comma-separated)
                    </label>
                    <input
                      type="text"
                      id="trake-frame-list"
                      className="w-full p-3 border border-gray-300 rounded-lg font-mono text-sm"
                      placeholder="e.g., 12345,67890,98765"
                      value={frames}
                      onChange={(e) => setFrames(e.target.value)}
                      required
                    />
                  </div>
                </div>
              )}

              <button
                type="submit"
                className="w-full bg-green-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-green-700 transition duration-300 shadow-lg"
              >
                Submit Answer
              </button>
            </form>

            {response && (
              <div className="mt-6 space-y-4">
                <h3 className="text-xl font-semibold text-gray-800">
                  Submission Response
                </h3>
                <div>
                  <h4 className="font-medium text-gray-700">Status:</h4>
                  <div
                    className={`mt-1 px-4 py-2 bg-gray-100 rounded-lg font-mono font-bold ${response.ok ? "text-green-600" : "text-red-600"}`}
                  >
                    {response.code}
                  </div>
                </div>
                <div>
                  <h4 className="font-medium text-gray-700">Body:</h4>
                  <pre className="bg-gray-900 text-white text-sm p-4 rounded-lg overflow-x-auto max-h-80">
                    <code>{response.body}</code>
                  </pre>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
