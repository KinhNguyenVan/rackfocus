import { FormEvent, useEffect, useMemo, useState } from "react";
import { useSearch } from "../hooks/useSearch";
import type { Result, Task } from "../types";

const topics = [
  "tin tức",
  "Tin tức+múa lân",
  "Tin tức+xe đạp",
  "Tin tức+dạy học online",
  "Tin tức+chương trình nấu ăn",
];

export default function SearchPage({
  goSubmission,
}: {
  goSubmission: () => void;
}) {
  const [query, setQuery] = useState("");
  const [submitted, setSubmitted] = useState("");
  const [flag, setFlag] = useState("");
  const [service, setService] = useState("image");
  const [chosenTopics, setChosenTopics] = useState<string[]>([]);
  const [selected, setSelected] = useState<Result[]>([]);
  const [task, setTask] = useState<Task>("kis");
  const [qaAnswer, setQaAnswer] = useState("");
  const [output, setOutput] = useState("");
  const [framesOpen, setFramesOpen] = useState(false);
  const [contextMenu, setContextMenu] = useState({
    visible: false,
    x: 0,
    y: 0,
    targetUrl: "",
  });
  const { hits, totalMs, loading, error } = useSearch(submitted);
  const results = useMemo(
    () =>
      hits.map((hit) => ({
        ...hit,
        video: hit.video_name,
        frame: hit.frame,
        url: hit.keyframe_url,
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
    } catch {
      /* Ignore invalid saved state. */
    }
  }, []);
  useEffect(() => {
    localStorage.setItem(
      "searchState",
      JSON.stringify({
        query,
        flagValue: flag,
        serviceValue: service,
        selectedTopics: chosenTopics,
      }),
    );
  }, [query, flag, service, chosenTopics]);
  useEffect(() => {
    const close = () =>
      setContextMenu((current) => ({ ...current, visible: false }));
    document.addEventListener("click", close);
    return () => document.removeEventListener("click", close);
  }, []);

  const toggle = (result: Result) =>
    setSelected((items) =>
      items.some((item) => item.point_id === result.point_id)
        ? items.filter((item) => item.point_id !== result.point_id)
        : [...items, result],
    );
  const search = (event: FormEvent) => {
    event.preventDefault();
    if (query.trim()) {
      setSelected([]);
      setSubmitted(query.trim());
    }
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
    setSubmitted("");
    setSelected([]);
    setOutput("");
  };
  const exportResult = () => {
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
                    start: first.frame * 33,
                    end: last.frame * 33,
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
                  { text: `QA-${qaAnswer}-${first.video}-${first.frame * 33}` },
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
                    text: `TR-${first.video}-${selected.map((item) => item.frame).join(",")}`,
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
  };
  const send = () => {
    if (!output)
      return alert("Vui lòng xuất kết quả trước khi gửi sang trang nộp bài.");
    localStorage.setItem("preparedSubmissionBody", output);
    goSubmission();
  };

  return (
    <div id="pageWrapper" className="d-flex">
      <aside
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
                    {index + 1}. {item.video} / {item.scene_idx}
                  </small>
                </div>
              </div>
            </div>
          ))}
        </div>
      </aside>
      {contextMenu.visible && (
        <div
          id="contextMenu"
          className="position-absolute bg-white border rounded shadow-sm"
          style={{ top: contextMenu.y, left: contextMenu.x, zIndex: 2000 }}
        >
          <button
            className="dropdown-item"
            onClick={() => {
              setFramesOpen(true);
              alert("Mock: Load frames (chưa nối API)");
            }}
          >
            📽️ Xem 25 frames trước/sau
          </button>
          <button
            className="dropdown-item"
            onClick={() => navigator.clipboard.writeText(contextMenu.targetUrl)}
          >
            🔗 Copy URL
          </button>
          <button
            className="dropdown-item"
            onClick={() => alert("Mock: Mở video gốc (chưa nối map)")}
          >
            ▶️ Mở video gốc trong tab mới
          </button>
        </div>
      )}
      <main
        id="mainContent"
        className="flex-grow-1"
        style={{
          marginLeft: "200px",
          zIndex: 1,
        }}
      >
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
              <button className="btn btn-sm btn-primary" onClick={goSubmission}>
                Trang nộp bài
              </button>
            </div>
          </div>
        </nav>
        <div className="container">
          <div className="row g-3">
            <section className="col-12 col-md-6">
              <div className="card h-100 shadow-sm">
                <div className="card-body">
                  <h5 className="card-title">
                    <i className="bi bi-image" /> Image Search
                  </h5>
                  <form onSubmit={search}>
                    <div className="input-group mb-3">
                      <input
                        className="form-control"
                        placeholder="Search by text..."
                        value={query}
                        onChange={(event) => setQuery(event.target.value)}
                      />
                      <input
                        className="form-control"
                        placeholder="Enter flag..."
                        value={flag}
                        onChange={(event) => setFlag(event.target.value)}
                      />
                      <select
                        className="form-select"
                        value={service}
                        onChange={(event) => setService(event.target.value)}
                      >
                        <option value="image">Image</option>
                        <option value="caption">Caption</option>
                        <option value="ocr">OCR</option>
                      </select>
                      <button
                        className="btn btn-outline-secondary"
                        type="submit"
                      >
                        <i className="bi bi-mic"></i>
                      </button>
                      <button
                        className="btn btn-sm btn-outline-danger ms-2"
                        type="button"
                        onClick={clear}
                        title="Clear saved search state"
                      >
                        Clear
                      </button>
                    </div>
                  </form>
                  <div className="mb-3">
                    <label className="form-label small fw-semibold">
                      Topics:
                    </label>
                    <div className="d-flex flex-wrap gap-2">
                      {topics.map((topic) => (
                        <label className="form-check small" key={topic}>
                          <input
                            className="form-check-input topic-checkbox"
                            type="checkbox"
                            checked={chosenTopics.includes(topic)}
                            onChange={() =>
                              setChosenTopics((items) =>
                                items.includes(topic)
                                  ? items.filter((item) => item !== topic)
                                  : [...items, topic],
                              )
                            }
                          />{" "}
                          {topic === "tin tức" ? "Tin tức" : topic}
                        </label>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </section>
            <section className="col-12 col-md-6">
              <div className="card h-100 shadow-sm">
                <div className="card-body">
                  <h5 className="card-title">
                    <i className="bi bi-ui-checks" /> Xuất thông báo kết quả
                  </h5>
                  <label className="form-label">Kiểu câu hỏi</label>
                  <select
                    className="form-select mb-2"
                    value={task}
                    onChange={(event) => setTask(event.target.value as Task)}
                  >
                    <option value="kis">kis</option>
                    <option value="qa">qa</option>
                    <option value="trake">trake</option>
                  </select>
                  {task === "qa" && (
                    <input
                      className="form-control mb-2"
                      placeholder="Nhập answer..."
                      value={qaAnswer}
                      onChange={(event) => setQaAnswer(event.target.value)}
                    />
                  )}
                  <div className="d-grid gap-2">
                    <button className="btn btn-primary" onClick={exportResult}>
                      Xuất kết quả
                    </button>
                    <button className="btn btn-outline-primary" onClick={send}>
                      Gửi sang trang nộp bài
                    </button>
                  </div>
                  <textarea
                    className="form-control mt-2"
                    rows={8}
                    readOnly
                    value={output}
                  />
                </div>
              </div>
            </section>
          </div>
          <section className="card mt-3 mb-4">
            <div className="card-body">
              <h5 className="card-title">
                <i className="bi bi-database" /> Images results
              </h5>
              {loading && (
                <div className="spinner-border text-primary" role="status" />
              )}{" "}
              {error && <p className="text-danger">{error}</p>}
              <div className="row row-cols-2 row-cols-sm-3 row-cols-md-5 g-2">
                {results.map((result) => {
                  const selectedIndex = selected.findIndex(
                    (item) => item.point_id === result.point_id,
                  );
                  return (
                    <div className="col p-2 text-center" key={result.point_id}>
                      <div className="card position-relative">
                        <img
                          src={result.url}
                          className="card-img-top main-img"
                          alt="scene"
                          onContextMenu={(event) => {
                            event.preventDefault();
                            setContextMenu({
                              visible: true,
                              x: event.pageX,
                              y: event.pageY,
                              targetUrl: result.url,
                            });
                          }}
                        />
                        <div className="card-body p-2">
                          <input
                            type="checkbox"
                            checked={selectedIndex >= 0}
                            onChange={() => toggle(result)}
                          />{" "}
                          <small>
                            {result.video} / {result.scene_idx}
                          </small>
                        </div>
                        {selectedIndex >= 0 && (
                          <span className="badge bg-primary position-absolute top-0 start-0 m-1">
                            {selectedIndex + 1}
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
              {totalMs !== null && (
                <small className="text-muted">
                  {results.length} results · {totalMs.toFixed(2)} ms
                </small>
              )}
            </div>
          </section>
        </div>
      </main>
      <aside
        id="frameSidebar"
        className="position-fixed top-0 end-0 bg-light shadow h-100"
        style={{
          width: framesOpen ? "400px" : "40px",
          transition: "0.3s",
          zIndex: 1050,
        }}
      >
        <div
          className="d-flex justify-content-center align-items-center h-100 border-start"
          style={{ width: "40px", float: "left", cursor: "pointer" }}
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
              width: "360px",
              float: "left",
              padding: "10px",
              height: "100%",
              overflowY: "auto",
              display: "block",
            }}
          >
            <div className="d-flex justify-content-between align-items-center border-bottom mb-2">
              <h6 className="mb-0">Frames</h6>
            </div>
            <div className="alert alert-warning p-2 small">
              Mock: /frames?url=... chưa nối BE.
            </div>
          </div>
        )}
      </aside>
      <div className="footer">&copy; 2025 Galaxy AI</div>
    </div>
  );
}
