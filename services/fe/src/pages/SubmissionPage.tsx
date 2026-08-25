import { FormEvent, useState } from "react";
import type { Task } from "../types";

export default function SubmissionPage({ goSearch }: { goSearch: () => void }) {
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
  const fillPrepared = (checked: boolean) => {
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
  };
  const submit = (event: FormEvent) => {
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
  };
  const field = (
    label: string,
    value: string,
    setValue: (value: string) => void,
    type = "text",
  ) => (
    <label className="block text-sm font-medium text-gray-700">
      {label}
      <input
        type={type}
        className="w-full p-3 border border-gray-300 rounded-lg font-mono text-sm"
        value={value}
        onChange={(event) => setValue(event.target.value)}
        required
      />
    </label>
  );
  return (
    <div className="bg-gray-100 font-sans min-h-screen flex items-center justify-center p-4">
      <div className="bg-white shadow-2xl rounded-xl p-6 md:p-8 w-full max-w-3xl space-y-6">
        <header className="flex justify-between items-center">
          <h1 className="text-2xl font-bold text-gray-800">
            AIC 2025 Submission Tool
          </h1>
          <button
            onClick={goSearch}
            className="px-4 py-2 rounded-lg border border-gray-300"
          >
            ← Quay về trang tìm kiếm
          </button>
        </header>
        <section className="border border-gray-200 rounded-lg p-5">
          <h2 className="text-2xl font-semibold mb-4">Step 1: Login</h2>
          <form onSubmit={mockLogin} className="space-y-4">
            <label className="block text-sm font-medium">
              API Base URL
              <input
                className="w-full p-3 border rounded-lg"
                value="https://eventretrieval.oj.io.vn/api/v2"
                readOnly
              />
            </label>
            <label className="block text-sm font-medium">
              Username
              <input
                className="w-full p-3 border rounded-lg"
                placeholder="Your team's username"
                required
              />
            </label>
            <label className="block text-sm font-medium">
              Password
              <input
                type="password"
                className="w-full p-3 border rounded-lg"
                placeholder="Your team's password"
                required
              />
            </label>
            <button className="w-full bg-blue-600 text-white font-bold py-3 rounded-lg">
              Login & Fetch Evaluation
            </button>
          </form>
          <div className="mt-3 flex gap-2">
            <button
              onClick={() => mockLogin()}
              className="bg-gray-600 text-white py-2 px-4 rounded"
            >
              Mock Login
            </button>
            <button
              onClick={() => {
                setSession(null);
                setEvaluation("");
              }}
              className="bg-red-600 text-white py-2 px-4 rounded"
            >
              Logout
            </button>
            <button
              onClick={clearPrepared}
              className="bg-gray-200 py-2 px-4 rounded"
            >
              Clear Prepared Body
            </button>
          </div>
          {status.text && (
            <p
              className={
                status.type === "error" ? "text-red-600" : "text-green-600"
              }
            >
              {status.text}
            </p>
          )}
        </section>
        {session && (
          <section className="border border-gray-200 rounded-lg p-5">
            <h2 className="text-2xl font-semibold mb-4">
              Step 2: Active Evaluation
            </h2>
            <p>
              <strong>Evaluation Name:</strong> Mock Evaluation
            </p>
            <p>
              <strong>Evaluation ID:</strong> {evaluation}
            </p>
          </section>
        )}
        {session && (
          <section className="border border-gray-200 rounded-lg p-5">
            <h2 className="text-2xl font-semibold mb-4">
              Step 3: Submit Answer
            </h2>
            <form onSubmit={submit} className="space-y-4">
              <div className="bg-yellow-50 border border-yellow-200 rounded p-3">
                <label>
                  <input
                    type="checkbox"
                    checked={usePrepared}
                    onChange={(event) => fillPrepared(event.target.checked)}
                  />{" "}
                  Use prepared body
                </label>
                <textarea
                  className="w-full p-2 border rounded font-mono text-xs mt-2"
                  rows={6}
                  value={prepared}
                  readOnly
                />
              </div>
              <label className="block text-sm font-medium">
                Submission Type
                <select
                  className="w-full p-3 border rounded-lg"
                  value={task}
                  onChange={(event) => setTask(event.target.value as Task)}
                >
                  <option value="kis">Known-Item Search (KIS)</option>
                  <option value="qa">Q&amp;A</option>
                  <option value="trake">TRAKE</option>
                </select>
              </label>
              {task === "kis" && (
                <div className="space-y-4">
                  {field("Video ID (mediaItemName)", video, setVideo)}
                  <div className="grid md:grid-cols-2 gap-4">
                    {field("Start Time (ms)", start, setStart, "number")}
                    {field("End Time (ms)", end, setEnd, "number")}
                  </div>
                </div>
              )}
              {task === "qa" && (
                <div className="space-y-4">
                  {field("Answer (Text)", answer, setAnswer)}
                  {field("Video ID", video, setVideo)}
                  {field("Time (ms)", start, setStart, "number")}
                </div>
              )}
              {task === "trake" && (
                <div className="space-y-4">
                  {field("Video ID", video, setVideo)}
                  {field("List Frame IDs (comma-separated)", frames, setFrames)}
                </div>
              )}
              <button className="w-full bg-green-600 text-white font-bold py-3 rounded-lg">
                Submit Answer
              </button>
            </form>
            {response && (
              <pre className="bg-gray-900 text-white text-sm p-4 rounded-lg mt-6 overflow-x-auto">
                {response.code}
                {"\n"}
                {response.body}
              </pre>
            )}
          </section>
        )}
      </div>
    </div>
  );
}
