import { useState } from "react";

type Hit = { scene_id: number; score: number; rank: number };

export default function App() {
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<Hit[]>([]);
  const [ms, setMs] = useState<number | null>(null);

  async function run() {
    const r = await fetch("/api/search", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text: q, top_k: 10 }),
    });
    const d = await r.json();
    setHits(d.hits ?? []);
    setMs(d.timings?.total_ms ?? null);
  }

  return (
    <main style={{ fontFamily: "system-ui", padding: 24, maxWidth: 720 }}>
      <h1>rackfocus</h1>
      <div style={{ display: "flex", gap: 8 }}>
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && run()}
          placeholder="Mô tả cảnh cần tìm..."
          style={{ flex: 1, padding: 8 }}
        />
        <button onClick={run}>Tìm</button>
      </div>
      {ms !== null && <p>{hits.length} kết quả · {ms.toFixed(2)}ms</p>}
      <ul>
        {hits.map((h) => (
          <li key={h.scene_id}>scene {h.scene_id} — {h.score.toFixed(3)}</li>
        ))}
      </ul>
    </main>
  );
}