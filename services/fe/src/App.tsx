import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { FormEvent, useEffect, useMemo, useState } from "react";
import type { SearchHit } from "./api/types";
import { useSearch } from "./hooks/useSearch";
import "./styles/global.css";

type Task = "kis" | "qa" | "trake";
// SearchHit đã có video_name/frame/keyframe_url thật (services/be/src/app/api/search.py)
// -- url/video là alias tiện dùng trong UI, point_id là khoá duy nhất thật (không có
// scene_id toàn cục nào cả, scene_idx chỉ là index trong 1 video).
type Result = SearchHit & {
  url: string;
  video: string;
};
const topics = [
  "tin tức",
  "Tin tức+múa lân",
  "Tin tức+xe đạp",
  "Tin tức+dạy học online",
  "Tin tức+chương trình nấu ăn",
];

export default function App() {
  const [route, setRoute] = useState(window.location.pathname);

  useEffect(() => {
    const assets = [
      [
        "bs-css",
        "link",
        "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css",
      ],
      [
        "bs-icons",
        "link",
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.2/font/bootstrap-icons.min.css",
      ],
      ["tw-script", "script", "https://cdn.tailwindcss.com"],
    ];
    assets.forEach(([id, tag, src]) => {
      if (document.getElementById(id)) return;
      const element = document.createElement(tag);
      element.id = id;
      element.setAttribute(
        tag === "link" ? "rel" : "src",
        tag === "link" ? "stylesheet" : src,
      );
      if (tag === "link") element.setAttribute("href", src);
      document.head.appendChild(element);
    });
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
