import { useEffect, useState } from "react";
import SearchPage from "./pages/SearchPage";
import SubmissionPage from "./pages/SubmissionPage";
import "./styles/global.css";

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
