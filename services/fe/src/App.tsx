import { useEffect, useState } from "react";
import SearchPage from "./SearchPage";
import SubmissionPage from "./SubmissionPage";
import "./styles/global.css";

export default function App() {
  const [route, setRoute] = useState(window.location.pathname);

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
