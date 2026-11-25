import { useEffect, useState } from "react";
import { storage } from "./utils.js";

export function useMediaQuery(query) {
  const [matches, setMatches] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const mql = window.matchMedia(query);
    const onChange = () => setMatches(mql.matches);
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, [query]);
  return matches;
}

export function useTheme() {
  const [theme, setTheme] = useState(() => storage.get("rag_theme", "system"));
  const prefersDark = useMediaQuery("(prefers-color-scheme: dark)");
  const resolved = theme === "system" ? (prefersDark ? "dark" : "light") : theme;

  useEffect(() => {
    storage.set("rag_theme", theme);
    document.documentElement.dataset.theme = resolved;
    document.documentElement.style.colorScheme = resolved;
  }, [theme, resolved]);

  return [theme, setTheme];
}
