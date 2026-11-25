import { useCallback, useEffect, useRef, useState } from "react";

/** Copy text to the clipboard and expose a short-lived "copied" flag for feedback. */
export function useCopy(resetMs = 1800) {
  const [copied, setCopied] = useState(false);
  const timer = useRef(null);

  useEffect(() => () => clearTimeout(timer.current), []);

  const copy = useCallback(
    async (text) => {
      try {
        await navigator.clipboard.writeText(text);
        setCopied(true);
        clearTimeout(timer.current);
        timer.current = setTimeout(() => setCopied(false), resetMs);
      } catch {
        // clipboard blocked (http on LAN): nothing to do
      }
    },
    [resetMs]
  );

  return [copied, copy];
}
