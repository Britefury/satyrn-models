/** Markdown and syntax-highlighting helpers. */

import hljs from "highlight.js/lib/core";
import python from "highlight.js/lib/languages/python";
import { Marked } from "marked";
import { markedHighlight } from "marked-highlight";

// Only the Python grammar is registered: every generated dataset row contains
// Python code, so pulling in the full highlight.js bundle is unnecessary.
hljs.registerLanguage("python", python);

export function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/** Highlight a raw Python snippet, returning HTML for use inside a <code>. */
export function highlightPython(code: string): string {
  return hljs.highlight(code, { language: "python" }).value;
}

const markdown = new Marked(
  markedHighlight({
    langPrefix: "hljs language-",
    highlight(code, language) {
      if (language && hljs.getLanguage(language)) {
        return hljs.highlight(code, { language }).value;
      }
      return escapeHtml(code);
    },
  }),
);

/** Render a markdown string, highlighting fenced code blocks. */
export function renderMarkdown(source: string): string {
  const result = markdown.parse(source, { async: false });
  return typeof result === "string" ? result : source;
}
