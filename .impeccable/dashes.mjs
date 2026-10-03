import { readdirSync, readFileSync } from "node:fs";

const root = new URL("../", import.meta.url);
const files = readdirSync(new URL("_parts/", root)).filter((f) => f.endsWith(".html"));

for (const f of files) {
  const lines = readFileSync(new URL(`_parts/${f}`, root), "utf8").split("\n");
  lines.forEach((ln, i) => {
    if (!ln.includes("—")) return;
    const inTag = /="[^"]*—/.test(ln);
    const inScript = /<script|<\/script/.test(ln);
    const visible = ln.replace(/<[^>]+>/g, "").replace(/&[a-z#0-9]+;/g, " ").replace(/\s+/g, " ").trim();
    const kind = inScript ? "script"
      : inTag ? "attribute"
      : /<code>|<span class="mono">/.test(ln) ? "code/mono"
      : /<!--/.test(ln) ? "comment"
      : /^(Table|Fig\.|Figure)\s*\d+\s*—/.test(visible) ? "caption-label"
      : /^[\w.\-]+(?:\s+et\s+al\.|\s+Team)?\.?\s*\d{4}\s*—/.test(visible) ? "bibliography"
      : /Qwen 3\.8-27B\s*—\s*The Architecture/.test(visible) ? "doc-title"
      : /^Results \d\d\d\d/.test(visible) ? "ref-year"
      : "PROSE?";
    console.log(`${f}:${i + 1} [${kind}] ${visible.slice(0, 110)}`);
  });
}