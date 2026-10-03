import { readdirSync, readFileSync } from "node:fs";

const root = new URL("../", import.meta.url);
const files = readdirSync(new URL("_parts/", root)).filter((f) => f.endsWith(".html"));

// Same rules facts.py enforces, so the two never disagree.
const PROSE = [
  ["NOTXBUTY", /\b(is not (just|merely|only)|isn't (just|merely)|it's not (just|merely|only)|not (just|merely|only) [^.,;]{3,70} but|rather than [^.,;]{3,70},)\b/i],
  ["SAYING", /\b(at its core|the real (question|issue|challenge)|what (really )?matters|fundamentally|the heart of the matter|the bottom line)\b/i],
  ["CLOSER", /\b(that'?s the real win|marks? a pivotal|stands as a testament|plays a key role|setting the stage for|evolving landscape|despite these challenges|looking ahead|the road ahead|ushering in)\b/i],
  ["RUNUP", /\b(this is what makes|this is why|this is the (economic|direct|key) |let'?s (dive|take|explore)|here'?s the thing|that is the real win)\b/i],
  ["ARGUEING", /\b(this isn'?t (mainly |really )?about|i'?m not (saying|arguing)|some might say|a tempting approach|one might be tempted|an obvious approach)\b/i],
  ["CHAT", /\b(i hope this helps|let me know if|would you like|want me to|great question)\b/i],
  ["GUESS", /\b(as of (my last|today)|based on available information|not publicly available|it is (believed|thought) that)\b/i],
  ["AVOIDS", /\b(serves as|stands as|functions as|boasts)\b/i],
  ["VAGUE", /\b(associated with|in association with|in connection with|is linked to|is tied to)\b/i],
  ["INGRIDER", /\b(highlighting|underscoring|emphasizing|cultivating|fostering a|encompassing|showcasing)\b/i],
  ["WORD", /\b(not merely|crucial|additionally|furthermore|moreover|delve|pivotal|seamless(ly)?|tapestry|underscore|vibrant|meticulous(ly)?|foster(ing)?|paradigm|holistic|comprehensive|notably|importantly|ultimately|robust|intuitive|landscape)\b/i],
];

let n = 0;
for (const f of files) {
  const lines = readFileSync(new URL(`_parts/${f}`, root), "utf8").split("\n");
  lines.forEach((ln, i) => {
    if (/<script|<\/script|<style|<\/style/.test(ln)) return;
    const t = ln.replace(/<[^>]+>/g, "").replace(/&[a-z#0-9]+;/g, " ").replace(/\s+/g, " ").trim();
    if (t.length < 12) return;
    const hits = PROSE.filter(([, re]) => re.test(t)).map(([k]) => k);
    if (hits.length) { n++; console.log(`${f}:${i + 1} [${hits.join(",")}] ${t.slice(0, 190)}`); }
  });
}
console.log(`--- ${n} residual hits ---`);