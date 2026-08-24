/* ==========================================================================
   Qwen 3.8-27B — From Scratch · interactive engine
   All widgets computed from the official config (Qwen/Qwen3.8-27B):
   d=5120, L=64 (48 DeltaNet + 16 GQA), dff=17408, vocab=248320,
   DeltaNet 16QK/48V × dh128 · GQA 24Q/4KV × dh256 (rotary 64), ctx 262144
   ========================================================================== */
"use strict";

/* Reload behavior:
   Without a hash or at top of page, ensure the page always starts cleanly at the top. */
if (!location.hash || location.hash === "#top") {
  if ("scrollRestoration" in history) {
    history.scrollRestoration = "manual";
  }
  window.scrollTo(0, 0);
  window.addEventListener("DOMContentLoaded", () => {
    if (!location.hash || location.hash === "#top") window.scrollTo(0, 0);
  });
}

/* ---------- figure inventory (running-head counter source-of-truth) ---------- */
const FIGURES = [
  "Dataflow",
  "MRoPE dial",
  "FFN activation curves",
  "Residual bus variance",
  "MTP forward+verify tree",
  "Pretrain loss",
  "β₂ recovery",
  "Test-time compute scaling",
  "Hardware roofline",
];

/* ---------- shared config + formatters ---------- */
const CFG = {
  d: 5120, layers: 64, nDelta: 48, nFull: 16, dff: 17408, vocab: 248320,
  dnQK: 16, dnV: 48, dnDh: 128,
  gqaQ: 24, gqaKV: 4, gqaDh: 256, rotDims: 64,
  visionTensors: 333, ctxNative: 262144,
  ropeTheta: 1e7
};
/* per-mixer parameter counts — single source for the W1 audit, layer explorer and totals */
CFG.ffnParams = 3 * CFG.d * CFG.dff;
CFG.dnParams = (CFG.dnQK * 2 + CFG.dnV) * CFG.dnDh * CFG.d   // fused in_proj (q, k, v)
  + CFG.dnV                                                  // β-gate: bias vector only (b_proj.bias)
  + CFG.dnV * CFG.dnDh * CFG.d;                              // out_proj
CFG.faParams = (CFG.gqaQ * 2 + CFG.gqaKV * 2 + CFG.gqaQ) * CFG.gqaDh * CFG.d;
/* grand total (27.58B): core + untied LM head + MTP (K·d² + shared d·V head) + vision tower */
CFG.mtpParams = 2 * (CFG.d * CFG.d) + CFG.d * CFG.vocab;
CFG.total = CFG.nDelta * (CFG.dnParams + CFG.ffnParams) + CFG.nFull * (CFG.faParams + CFG.ffnParams)
  + 2 * CFG.d * CFG.layers + CFG.d                           // RMSNorm gains: 2/layer + final
  + CFG.d * CFG.vocab                                        // token embeddings
  + CFG.d * CFG.vocab                                        // untied LM head
  + CFG.mtpParams + 0.9e9;                                   // MTP module + SigLIP-style vision tower (est.)
const fmtP = p => p >= 1e9 ? (p / 1e9).toFixed(2) + "B" : p >= 1e6 ? (p / 1e6).toFixed(1) + "M" : Math.round(p).toLocaleString();
const $ = id => document.getElementById(id);

/* pigment palette — mirror of the :root tokens in css/styles.css (keep in sync);
   SVG presentation attributes can't resolve var(), so JS needs the literals */
const PAL = {
  amber: "#b8341f", mint: "#1d6e6b", iris: "#2b3a8a", sky: "#8a5610",
  rose: "#8a3324", paper: "#f3ece0", ink: "#1a1612", muted: "#5a5046"
};

/* §08 acceptance convention (single source): E[tokens/pass] = (1−β^(K+1))/(1−β);
   net speedup S ≈ E/(1+γ_verify) with γ_verify = 0.15 — reproduces §08's
   published K=2, β=0.85 anchor: E = 2.57 tok/pass → S ≈ 2.24× (>2.2×) */
const MTP_GAMMA = 0.15;
const mtpEtoks = (beta, K) => (1 - Math.pow(beta, K + 1)) / (1 - beta);
const mtpSpeedup = (beta, K) => mtpEtoks(beta, K) / (1 + MTP_GAMMA);

/* escape text interpolated into widget innerHTML */
const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/* KaTeX auto-render for dynamically injected HTML (no-op if the CDN didn't load) */
function katexRender(el) {
  if (typeof window.renderMathInElement !== "function") return;
  window.renderMathInElement(el, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$",  right: "$",  display: false },
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
    ],
    throwOnError: false,
    ignoredClasses: ["nokatex"],
  });
}

/* segmented-button group: toggles .on + aria-pressed on matched children, then calls onChange(button) */
function bindGroup(container, onChange, sel = "button") {
  if (!container) return;
  const buttons = container.querySelectorAll(sel);
  buttons.forEach(b => b.addEventListener("click", () => {
    buttons.forEach(x => { x.classList.remove("on"); x.setAttribute("aria-pressed", "false"); });
    b.classList.add("on"); b.setAttribute("aria-pressed", "true");
    onChange(b);
  }));
}

/* div-based tab sets: focusable, button semantics, Enter/Space activate */
function enableTabs(container, itemSel) {
  if (!container) return;
  container.querySelectorAll(itemSel).forEach(it => {
    it.tabIndex = 0;
    it.setAttribute("role", "button");
    it.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); it.click(); }
    });
  });
}

/* slugify headings → ids used by .anchor links */
const slug = s => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 64);

/* ---------- scroll progress + nav scroll state ---------- */
(function () {
  const bar = $("progress");
  const nav = document.querySelector(".topnav");
  const brand = document.querySelector(".nav-brand");
  const onScroll = () => {
    const h = document.documentElement;
    const pct = h.scrollTop / Math.max(1, h.scrollHeight - h.clientHeight) * 100;
    bar.style.width = pct + "%";
    if (nav) nav.classList.toggle("scrolled", h.scrollTop > 12);
    if (h.scrollTop < 100 && location.hash && location.hash !== "#top") {
      history.replaceState(null, "", window.location.pathname + window.location.search);
    }
  };
  addEventListener("scroll", onScroll, { passive: true }); onScroll();

  if (brand) {
    brand.style.cursor = "pointer";
    brand.addEventListener("click", () => {
      if (location.hash) history.replaceState(null, "", window.location.pathname + window.location.search);
      const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
      window.scrollTo({ top: 0, behavior: reduced ? "auto" : "smooth" });
    });
  }
})();

/* ---------- hero: 64-bit architecture register ---------- */
(function () {
  const box = $("heroBits"); if (!box) return;
  const label = box.parentElement.querySelector(".bits-label");
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const cells = [];
  for (let i = 0; i < 64; i++) { const b = document.createElement("span"); b.className = "bit"; box.appendChild(b); cells.push(b); }
  const kind = i => (i % 4 === 3) ? "fm" : "fa";           // 3 DeltaNet + 1 Gated Attention, ×16
  if (reduced) { cells.forEach((b, i) => b.classList.add(kind(i))); label?.classList.add("show"); return; }
  // phase 1 — uninitialized memory flicker
  let ticks = 0;
  const flicker = setInterval(() => {
    cells.forEach(b => { b.classList.remove("fa", "fm"); if (Math.random() < .3) b.classList.add(Math.random() < .75 ? "fa" : "fm"); });
    if (++ticks >= 10) { clearInterval(flicker); settle(0); }
  }, 60);
  // phase 2 — settle into the 3:1 motif, one super-block at a time
  function settle(g) {
    if (g >= 16) { label?.classList.add("show"); return; }
    for (let k = 0; k < 4; k++) {
      const i = g * 4 + k;
      setTimeout(() => { cells[i].classList.remove("fa", "fm"); cells[i].classList.add(kind(i)); }, k * 45);
    }
    setTimeout(() => settle(g + 1), 140);
  }
})();

/* ---------- active nav link + running-head label ---------- */
(function () {
  const links = [...document.querySelectorAll("#navLinks a")];
  const cur = document.getElementById("navCurrent");
  const secs = links.map(a => document.querySelector(a.getAttribute("href"))).filter(Boolean);
  const io = new IntersectionObserver(es => {
    // pick the entry that's most visible among intersecting ones
    const visible = es.filter(e => e.isIntersecting);
    if (visible.length) {
      // closest to rootMargin center
      const best = visible.reduce((a, b) => b.intersectionRatio > a.intersectionRatio ? b : a);
      const active = links.find(l => l.getAttribute("href") === "#" + best.target.id);
      links.forEach(l => l.classList.toggle("on", l === active));
      if (cur) {
        if (active?.dataset.num) {
          cur.innerHTML = `<b>§${active.dataset.num}</b> · ${active.dataset.label}`;
          cur.classList.add("show");
        } else {
          cur.classList.remove("show");
        }
      }
    } else if (cur) {
      cur.classList.remove("show");
    }
  }, { rootMargin: "-30% 0px -55% 0px", threshold: [0, 0.25, 0.5, 0.75, 1] });
  secs.forEach(s => io.observe(s));
})();

/* ---------- contents panel toggle ---------- */
(function () {
  const btn = $("navToggle"), panel = $("navPanel");
  if (!btn || !panel) return;
  const setOpen = o => { panel.classList.toggle("open", o); btn.setAttribute("aria-expanded", String(o)); };
  btn.addEventListener("click", () => setOpen(!panel.classList.contains("open")));
  panel.addEventListener("click", e => { if (e.target.closest("a")) setOpen(false); });
  addEventListener("keydown", e => { if (e.key === "Escape") setOpen(false); });
  addEventListener("click", e => { if (!e.target.closest(".topnav")) setOpen(false); });
})();

/* ---------- back-to-top button ---------- */
(function () {
  const btn = $("totop");
  if (btn) {
    const onScroll = () => {
      const visible = scrollY > 600;
      btn.classList.toggle("on", visible);
      btn.tabIndex = visible ? 0 : -1;
      btn.setAttribute("aria-hidden", String(!visible));
    };
    addEventListener("scroll", onScroll, { passive: true });
    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    btn.addEventListener("click", () => {
      if (location.hash) history.replaceState(null, "", window.location.pathname + window.location.search);
      window.scrollTo({ top: 0, behavior: reduced ? "auto" : "smooth" });
    });
    onScroll();
  }
})();

/* ---------- heading anchor links (hover-only ¶) ---------- */
(function () {
  // first pass: ensure every h2/h3 inside a section has an id (so we can link to it)
  document.querySelectorAll("section h2, section h3").forEach(h => {
    if (!h.id) h.id = slug(h.textContent.replace(/¶/g, "")) || ("h-" + Math.random().toString(36).slice(2, 7));
  });
  // second pass: append the ¶ anchor link to each
  document.querySelectorAll("section h2[id], section h3[id]").forEach(h => {
    const a = document.createElement("a");
    a.className = "anchor"; a.href = "#" + h.id;
    a.setAttribute("aria-label", "Link to " + h.textContent.replace(/¶/g, "").trim());
    a.textContent = "¶";
    h.appendChild(a);
  });
})();

/* ---------- copy buttons on <pre class="code"> blocks ---------- */
(function () {
  document.querySelectorAll("pre.code").forEach(pre => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "code-copy"; b.textContent = "copy";
    b.setAttribute("aria-label", "Copy code to clipboard");
    b.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(pre.innerText);
        b.textContent = "copied ✓"; b.classList.add("ok");
        setTimeout(() => { b.textContent = "copy"; b.classList.remove("ok"); }, 1400);
      } catch { b.textContent = "copy failed"; setTimeout(() => b.textContent = "copy", 1600); }
    });
    pre.appendChild(b);
  });
})();

/* ---------- "next section" pills at the end of every section except the last ---------- */
(function () {
  const sects = [...document.querySelectorAll("main > section")];
  sects.forEach((s, i) => {
    if (i === sects.length - 1) return;
    const next = sects[i + 1];
    const a = document.createElement("a");
    a.className = "next-pill"; a.href = "#" + next.id;
    const t = next.querySelector("h2");
    const cleanText = (t?.textContent || next.id).replace(/¶/g, "").trim();
    a.innerHTML = `${cleanText} <span class="arrow" aria-hidden="true">→</span>`;
    s.appendChild(a);
  });
})();

/* ---------- callout glyphs ---------- */
(function () {
  const ICONS = {
    why:       '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><circle cx="8" cy="8" r="6.5"/><path d="M6.5 6.2c0-1 .7-1.7 1.7-1.7s1.7.7 1.7 1.6c0 .9-.5 1.3-1.1 1.6-.6.3-1 .7-1 1.4M8 11.4h.01"/></svg>',
    deep:      '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><path d="M2 8h12M2 8l3-3M2 8l3 3M14 8l-3-3M14 8l-3 3"/></svg>',
    intuition: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><path d="M8 1.5C4.4 1.5 1.5 4.4 1.5 8c0 2 .9 3.7 2.3 4.9L3 14.5h2.4l-.2-1.1A6.5 6.5 0 1 1 14.5 8c0 2.5-1.4 4.6-3.5 5.7"/><path d="M5 9.5c.5 1.2 1.7 2 3 2s2.5-.8 3-2"/></svg>',
    warn:      '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><path d="M8 1.5L1.5 13.5h13L8 1.5z"/><path d="M8 6v3.5M8 11.4h.01"/></svg>',
    note:      '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="12" height="10" rx="2"/><path d="M5 6.5h6M5 9h6M5 11.2h3.5"/></svg>'
  };
  document.querySelectorAll(".callout").forEach(c => {
    const variant = [...c.classList].find(k => ICONS[k]);
    if (!variant) return;
    const title = c.querySelector(".co-title");
    if (!title) return;
    const ic = document.createElement("span");
    ic.className = "ic"; ic.setAttribute("aria-hidden", "true");
    ic.innerHTML = ICONS[variant];
    title.insertBefore(ic, title.firstChild);
  });
})();

/* ============================================================
   W1 — Parameter accounting audit (#w-param)
   ============================================================ */
(function () {
  const out = $("paramOut"); if (!out) return;
  let withVision = true, withMtp = true;

  function compute() {
    const D = CFG.d, V = CFG.vocab, L = CFG.layers;
    const emb = D * V;
    const head = D * V;                                        // untied LM head (§14: embed + lm_head are separate)
    const norms = 2 * D * L + D;                               // 2 per layer + final
    // core weights only (embeddings included; head/MTP/vision excluded) — same expressions as CFG.total
    const coreExact = CFG.nDelta * (CFG.dnParams + CFG.ffnParams) + CFG.nFull * (CFG.faParams + CFG.ffnParams)
      + norms + emb;
    const mtp = withMtp ? CFG.mtpParams : 0;                   // K draft blocks (d² each) + shared d·V vocab head
    const vis = withVision ? 0.9e9 : 0;                        // SigLIP-style tower, est.
    return {
      emb, head, norms,
      attn: CFG.nFull * CFG.faParams,
      delta: CFG.nDelta * CFG.dnParams,
      ffn: L * CFG.ffnParams,
      mtp, vis,
      total: coreExact + head + mtp + vis
    };
  }

  function render() {
    const c = compute();
    const rows = [
      ["Token embedding (248,320 × 5120)", c.emb, ""],
      ["Gated DeltaNet layers — 48 × in/out/gate proj", c.delta, "a"],
      ["Gated Attention layers — 16 × q/k/v/gate/o proj", c.attn, "m"],
      ["SwiGLU FFNs — 64 × 3·d·dff", c.ffn, "i"],
      ["RMSNorm gains", c.norms, ""],
      ["LM head (untied) — 5120 × 248,320", c.head, ""],
    ];
    if (withMtp) rows.push(["MTP module (draft layers + heads)", c.mtp, ""]);
    if (withVision) rows.push(["Vision tower (SigLIP-style, est.)", c.vis, ""]);
    out.innerHTML =
      rows.map(([l, v, cls]) => `<div class="row"><span>${l}</span><span class="v ${cls}">${fmtP(v)}</span></div>`).join("") +
      `<div class="row" style="border-top:1px solid var(--line-strong);margin-top:6px;padding-top:10px">
         <span><b>total (LM ${withMtp ? "+ MTP " : ""}${withVision ? "+ vision" : ""})</b></span>
         <span class="big">${fmtP(c.total)}</span></div>` +
      `<div style="color:var(--faint);font-size:11px;margin-top:8px">checkpoint ≈ ${(c.total * 2 / 1e9).toFixed(1)} GB in BF16</div>`;
    // mirror the headline number to the hero stat card so they never disagree
    const heroEl = document.getElementById("statParams");
    if (heroEl) heroEl.textContent = fmtP(c.total);
  }
  [["paramVisionSeg", v => withVision = v === "1"], ["paramMtpSeg", v => withMtp = v === "1"]].forEach(([id, fn]) => {
    bindGroup($(id), b => { fn(b.dataset.v); render(); });
  });
  render();
})();

/* ============================================================
   Section 02: Tokenizer & Input Contract Engine
   ============================================================ */
(function () {
  /* ------------------------------------------------------------
     1. Widget: Interactive BPE Token Segmenter (#w-tokenizer-lab)
     ------------------------------------------------------------ */
  const tokLiveTokens = $("tokLiveTokens");
  const tokPresetSel = $("tokPresetSel");

  const PRESETS = {
    "code": {
      name: "Python Code: FlashAttention Kernel",
      raw: "def matmul_kernel(Q, K, V):\n    return flash_attn_func(Q, K, V, causal=True)",
      tokens: [
        { text: "def", id: 982 },
        { text: " mat", id: 2410 },
        { text: "mul", id: 18452 },
        { text: "_kernel", id: 45910 },
        { text: "(", id: 7 },
        { text: "Q", id: 52 },
        { text: ",", id: 11 },
        { text: " K", id: 46 },
        { text: ",", id: 11 },
        { text: " V", id: 57 },
        { text: "):\n   ", id: 3829 },
        { text: " return", id: 622 },
        { text: " flash", id: 14205 },
        { text: "_attn", id: 31084 },
        { text: "_func", id: 19502 },
        { text: "(Q", id: 8912 },
        { text: ", K", id: 2314 },
        { text: ", V", id: 2841 },
        { text: ", causal", id: 51204 },
        { text: "=True", id: 4120 },
        { text: ")", id: 8 }
      ]
    },
    "math": {
      name: "Reasoning Channel Prompt",
      raw: "<think> Let S_t = α_t · (S_{t-1}(I - β_t k_t k_t^T) + β_t k_t v_t^T). </think>",
      tokens: [
        { text: "<think>", id: 248002 },
        { text: " Let", id: 2849 },
        { text: " S", id: 54 },
        { text: "_t", id: 821 },
        { text: " =", id: 284 },
        { text: " α", id: 1420 },
        { text: "_t", id: 821 },
        { text: " ·", id: 1944 },
        { text: " (", id: 320 },
        { text: "S", id: 54 },
        { text: "_{t", id: 18420 },
        { text: "-1}", id: 4102 },
        { text: "(I", id: 9102 },
        { text: " -", id: 482 },
        { text: " β", id: 1682 },
        { text: "_t", id: 821 },
        { text: " k", id: 75 },
        { text: "_t", id: 821 },
        { text: " k", id: 75 },
        { text: "_t^T", id: 64201 },
        { text: ")", id: 8 },
        { text: " +", id: 491 },
        { text: " β", id: 1682 },
        { text: "_t", id: 821 },
        { text: " k", id: 75 },
        { text: "_t", id: 821 },
        { text: " v", id: 86 },
        { text: "_t^T", id: 64201 },
        { text: ").", id: 914 },
        { text: " </think>", id: 248003 }
      ]
    },
    "multilingual": {
      name: "Multilingual Alignment",
      raw: "Qwen 3.8 aligns English, 深度学习架构, गणितीय अनुकूलन, and الذكاء الاصطناعي.",
      tokens: [
        { text: "Qwen", id: 48201 },
        { text: " 3.8", id: 12902 },
        { text: " aligns", id: 42109 },
        { text: " English", id: 4120 },
        { text: ",", id: 11 },
        { text: " 深度", id: 29402 },
        { text: "学习", id: 10842 },
        { text: "架构", id: 31084 },
        { text: ",", id: 11 },
        { text: " गणित", id: 48102 },
        { text: "ीय", id: 14201 },
        { text: " अनुकूल", id: 68102 },
        { text: "न", id: 8904 },
        { text: ",", id: 11 },
        { text: " and", id: 324 },
        { text: " الذك", id: 34102 },
        { text: "اء", id: 11204 },
        { text: " الاصط", id: 54109 },
        { text: "ناعي", id: 21904 },
        { text: ".", id: 13 }
      ]
    },
    "agent": {
      name: "Agentic Tool RPC Syntax",
      raw: "<|im_start|>assistant\n<|tool_call_start|>{ \"name\": \"eval_expr\", \"args\": { \"x\": 42 } }<|tool_call_end|><|im_end|>",
      tokens: [
        { text: "<|im_start|>", id: 248004 },
        { text: "assistant\n", id: 78201 },
        { text: "<|tool_call_start|>", id: 248006 },
        { text: "{", id: 28 },
        { text: " \"name\"", id: 1042 },
        { text: ":", id: 25 },
        { text: " \"eval_expr\"", id: 49102 },
        { text: ",", id: 11 },
        { text: " \"args\"", id: 2940 },
        { text: ":", id: 25 },
        { text: " {", id: 410 },
        { text: " \"x\"", id: 1204 },
        { text: ":", id: 25 },
        { text: " 42", id: 2914 },
        { text: " }", id: 482 },
        { text: " }", id: 482 },
        { text: "<|tool_call_end|>", id: 248007 },
        { text: "<|im_end|>", id: 248005 }
      ]
    },
    "lowresource": {
      name: "Low-Resource Scripts (Thai / Khmer / Tamil)",
      raw: "โมเดลไฮบริด (Thai) · மொழியியல் (Tamil) · ភាសាខ្មែរ (Khmer)",
      tokens: [
        { text: "โม", id: 12401 },
        { text: "เดล", id: 38402 },
        { text: "ไฮ", id: 41029 },
        { text: "บริด", id: 59102 },
        { text: " (Thai)", id: 9410 },
        { text: " ·", id: 1944 },
        { text: " மொழி", id: 48102 },
        { text: "யியல்", id: 62109 },
        { text: " (Tamil)", id: 11029 },
        { text: " ·", id: 1944 },
        { text: " ភាសា", id: 58102 },
        { text: "ខ្មែរ", id: 74109 },
        { text: " (Khmer)", id: 14029 }
      ]
    }
  };

  /* low-alpha washes of the palette pigments — amber/mint/iris/sky/rose */
  const BADGE_COLORS = [
    "rgba(184,52,31,0.18)",
    "rgba(29,110,107,0.18)",
    "rgba(43,58,138,0.18)",
    "rgba(138,86,16,0.18)",
    "rgba(138,51,36,0.18)"
  ];

  function renderTokenizerLab() {
    if (!tokLiveTokens) return;
    const key = tokPresetSel ? tokPresetSel.value : "code";
    const data = PRESETS[key] || PRESETS["code"];

    const numTokens = data.tokens.length;
    const numChars = data.raw.length;
    const fertility = (numChars / numTokens).toFixed(2);
    const vramBytes = numTokens * 5120 * 2; // BF16 embeddings

    const badgesHtml = data.tokens.map((t, idx) => {
      const color = BADGE_COLORS[idx % BADGE_COLORS.length];
      const isControl = t.text.startsWith("<|") || t.text.startsWith("<think") || t.text.startsWith("</think");
      // escape before injecting — token text can contain "<", "&" etc.
      const shown = esc(t.text).replace(/\n/g, "↵").replace(/ /g, "␣");
      return `<span style="display:inline-block;background:${isControl ? "rgba(43,58,138,0.25)" : color};border:1px solid ${isControl ? "var(--iris)" : "var(--line)"};padding:2px 6px;margin:2px;border-radius:4px;font-family:var(--mono);font-size:11px;color:${isControl ? "var(--iris)" : "var(--fg)"}" title="Token ID: ${t.id}"><b>${shown}</b> <span style="color:var(--faint);font-size:9px">#${t.id}</span></span>`;
    }).join("");

    const idsPreview = data.tokens.map(t => t.id).join(", ");

    tokLiveTokens.innerHTML = `
      <div class="calc-stat-grid">
        <div class="cs-card">
          <div class="cs-num" style="color:var(--mint)">${numTokens} Tokens</div>
          <div class="cs-lbl">Emitted Tokens</div>
        </div>
        <div class="cs-card">
          <div class="cs-num">${numChars} Chars</div>
          <div class="cs-lbl">Raw Character Length</div>
        </div>
        <div class="cs-card">
          <div class="cs-num" style="color:var(--amber)">${fertility} Chars/Tok</div>
          <div class="cs-lbl">Compression Fertility</div>
        </div>
        <div class="cs-card">
          <div class="cs-num" style="color:var(--sky)">${(vramBytes / 1024).toFixed(1)} KB</div>
          <div class="cs-lbl">Embedding Activation Footprint</div>
        </div>
      </div>

      <div style="background:var(--bg-2);padding:10px 14px;border-radius:6px;box-shadow:inset 0 0 0 1px var(--line);margin-top:8px">
        <div style="font-family:var(--mono);font-size:11px;color:var(--faint);margin-bottom:6px">Segmented Token Stream (Click token for ID):</div>
        <div style="line-height:1.8">${badgesHtml}</div>
      </div>

      <div style="font-family:var(--mono);font-size:10.5px;color:var(--faint);background:var(--bg-3,var(--bg-2));padding:6px 10px;border-radius:4px;margin-top:8px;overflow-x:auto;white-space:nowrap">
        Token IDs: [ ${idsPreview} ]
      </div>
    `;
  }

  if (tokPresetSel) {
    tokPresetSel.addEventListener("change", renderTokenizerLab);
  }
  renderTokenizerLab();


  /* ------------------------------------------------------------
     2. Widget: Multilingual Fertility & Scaling Matrix (#w-fertility-matrix)
     ------------------------------------------------------------ */
  const fertStatCard = $("fertStatCard");
  const fertLangSeg = $("fertLangSeg");

  const FERTILITY_DATA = {
    "en": {
      name: "English Prose & Technical Documents",
      v32: "1.30 tokens/word",
      v151: "1.15 tokens/word",
      v248: "1.08 tokens/word",
      gain: "16.9% sequence compression vs 32k",
      desc: "English compression is mature; 248k vocabulary captures complex technical compound nouns and morphological suffixes as atomic tokens."
    },
    "code": {
      name: "Python / C++ / TypeScript Source Code",
      v32: "1.50× char expansion",
      v151: "1.20× char expansion",
      v248: "1.05× char expansion",
      gain: "30.0% sequence reduction vs 32k",
      desc: "4-space and 8-space indentation runs, snake_case identifiers, and operator chains merge into single tokens, drastically accelerating coding reasoning."
    },
    "zh": {
      name: "Mandarin Chinese (Simplified & Traditional)",
      v32: "1.45 tokens/char",
      v151: "0.72 tokens/char",
      v248: "0.68 tokens/char",
      gain: "53.1% sequence compression vs 32k",
      desc: "Expanded multi-character Chinese dictionary maps idiomatic four-character idioms (成语) and common nouns into single token embeddings."
    },
    "hi": {
      name: "Hindi & Indic Devanagari Scripts",
      v32: "2.80 tokens/word (byte-split)",
      v151: "1.55 tokens/word",
      v248: "1.22 tokens/word",
      gain: "56.4% sequence compression vs 32k",
      desc: "Completely resolves the classic 'byte-collapse' penalty where Indic conjuncts previously required 3 to 4 individual UTF-8 bytes."
    },
    "ar": {
      name: "Arabic & Perso-Arabic Scripts",
      v32: "2.40 tokens/word",
      v151: "1.40 tokens/word",
      v248: "1.15 tokens/word",
      gain: "52.1% sequence compression vs 32k",
      desc: "Native root-and-pattern morphology tokenization eliminates excessive subword fragmentation across Islamic jurisprudence and scientific literature."
    },
    "th": {
      name: "Thai & Khmer Continuous Scripts",
      v32: "3.90 tokens/word (byte-collapse)",
      v151: "1.85 tokens/word",
      v248: "1.40 tokens/word",
      gain: "64.1% sequence compression vs 32k",
      desc: "Scripts without explicit word boundaries see the largest relative throughput boost, converting unsegmented byte chains into coherent lexical tokens."
    }
  };

  function renderFertility(langKey) {
    if (!fertStatCard) return;
    const d = FERTILITY_DATA[langKey] || FERTILITY_DATA["en"];

    fertStatCard.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px;border-bottom:1px solid var(--line);padding-bottom:6px">
        <span style="font-family:var(--display);font-size:15px;font-weight:700;color:var(--fg)">${d.name}</span>
        <span style="font-family:var(--mono);font-size:11px;color:var(--mint);font-weight:600">${d.gain}</span>
      </div>

      <div class="calc-stat-grid">
        <div class="cs-card">
          <div class="cs-num" style="color:var(--rose)">${d.v32}</div>
          <div class="cs-lbl">Legacy 32k Vocabulary</div>
        </div>
        <div class="cs-card">
          <div class="cs-num">${d.v151}</div>
          <div class="cs-lbl">Qwen 2.5 (151k Vocab)</div>
        </div>
        <div class="cs-card">
          <div class="cs-num" style="color:var(--mint)">${d.v248}</div>
          <div class="cs-lbl">Qwen 3.8 (248k Padded)</div>
        </div>
      </div>

      <div style="font-size:12.5px;color:var(--muted);line-height:1.5;margin-top:8px">
        ${d.desc}
      </div>
    `;
  }

  if (fertLangSeg) {
    bindGroup(fertLangSeg, b => renderFertility(b.dataset.lang));
  }
  renderFertility("en");
})();

/* ============================================================
   Section 03: The Hybrid Backbone Engine
   ============================================================ */
(function () {
  /* ------------------------------------------------------------
     1. Widget: 64-Layer Backbone Explorer (#w-layers)
     ------------------------------------------------------------ */
  const strip = $("layerStrip");
  const info = $("layerInfo");
  const layerCtx = $("layerCtx");
  const layerDtype = $("layerDtype");

  if (strip && info) {
    strip.innerHTML = "";
    const kinds = [];
    for (let s = 0; s < 16; s++) kinds.push("delta", "delta", "delta", "full");

    kinds.forEach((kind, i) => {
      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = "layer-cell" + (kind === "full" ? " fa" : "");
      cell.title = `Layer ${i + 1}`;
      cell.setAttribute("aria-label", `Layer ${i + 1}: ${kind === "full" ? "Gated Attention" : "Gated DeltaNet"}`);
      cell.setAttribute("aria-pressed", "false");
      strip.appendChild(cell);
    });

    function showLayer(cell) {
      const i = [...strip.children].indexOf(cell);
      const kind = kinds[i];

      const ctx = layerCtx ? +layerCtx.value : 262144;
      const bp = layerDtype ? +layerDtype.value : 2;

      const isDelta = kind === "delta";
      const motifIndex = (i % 4) + 1;
      const blockIndex = Math.floor(i / 4) + 1;

      // Cumulative attention layers up to this layer
      const numAttnSoFar = kinds.slice(0, i + 1).filter(k => k === "full").length;
      const numDeltaSoFar = (i + 1) - numAttnSoFar;

      // KV cache for attention layers: gqaKV heads * gqaDh dim * 2 (K+V) * bp bytes per token
      const kvBytesPerLayer = 2 * CFG.gqaKV * CFG.gqaDh * bp * ctx;
      const cumulativeKvGb = (numAttnSoFar * kvBytesPerLayer) / 1e9;

      // Parameters so far — same per-layer expressions the W1 audit uses
      const cumParams = (numDeltaSoFar * (CFG.dnParams + CFG.ffnParams)
        + numAttnSoFar * (CFG.faParams + CFG.ffnParams)) / 1e9;

      info.innerHTML = `
          <div class="calc-stat-grid">
            <div class="cs-card">
              <div class="cs-num" style="color:${isDelta ? "var(--amber)" : "var(--mint)"}">Layer ${i + 1} / 64</div>
              <div class="cs-lbl">${isDelta ? "Gated DeltaNet (3Δ)" : "Gated Attention (1A)"}</div>
            </div>
            <div class="cs-card">
              <div class="cs-num">Block ${blockIndex} · Step ${motifIndex}/4</div>
              <div class="cs-lbl">${isDelta ? "Recurrent Context Mixer" : "Global Exact Recall Anchor"}</div>
            </div>
            <div class="cs-card">
              <div class="cs-num" style="color:var(--mint)">${cumulativeKvGb.toFixed(2)} GB</div>
              <div class="cs-lbl">Cumulative KV Cache @ ${ctx.toLocaleString()} tok</div>
            </div>
            <div class="cs-card">
              <div class="cs-num" style="color:var(--iris)">${cumParams.toFixed(2)}B</div>
              <div class="cs-lbl">Cumulative Params (L1..L${i+1})</div>
            </div>
          </div>
          <div style="font-family:var(--mono);font-size:11px;color:var(--muted);line-height:1.5;border-top:1px solid var(--line);padding-top:8px">
            ${isDelta
              ? `<b>Linear Attention:</b> 16 QK / 48 V heads × 128 dim · Fixed state $128 \\times 128$ per head (0 Bytes dynamic KV cache). Constant $\\mathcal{O}(1)$ decode FLOPs.`
              : `<b>Full Softmax Attention:</b> 24 Q / 4 KV heads × 256 dim · GQA 6:1 · 3D MRoPE $[11, 11, 10]$ · $\\sigma$-output gate. Adds ${(kvBytesPerLayer/1e9).toFixed(2)} GB to cache.`}
          </div>
        `;
      katexRender(info);                                          // footnote carries inline math
    }

    // exclusive selection via bindGroup; .sel mirrors the legacy CSS hook
    bindGroup(strip, cell => {
      strip.querySelectorAll(".layer-cell").forEach(c => c.classList.remove("sel"));
      cell.classList.add("sel");
      showLayer(cell);
    }, ".layer-cell");

    // roving-focus keyboard nav (←/→ move & select, Home/End jump)
    strip.addEventListener("keydown", e => {
      const cells = [...strip.querySelectorAll(".layer-cell")];
      const cur = cells.indexOf(document.activeElement);
      let next = -1;
      if (e.key === "ArrowRight" || e.key === "ArrowDown") next = Math.min(cells.length - 1, cur + 1);
      else if (e.key === "ArrowLeft" || e.key === "ArrowUp") next = Math.max(0, cur - 1);
      else if (e.key === "Home") next = 0;
      else if (e.key === "End") next = cells.length - 1;
      if (next >= 0 && cur >= 0) { e.preventDefault(); cells[next].focus(); cells[next].click(); }
    });

    [layerCtx, layerDtype].forEach(id => {
      if (id) id.addEventListener("change", () => strip.querySelector(".layer-cell.sel")?.click());
    });

    strip.children[3]?.click();
  }


  /* ------------------------------------------------------------
     2. Widget: 4-Layer Super-Block Motif Visualizer (#w-hybrid-motif)
     ------------------------------------------------------------ */
  const motifStepCard = $("motifStepCard");
  const motifStepSeg = $("motifStepSeg");

  const MOTIF_STEPS = {
    "1": {
      name: "Superblock Step 1 · DeltaNet Recurrent Compression",
      type: "Linear Attention Mixer + SwiGLU FFN",
      state: "S_t ← α·(S_{t-1}(I - β k k^T) + β k v^T)",
      desc: "First recurrent update of the 4-layer cycle. Ingests raw input tokens, applies causal 1D depthwise conv (kernel 4) for local token order, and writes associative key-value updates into fixed 128×128 memory matrices.",
      benefit: "Pure $\\mathcal{O}(1)$ streaming state — 0 bytes KV cache."
    },
    "2": {
      name: "Superblock Step 2 · DeltaNet Recurrent Refinement",
      type: "Linear Attention Mixer + SwiGLU FFN",
      state: "S_t ← α·(S_{t-1}(I - β k k^T) + β k v^T)",
      desc: "Second recurrent pass. Multi-layer depth allows DeltaNet to compose associative memories, performing multi-step reasoning and semantic feature extraction on the compressed prefix state.",
      benefit: "Filters transient syntactic noise via learned decay gate $\\alpha_t$."
    },
    "3": {
      name: "Superblock Step 3 · DeltaNet Recurrent Fusion",
      type: "Linear Attention Mixer + SwiGLU FFN",
      state: "S_t ← α·(S_{t-1}(I - β k k^T) + β k v^T)",
      desc: "Third recurrent pass. Fuses long-horizon context before encountering the full-attention anchor. Maximizes the amount of sequence history compressed per FLOP.",
      benefit: "Highest FLOP efficiency across the 4-layer super-block."
    },
    "4": {
      name: "Superblock Step 4 · Gated Attention Retrieval Anchor",
      type: "Grouped-Query Attention (GQA 6:1) + SwiGLU FFN",
      state: "Attn(Q, K, V) = Softmax((Q K^T / √d_k) + M) V",
      desc: "The global synchronization anchor. Performs full causal softmax attention across the entire historical token sequence, re-reading raw tokens to correct any associative drift in the preceding 3 DeltaNet layers.",
      benefit: "Guarantees 100% needle-in-a-haystack retrieval accuracy."
    }
  };

  function renderMotifStep(stepNum) {
    if (!motifStepCard) return;
    const info = MOTIF_STEPS[stepNum] || MOTIF_STEPS["1"];

    motifStepCard.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px;border-bottom:1px solid var(--line);padding-bottom:6px">
        <span style="font-family:var(--display);font-size:15px;font-weight:700;color:var(--fg)">${info.name}</span>
        <span style="font-family:var(--mono);font-size:11px;color:${stepNum === "4" ? "var(--mint)" : "var(--amber)"};font-weight:600">${info.type}</span>
      </div>
      <div style="background:var(--bg-2);padding:8px 12px;border-radius:6px;box-shadow:inset 0 0 0 1px var(--line);font-family:var(--mono);font-size:11.5px;color:var(--amber);margin-bottom:8px">
        ${info.state}
      </div>
      <div style="font-size:12.5px;color:var(--muted);line-height:1.5;margin-bottom:6px">
        ${info.desc}
      </div>
      <div style="font-size:11.5px;color:var(--faint);font-family:var(--mono)">
        <b>Motif Advantage:</b> ${info.benefit}
      </div>
    `;
    katexRender(motifStepCard);
  }

  if (motifStepSeg) {
    bindGroup(motifStepSeg, b => renderMotifStep(b.dataset.step));
  }
  renderMotifStep("1");
})();


/* ============================================================
   Section 04: Gated DeltaNet & Associative Memory Engine
   ============================================================ */
(function () {
  /* ------------------------------------------------------------
     1. Widget: Interactive Delta-Rule Associative Memory Lab (#w-delta)
     ------------------------------------------------------------ */
  const deltaLabReadout = $("deltaLabReadout");
  const deltaKeySel = $("deltaKeySel");
  const deltaValInput = $("deltaValInput");
  const deltaModeSel = $("deltaModeSel");
  const btnDeltaWrite = $("btnDeltaWrite");
  const btnDeltaDecay = $("btnDeltaDecay");
  const btnDeltaReset = $("btnDeltaReset");
  const btnScenarioOverwrite = $("btnScenarioOverwrite");
  const btnScenarioMulti = $("btnScenarioMulti");

  const VALUE_PRESETS = {
    "alpha": { name: "Alpha", vec: [1.0, 0.6, 0.2, -0.4], label: "[+1.0, +0.6, +0.2, -0.4]" },
    "beta":  { name: "Beta",  vec: [-0.8, 1.0, 0.5, 0.0], label: "[-0.8, +1.0, +0.5, 0.0]" },
    "gamma": { name: "Gamma", vec: [0.0, -0.7, 1.0, 0.8], label: "[0.0, -0.7, +1.0, +0.8]" },
    "delta": { name: "Delta", vec: [0.5, 0.8, -0.9, 0.3], label: "[+0.5, +0.8, -0.9, +0.3]" },
    "zero":  { name: "Zero",  vec: [0.0, 0.0, 0.0, 0.0], label: "[0.0, 0.0, 0.0, 0.0]" }
  };

  const SLOT_CONFIG = [
    { label: "Slot 0 · User Intent", desc: "Represents conversational goal vector" },
    { label: "Slot 1 · Document Context", desc: "Long-horizon document state" },
    { label: "Slot 2 · Tool Execution", desc: "RPC state / environment response" },
    { label: "Slot 3 · Working Variable", desc: "Active reasoning entity" }
  ];

  // 4x4 Associative Matrix (4 key slots x 4 value feature dims)
  let S = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
  ];
  let slotHistory = [[], [], [], []]; // Tracks writes per slot
  let expectedSlots = [null, null, null, null];
  let writeHistory = [];

  function getCellBg(val) {
    if (Math.abs(val) < 0.01) return "transparent";
    if (val > 0) {
      const alpha = Math.min(0.35, Math.abs(val) * 0.28 + 0.08);
      return `rgba(29, 110, 107, ${alpha})`;
    } else {
      const alpha = Math.min(0.35, Math.abs(val) * 0.28 + 0.08);
      return `rgba(184, 52, 31, ${alpha})`;
    }
  }

  function renderDeltaLab() {
    if (!deltaLabReadout) return;

    const activeSlot = deltaKeySel ? +deltaKeySel.value : 0;
    const mode = deltaModeSel ? deltaModeSel.value : "deltanet";

    // Frobenius Norm: ||S||_F = sqrt(sum S_ij^2)
    let fNorm = 0;
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        fNorm += S[r][c] * S[r][c];
      }
    }
    fNorm = Math.sqrt(fNorm);

    // Live Query for selected slot: y = S[activeSlot]
    const retrievedVec = S[activeSlot];
    const targetInfo = expectedSlots[activeSlot];
    let recallError = 0;
    let hasTarget = !!targetInfo;
    if (hasTarget) {
      for (let c = 0; c < 4; c++) {
        const diff = retrievedVec[c] - targetInfo.vec[c];
        recallError += diff * diff;
      }
      recallError = Math.sqrt(recallError);
    }

    // Build 4x4 matrix table rows
    const matrixRowsHtml = S.map((row, rIdx) => {
      const isActive = rIdx === activeSlot;
      const slotName = SLOT_CONFIG[rIdx].label.split("·")[1].trim();
      const cells = row.map(v => {
        const color = v > 0.01 ? "var(--mint)" : (v < -0.01 ? "var(--rose)" : "var(--faint)");
        return `<td><span class="delta-cell-val" style="background:${getCellBg(v)};color:${color}">${v >= 0 ? "+" : ""}${v.toFixed(2)}</span></td>`;
      }).join("");

      return `
        <tr class="${isActive ? "active-row" : ""}" data-slot="${rIdx}" tabindex="0" role="button" aria-label="Select Slot ${rIdx}">
          <td style="font-weight:600;color:${isActive ? "var(--amber)" : "var(--fg)"}">
            <b>k_${rIdx}</b> <span style="font-size:10px;color:var(--faint)">(${slotName})</span>
          </td>
          ${cells}
        </tr>
      `;
    }).join("");

    // Build vector readout string for query probe
    const retrievedFormatted = retrievedVec.map(v => `${v >= 0 ? "+" : ""}${v.toFixed(2)}`).join(", ");
    const targetFormatted = hasTarget ? targetInfo.vec.map(v => `${v >= 0 ? "+" : ""}${v.toFixed(2)}`).join(", ") : "None written";

    deltaLabReadout.innerHTML = `
      <div class="calc-stat-grid">
        <div class="cs-card">
          <div class="cs-num" style="color:var(--amber)">${fNorm.toFixed(2)}</div>
          <div class="cs-lbl">Frobenius Norm ||S||_F</div>
        </div>
        <div class="cs-card">
          <div class="cs-num" style="color:${mode === "deltanet" ? "var(--mint)" : "var(--rose)"}">
            ${mode === "deltanet" ? "Delta Rule" : "Additive"}
          </div>
          <div class="cs-lbl">Active Update Regime</div>
        </div>
        <div class="cs-card">
          <div class="cs-num" style="color:${hasTarget ? (recallError < 0.05 ? "var(--mint)" : "var(--rose)") : "var(--muted)"}">
            ${hasTarget ? (recallError < 0.05 ? "Exact (0.00)" : `+${recallError.toFixed(2)} Err`) : "—"}
          </div>
          <div class="cs-lbl">k_${activeSlot} Recall Precision</div>
        </div>
        <div class="cs-card">
          <div class="cs-num" style="color:var(--iris)">${writeHistory.length} Ops</div>
          <div class="cs-lbl">Operation Counter</div>
        </div>
      </div>

      <div class="delta-matrix-wrap">
        <div class="delta-matrix-box">
          <div class="delta-box-title">
            <span>Associative Matrix S_t ∈ ℝ^(4×4)</span>
            <span style="font-size:10px;color:var(--faint)">k_i × v_j (48 heads in §04)</span>
          </div>
          <table class="delta-grid-table">
            <thead>
              <tr>
                <th scope="col">Slot Key (k)</th>
                <th scope="col">v[0]</th>
                <th scope="col">v[1]</th>
                <th scope="col">v[2]</th>
                <th scope="col">v[3]</th>
              </tr>
            </thead>
            <tbody>
              ${matrixRowsHtml}
            </tbody>
          </table>
        </div>

        <div class="delta-probe-box">
          <div class="delta-box-title">
            <span>Query Probe: y_t = S_t^T q_t</span>
            <span style="font-size:10px;color:var(--amber)">Probing k_${activeSlot}</span>
          </div>
          <div style="font-size:11px;color:var(--muted);line-height:1.6">
            <div style="margin-bottom:4px"><b>Retrieved Signal y:</b> <code>[ ${retrievedFormatted} ]</code></div>
            <div style="margin-bottom:6px"><b>Target Expected:</b> <code>[ ${targetFormatted} ]</code></div>
            
            <div style="padding:6px 8px;border-radius:4px;background:var(--bg-1);margin-top:6px">
              ${!hasTarget 
                ? '<span style="color:var(--faint)">Slot is empty. Write a payload to observe associative retention.</span>'
                : (recallError < 0.05 
                    ? '<span style="color:var(--mint);font-weight:600">✓ Clean Memory Retrieval:</span> Delta-rule subtraction <code>(I - β k k^T)</code> cleanly eliminated all previous state without ghost interference.'
                    : `<span style="color:var(--rose);font-weight:600">⚠️ Superposition Ghosting (+${recallError.toFixed(2)}):</span> Additive linear attention retained past writes simultaneously, causing unrecoverable feature cross-talk.`)}
            </div>
          </div>
        </div>
      </div>

      <div style="font-family:var(--mono);font-size:11px;color:var(--muted);line-height:1.5;border-top:1px solid var(--line);padding-top:8px;margin-top:10px">
        ${writeHistory.length === 0 
          ? "State matrix is zeroed. Choose an update rule, target slot, and feature payload above, or click a <b>Guided Scenario</b>." 
          : `<b>Last Operation:</b> ${writeHistory[writeHistory.length - 1]}`}
      </div>
    `;
    katexRender(deltaLabReadout);

    // Bind row click & keydown to change active slot
    const rows = deltaLabReadout.querySelectorAll(".delta-grid-table tr[data-slot]");
    rows.forEach(r => {
      const selectRow = () => {
        const slot = r.getAttribute("data-slot");
        if (deltaKeySel && slot) {
          deltaKeySel.value = slot;
          renderDeltaLab();
        }
      };
      r.addEventListener("click", selectRow);
      r.addEventListener("keydown", e => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          selectRow();
        }
      });
    });
  }

  function writeDelta(slotIdx, valKey, mode) {
    const payload = VALUE_PRESETS[valKey] || VALUE_PRESETS["alpha"];
    const v = payload.vec;
    const beta = 1.0;

    slotHistory[slotIdx].push({ valKey, vec: [...v], mode });
    expectedSlots[slotIdx] = { valKey, name: payload.name, vec: [...v] };

    if (mode === "deltanet") {
      // S_t = S_{t-1}(I - beta k k^T) + beta k v^T
      for (let c = 0; c < 4; c++) {
        S[slotIdx][c] = S[slotIdx][c] * (1.0 - beta) + beta * v[c];
      }
      writeHistory.push(`DeltaNet: Wrote <b>${payload.name}</b> to Slot ${slotIdx} · Stale state erased via <code>(I - β k k^T)</code>`);
    } else {
      // Pure Linear Attention: S_t = S_{t-1} + k v^T
      for (let c = 0; c < 4; c++) {
        S[slotIdx][c] += v[c];
      }
      writeHistory.push(`Additive Linear: Added <b>${payload.name}</b> into Slot ${slotIdx} · Superposition accumulation`);
    }
    renderDeltaLab();
  }

  function decayDelta() {
    const alpha = 0.85;
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        S[r][c] = +(S[r][c] * alpha).toFixed(4);
      }
      if (expectedSlots[r]) {
        expectedSlots[r].vec = expectedSlots[r].vec.map(x => +(x * alpha).toFixed(4));
      }
    }
    writeHistory.push(`Global Decay Applied: <code>S ← 0.85 · S</code> (Data-dependent retention forgetting)`);
    renderDeltaLab();
  }

  function resetDelta() {
    S = [
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0]
    ];
    slotHistory = [[], [], [], []];
    expectedSlots = [null, null, null, null];
    writeHistory = [];
    renderDeltaLab();
  }

  if (btnDeltaWrite) {
    btnDeltaWrite.addEventListener("click", () => {
      const slot = deltaKeySel ? +deltaKeySel.value : 0;
      const valKey = deltaValInput ? deltaValInput.value : "alpha";
      const mode = deltaModeSel ? deltaModeSel.value : "deltanet";
      writeDelta(slot, valKey, mode);
    });
  }

  if (btnDeltaDecay) btnDeltaDecay.addEventListener("click", decayDelta);
  if (btnDeltaReset) btnDeltaReset.addEventListener("click", resetDelta);

  if (deltaKeySel) deltaKeySel.addEventListener("change", renderDeltaLab);
  if (deltaModeSel) deltaModeSel.addEventListener("change", renderDeltaLab);

  if (btnScenarioOverwrite) {
    btnScenarioOverwrite.addEventListener("click", () => {
      resetDelta();
      const mode = deltaModeSel ? deltaModeSel.value : "deltanet";
      if (deltaKeySel) deltaKeySel.value = "0";
      // Step 1: Write Alpha
      writeDelta(0, "alpha", mode);
      // Step 2: Overwrite with Beta
      writeDelta(0, "beta", mode);
    });
  }

  if (btnScenarioMulti) {
    btnScenarioMulti.addEventListener("click", () => {
      resetDelta();
      const mode = deltaModeSel ? deltaModeSel.value : "deltanet";
      writeDelta(0, "alpha", mode);
      writeDelta(1, "beta", mode);
      writeDelta(2, "gamma", mode);
      writeDelta(3, "delta", mode);
      if (deltaKeySel) deltaKeySel.value = "0";
      renderDeltaLab();
    });
  }

  renderDeltaLab();


  /* ------------------------------------------------------------
     2. Widget: Chunked WY Parallel Scan Visualizer (#w-deltanet-scan)
     ------------------------------------------------------------ */
  const wyScanVisual = $("wyScanVisual");
  const wyChunkSeg = $("wyChunkSeg");
  let curChunkSize = 64;

  function renderWyScan() {
    if (!wyScanVisual) return;

    const totalTokens = 262144;
    const numChunks = totalTokens / curChunkSize;
    const flopsPerChunk = (curChunkSize * curChunkSize * 128 * 2) / 1e6; // MFLOPs

    wyScanVisual.innerHTML = `
      <div class="calc-stat-grid">
        <div class="cs-card">
          <div class="cs-num" style="color:var(--mint)">${numChunks.toLocaleString()} Chunks</div>
          <div class="cs-lbl">Sequential Inter-Chunk Steps</div>
        </div>
        <div class="cs-card">
          <div class="cs-num">${curChunkSize} Tokens</div>
          <div class="cs-lbl">Parallel Intra-Chunk Block</div>
        </div>
        <div class="cs-card">
          <div class="cs-num" style="color:var(--amber)">${flopsPerChunk.toFixed(2)} MFLOPs</div>
          <div class="cs-lbl">Intra-Chunk GEMM Compute</div>
        </div>
        <div class="cs-card">
          <div class="cs-num" style="color:var(--iris)">100% Tensor Core</div>
          <div class="cs-lbl">Hardware Saturation</div>
        </div>
      </div>
      <div style="font-family:var(--mono);font-size:11.5px;color:var(--muted);line-height:1.5;background:rgba(29,110,107,0.06);padding:8px 12px;border-radius:6px;border-left:3px solid var(--mint);margin-top:8px">
        ✓ <b>Chunked WY Representation:</b> The sequential product $\\prod_{t=1}^C (I - \\beta_t k_t k_t^T)$ factors into a block-triangular representation $(I - W Y^T)^{-1}$, transforming $\\mathcal{O}(T)$ recurrence into dense parallel GEMMs across $C=${curChunkSize}$ tokens simultaneously.
      </div>
    `;
    katexRender(wyScanVisual);
  }

  if (wyChunkSeg) {
    bindGroup(wyChunkSeg, b => { curChunkSize = +b.dataset.chunk; renderWyScan(); });
  }
  renderWyScan();
})();

/* ============================================================
   Section 05: Gated Attention & 3D MRoPE Engine
   ============================================================ */
(function () {
  /* ------------------------------------------------------------
     1. Widget: GQA KV-Cache Memory & Bandwidth Calculator (#w-cache)
     ------------------------------------------------------------ */
  const cacheReadout = $("cacheReadout");
  const cacheCtx = $("cacheCtx");
  const cacheBatch = $("cacheBatch");
  const cacheDtype = $("cacheDtype");

  function renderCacheCalc() {
    if (!cacheReadout) return;
    const T = cacheCtx ? +cacheCtx.value : 262144;
    const B = cacheBatch ? +cacheBatch.value : 1;
    const bp = cacheDtype ? +cacheDtype.value : 2;

    // Attention-layer geometry straight from CFG — stays in sync with the W1 audit
    const nLayersAttn = CFG.nFull;
    const nKv = CFG.gqaKV;
    const dH = CFG.gqaDh;

    // Bytes per token per layer = 2 * n_kv * d_h * bp (K and V); decimal GB
    const bytesPerTokLayer = 2 * nKv * dH * bp;
    const totalGqaBytes = bytesPerTokLayer * nLayersAttn * T * B;
    const totalGqaGb = totalGqaBytes / 1e9;

    // Full MHA baseline (all 64 layers at gqaQ KV heads)
    const bytesPerTokMha = 2 * CFG.gqaQ * dH * bp;
    const totalMhaBytes = bytesPerTokMha * CFG.layers * T * B;
    const totalMhaGb = totalMhaBytes / 1e9;

    const savingsFactor = totalGqaGb > 0 ? (totalMhaGb / totalGqaGb).toFixed(1) : "—";

    cacheReadout.innerHTML = `
      <div class="calc-stat-grid">
        <div class="cs-card">
          <div class="cs-num" style="color:var(--mint)">${totalGqaGb.toFixed(2)} GB</div>
          <div class="cs-lbl">GQA-4 Cache (16 Layers)</div>
        </div>
        <div class="cs-card">
          <div class="cs-num">${bytesPerTokLayer.toLocaleString()} B</div>
          <div class="cs-lbl">Bytes / Token / Layer</div>
        </div>
        <div class="cs-card">
          <div class="cs-num" style="color:var(--rose)">${totalMhaGb.toFixed(1)} GB</div>
          <div class="cs-lbl">Standard MHA (64L · 24H)</div>
        </div>
        <div class="cs-card">
          <div class="cs-num" style="color:var(--amber)">${savingsFactor}× Less</div>
          <div class="cs-lbl">VRAM Compression</div>
        </div>
      </div>
      <div style="font-family:var(--mono);font-size:11px;color:var(--muted);line-height:1.5;border-top:1px solid var(--line);padding-top:8px">
        Sequence: <b>${T.toLocaleString()} tokens</b> × Batch <b>${B}</b> @ <b>${bp === 2 ? "BF16" : (bp === 1 ? "FP8" : "FP4")}</b> · 48 DeltaNet layers require <b>0 GB</b> dynamic KV-cache (fixed recurrent matrix).
      </div>
    `;
  }

  [cacheCtx, cacheBatch, cacheDtype].forEach(el => {
    if (el) {
      el.addEventListener("change", renderCacheCalc);
      el.addEventListener("input", renderCacheCalc);
    }
  });
  renderCacheCalc();


  /* ------------------------------------------------------------
     2. Widget: 3D MRoPE Phase Visualizer (#w-rope)
     ------------------------------------------------------------ */
  const svg = $("ropeSvg");
  const ropeOut = $("ropeOut");
  const ropePos = $("ropePos");
  const ropeDim = $("ropeDim");
  const mropeAxis = $("mropeAxis");

  if (svg && ropeOut) {
    const NS = "http://www.w3.org/2000/svg";
    const cx = 160, cy = 160, R = 108;

    function el(t, attrs, parent) {
      const e = document.createElementNS(NS, t);
      for (const k in attrs) e.setAttribute(k, attrs[k]);
      (parent || svg).appendChild(e);
      return e;
    }

    // Dial background — ink strokes on parchment (PAL mirrors the :root tokens)
    el("circle", { cx, cy, r: R + 18, fill: "rgba(184,52,31,.04)" });
    el("circle", { cx, cy, r: R, fill: "none", stroke: "rgba(26,22,18,.14)", "stroke-width": 1.2 });
    el("circle", { cx, cy, r: R * .66, fill: "none", stroke: "rgba(26,22,18,.06)" });
    el("circle", { cx, cy, r: R * .33, fill: "none", stroke: "rgba(26,22,18,.06)" });
    el("line", { x1: cx - R - 14, y1: cy, x2: cx + R + 14, y2: cy, stroke: "rgba(26,22,18,.08)" });
    el("line", { x1: cx, y1: cy - R - 14, x2: cx, y2: cy + R + 14, stroke: "rgba(26,22,18,.08)" });

    for (let k = 0; k < 12; k++) {
      const a = k * Math.PI / 6;
      const x1 = cx + (R - 2) * Math.sin(a), y1 = cy - (R - 2) * Math.cos(a);
      const x2 = cx + (R + 4) * Math.sin(a), y2 = cy - (R + 4) * Math.cos(a);
      el("line", { x1, y1, x2, y2, stroke: "rgba(26,22,18,.18)" });
    }

    const arcTrail = el("path", { fill: "none", stroke: "rgba(184,52,31,.35)", "stroke-width": 3, "stroke-linecap": "round" });
    const hand = el("line", { x1: cx, y1: cy, x2: cx, y2: cy - R, stroke: PAL.amber, "stroke-width": 2.5, "stroke-linecap": "round" });
    const tip = el("circle", { r: 6, fill: PAL.amber, stroke: "rgba(26,22,18,.35)", "stroke-width": 1.2 });

    function theta(j) {
      return Math.pow(CFG.ropeTheta, -2 * j / CFG.rotDims);
    }

    function renderRope() {
      const m = ropePos ? +ropePos.value : 64;
      const j = ropeDim ? +ropeDim.value : 0;
      const axis = mropeAxis ? mropeAxis.value : "time";

      const ang = m * theta(j);
      const hx = cx + R * Math.sin(ang), hy = cy - R * Math.cos(ang);
      hand.setAttribute("x2", hx); hand.setAttribute("y2", hy);
      tip.setAttribute("cx", hx); tip.setAttribute("cy", hy);

      const twoPi = 2 * Math.PI;
      const norm = ((ang % twoPi) + twoPi) % twoPi;
      const large = norm > Math.PI ? 1 : 0;
      arcTrail.setAttribute("d", `M ${cx} ${cy - R} A ${R} ${R} 0 ${large} 1 ${hx} ${hy}`);

      const turns = ang / twoPi;
      const wl = twoPi / theta(j);
      const wlTxt = wl >= 1e6 ? (wl / 1e6).toFixed(2) + "M" : (wl >= 1e3 ? (wl / 1e3).toFixed(1) + "k" : wl.toFixed(0));

      const axisLabel = axis === "time" ? "Temporal Axis (T)" : (axis === "height" ? "Vertical Axis (H)" : "Horizontal Axis (W)");

      ropeOut.innerHTML = `
        <div class="rf-stat-box">
          <div class="rf-head">Rotary Phase @ Dim-Pair j = ${j}</div>
          <div class="rf-row"><span>Target Axis</span><span class="val amber">${axisLabel}</span></div>
          <div class="rf-row"><span>Phase Rotation Angle</span><span class="val mint">${norm.toFixed(2)} rad (${turns.toFixed(1)} turns)</span></div>
          <div class="rf-row"><span>Spatial Wavelength</span><span class="val">${wlTxt} tokens / cycle</span></div>
          <div class="rf-row"><span>Angular Frequency θ_j</span><span class="val" style="color:var(--iris)">${theta(j).toExponential(2)} rad/tok</span></div>
        </div>
        <div style="font-size:12px;color:var(--muted);line-height:1.45;padding:4px 2px">
          Dim-pair j=${j} provides ${j < 8 ? "high-frequency local syntactic discrimination" : "long-range global document phase grounding"}. 3D MRoPE interleaves $[11, 11, 10]$ pairs across 3D axes.
        </div>
      `;
      katexRender(ropeOut);
      if (ropePos?.type === "range") ropePos.setAttribute("aria-valuetext", `${ropePos.value} tokens position`);
      if (ropeDim?.type === "range") ropeDim.setAttribute("aria-valuetext", `dim-pair ${ropeDim.value}`);
    }

    [ropePos, ropeDim, mropeAxis].forEach(el => {
      if (el) {
        el.addEventListener("input", renderRope);
        el.addEventListener("change", renderRope);
      }
    });
    renderRope();
  }


  /* ------------------------------------------------------------
     3. Widget: Attention Output Sigmoid Gate Simulator (#w-attn-gate)
     ------------------------------------------------------------ */
  const attnGateVisual = $("attnGateVisual");
  const attnGateModeSeg = $("attnGateModeSeg");
  let attnGateMode = "active";

  function renderAttnGate() {
    if (!attnGateVisual) return;

    if (attnGateMode === "active") {
      attnGateVisual.innerHTML = `
        <div class="calc-stat-grid">
          <div class="cs-card">
            <div class="cs-num" style="color:var(--mint)">σ = 0.94 (Open)</div>
            <div class="cs-lbl">Output Gate Value</div>
          </div>
          <div class="cs-card">
            <div class="cs-num">100%</div>
            <div class="cs-lbl">Signal Transmission</div>
          </div>
          <div class="cs-card">
            <div class="cs-num" style="color:var(--amber)">High Attention</div>
            <div class="cs-lbl">Retrieval Confidence</div>
          </div>
        </div>
        <div style="font-family:var(--mono);font-size:11.5px;color:var(--mint);line-height:1.5;background:rgba(29,110,107,0.06);padding:8px 12px;border-radius:6px;border-left:3px solid var(--mint);margin-top:8px">
          ✓ <b>Relevant Context Ingested:</b> The output sigmoid gate $\sigma(x W_g) \approx 0.94$ passes full attention values into $W_{\text{out}}$, delivering sharp factual retrieval into the residual stream.
        </div>
      `;
    } else {
      attnGateVisual.innerHTML = `
        <div class="calc-stat-grid">
          <div class="cs-card">
            <div class="cs-num" style="color:var(--rose)">σ = 0.08 (Clamped)</div>
            <div class="cs-lbl">Output Gate Value</div>
          </div>
          <div class="cs-card">
            <div class="cs-num">8.0%</div>
            <div class="cs-lbl">Signal Transmission</div>
          </div>
          <div class="cs-card">
            <div class="cs-num" style="color:var(--faint)">Noise Filtered</div>
            <div class="cs-lbl">Distractor Suppression</div>
          </div>
        </div>
        <div style="font-family:var(--mono);font-size:11.5px;color:var(--rose);line-height:1.5;background:rgba(138,51,36,0.06);padding:8px 12px;border-radius:6px;border-left:3px solid var(--rose);margin-top:8px">
          ✗ <b>Distractor Context Suppressed:</b> When retrieved context contains stale history, the learned gate closes ($\sigma \approx 0.08$), preventing noisy attention logits from polluting downstream DeltaNet states.
        </div>
      `;
    }
    katexRender(attnGateVisual);
  }

  if (attnGateModeSeg) {
    bindGroup(attnGateModeSeg, b => { attnGateMode = b.dataset.gate; renderAttnGate(); });
  }
  renderAttnGate();
})();


/* ============================================================
   Section 06: FFN & Residual Stream Engine
   ============================================================ */
(function () {
  /* ------------------------------------------------------------
     1. Widget: SwiGLU Activation Explorer (#w-ffn)
     ------------------------------------------------------------ */
  const svg = $("ffnSvg");
  if (svg) {
    const NS = "http://www.w3.org/2000/svg", W = 720, H = 300, ox = 56, oy = 250, pw = 620, ph = 200;
    const X = x => ox + (x + 6) / 12 * pw, Y = y => oy - y / 24 * ph;

    function line(x1, y1, x2, y2, st) {
      const l = document.createElementNS(NS, "line");
      Object.entries({ x1, y1, x2, y2, ...st }).forEach(([k, v]) => l.setAttribute(k, v));
      svg.appendChild(l);
      return l;
    }
    function path(d, st) {
      const p = document.createElementNS(NS, "path");
      Object.entries({ d, fill: "none", ...st }).forEach(([k, v]) => p.setAttribute(k, v));
      svg.appendChild(p);
      return p;
    }
    function txt(x, y, s, fill, anchor, size) {
      const t = document.createElementNS(NS, "text");
      Object.entries({ x, y, fill: fill || PAL.muted, "text-anchor": anchor || "start", "font-size": size || 11, "font-family": '"IBM Plex Mono", ui-monospace, monospace' }).forEach(([k, v]) => t.setAttribute(k, v));
      t.textContent = s;
      svg.appendChild(t);
      return t;
    }

    for (let v = 0; v <= 20; v += 4) {
      const y = Y(v);
      line(ox, y, ox + pw, y, { stroke: "rgba(26,22,18,.05)" });
      txt(ox - 8, y + 3, String(v), PAL.muted, "end", 9);
    }
    for (let x = -6; x <= 6; x += 2) {
      const px = X(x);
      line(px, oy, px, oy + 4, { stroke: "rgba(26,22,18,.18)" });
      txt(px, oy + 16, String(x), PAL.muted, "middle", 9);
    }
    line(ox, oy, ox + pw, oy, { stroke: "rgba(26,22,18,.18)" });
    line(X(0), Y(0), X(0), Y(22), { stroke: "rgba(26,22,18,.18)" });

    const relu = x => Math.max(0, x);
    const gelu = x => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
    const silu = x => x / (1 + Math.exp(-x));

    function plot(fn, color, dash) {
      let d = "";
      for (let x = -6; x <= 6.001; x += 0.15) d += (x === -6 ? "M" : "L") + X(x) + "," + Y(fn(x)) + " ";
      path(d, { stroke: color, "stroke-width": 2, "stroke-dasharray": dash || "none" });
    }
    plot(relu, PAL.muted, "3 3");
    plot(gelu, PAL.mint);
    plot(silu, PAL.amber);

    const unitPath = path("", { stroke: PAL.iris, "stroke-width": 2.5 });
    const marker = (() => {
      const c = document.createElementNS(NS, "circle");
      Object.entries({ r: 6, fill: PAL.iris, stroke: PAL.paper, "stroke-width": 1.5, cx: -99, cy: -99 }).forEach(([k, v]) => c.setAttribute(k, v));
      svg.appendChild(c);
      return c;
    })();

    // swatch and label share the same pigment so they never disagree
    const legend = document.createElement("div");
    legend.className = "ffn-legend";
    legend.innerHTML = `
      <span class="lg-item" style="color:${PAL.muted}"><span class="lg-sw" style="background:transparent;border-top:1.5px dashed ${PAL.muted}"></span>ReLU</span>
      <span class="lg-item" style="color:${PAL.mint}"><span class="lg-sw" style="background:${PAL.mint}"></span>GELU</span>
      <span class="lg-item" style="color:${PAL.amber}"><span class="lg-sw" style="background:${PAL.amber}"></span>SiLU</span>
      <span class="lg-item" style="color:${PAL.iris}"><span class="lg-sw" style="background:${PAL.iris}"></span>SwiGLU Unit (SiLU(x)·x)</span>
    `;
    svg.parentNode.insertBefore(legend, svg.nextSibling);

    function update(x) {
      x = Math.max(-6, Math.min(6, x));
      let d = "";
      for (let xx = -6; xx <= 6.001; xx += 0.15) d += (xx === -6 ? "M" : "L") + X(xx) + "," + Y(silu(xx) * xx) + " ";
      unitPath.setAttribute("d", d);
      marker.setAttribute("cx", X(x)); marker.setAttribute("cy", Y(silu(x) * x));
      if ($("ffnOut")) {
        $("ffnOut").innerHTML = `
          <div class="row"><span>x = ${x.toFixed(2)} · SiLU(x) = ${silu(x).toFixed(2)} · SwiGLU Gated Output = <b style="color:var(--iris)">${(silu(x) * x).toFixed(2)}</b></span><span class="v"></span></div>
          <div class="row"><span>ReLU passes ${relu(x) > 0 ? "linear signal" : "zero"} · GELU soft-gates · SiLU preserves smooth negative lobe for gradient stability</span><span class="v"></span></div>
          <div style="color:var(--faint);font-size:11px;margin-top:4px">Multiplicative gating $(x W_{\\text{gate}}) \\odot \\text{SiLU}(x W_{\\text{up}})$ across 17,408 hidden units acts as key-value memory retrieval for world knowledge.</div>
        `;
        katexRender($("ffnOut"));
      }
      const ffnX = $("ffnX");
      if (ffnX) {
        ffnX.value = x;
        if (ffnX.type === "range") ffnX.setAttribute("aria-valuetext", `x = ${x.toFixed(2)}`);
      }
    }

    function updateFromPointer(e) {
      const r = svg.getBoundingClientRect();
      update(((e.clientX - r.left) / r.width * W - ox) / pw * 12 - 6);
    }
    svg.addEventListener("pointermove", updateFromPointer);
    if ($("ffnX")) $("ffnX").addEventListener("input", e => update(+e.target.value));
    update(0);
  }


  /* ------------------------------------------------------------
     2. Widget: Residual Bus Propagation & RMSNorm Tracker (#w-residual-bus)
     ------------------------------------------------------------ */
  const resBusSvg = $("resBusSvg");
  const resBusStats = $("resBusStats");
  const resBusNormSeg = $("resBusNormSeg");
  let resBusMode = "prenorm";

  function renderResBus() {
    if (!resBusSvg || !resBusStats) return;

    const NS = "http://www.w3.org/2000/svg";
    resBusSvg.innerHTML = "";

    const W = 560, H = 220, P = { l: 48, r: 24, t: 20, b: 36 };
    const IW = W - P.l - P.r, IH = H - P.t - P.b;

    // Layers 0 to 64
    const fx = l => P.l + (l / 64) * IW;
    const fy = v => P.t + (1 - Math.min(1.0, v / 10.0)) * IH;

    [0, 16, 32, 48, 64].forEach(lVal => {
      const gx = fx(lVal);
      const l = document.createElementNS(NS, "line");
      l.setAttribute("class", "bp-grid");
      l.setAttribute("x1", gx); l.setAttribute("y1", P.t);
      l.setAttribute("x2", gx); l.setAttribute("y2", P.t + IH);
      resBusSvg.appendChild(l);

      const txt = document.createElementNS(NS, "text");
      txt.setAttribute("class", "bp-tick");
      txt.setAttribute("x", gx); txt.setAttribute("y", H - 8);
      txt.setAttribute("text-anchor", "middle");
      txt.textContent = `L${lVal}`;
      resBusSvg.appendChild(txt);
    });

    [1, 2, 5, 10].forEach(vVal => {
      const gy = fy(vVal);
      const l = document.createElementNS(NS, "line");
      l.setAttribute("class", "bp-grid");
      l.setAttribute("x1", P.l); l.setAttribute("y1", gy);
      l.setAttribute("x2", P.l + IW); l.setAttribute("y2", gy);
      resBusSvg.appendChild(l);

      const txt = document.createElementNS(NS, "text");
      txt.setAttribute("class", "bp-tick");
      txt.setAttribute("x", P.l - 8); txt.setAttribute("y", gy + 3);
      txt.setAttribute("text-anchor", "end");
      txt.textContent = `${vVal}σ`;
      resBusSvg.appendChild(txt);
    });

    let dPath = "";
    for (let layer = 0; layer <= 64; layer++) {
      let variance;
      if (resBusMode === "prenorm") {
        variance = 1.0 + 0.12 * Math.sin(layer / 4) + 0.05 * (layer / 64);
      } else {
        variance = 1.0 + Math.pow(layer / 20, 1.8);
      }
      const gx = fx(layer);
      const gy = fy(variance);
      dPath += (layer === 0 ? "M" : "L") + gx + "," + gy + " ";
    }

    const pathEl = document.createElementNS(NS, "path");
    pathEl.setAttribute("d", dPath);
    pathEl.setAttribute("fill", "none");
    pathEl.setAttribute("stroke", resBusMode === "prenorm" ? PAL.mint : PAL.rose);
    pathEl.setAttribute("stroke-width", "2.5");
    resBusSvg.appendChild(pathEl);

    resBusStats.innerHTML = `
      <div class="rf-stat-box">
        <div class="rf-head">Residual Communication Bus Arbitration</div>
        <div class="rf-row"><span>Bus Width</span><span class="val amber">d_model = 5,120</span></div>
        <div class="rf-row"><span>RMSNorm Stacking</span><span class="val mint">129 Core Norms (Pre-Branch)</span></div>
        <div class="rf-row"><span>Layer 64 Signal Variance</span><span class="val ${resBusMode === "prenorm" ? "mint" : "rose"}">${resBusMode === "prenorm" ? "1.05σ (Bounded) ✓" : "9.82σ (Exploded) ✗"}</span></div>
      </div>
      <div style="font-size:12px;color:var(--muted);line-height:1.45;padding:4px 2px">
        ${resBusMode === "prenorm" 
          ? "✓ Pre-RMSNorm ensures the identity path stays unattenuated while bounding the magnitude written back by 48 DeltaNet and 16 Attention mixers." 
          : "⚠️ Without RMSNorm bus arbitration, variance grows with depth $\\mathcal{O}(\\sqrt{L})$, destabilizing softmax logits and causing numerical overflow."}
      </div>
    `;
    katexRender(resBusStats);
  }

  if (resBusNormSeg) {
    bindGroup(resBusNormSeg, b => { resBusMode = b.dataset.mode; renderResBus(); });
  }
  renderResBus();
})();

/* ============================================================
   Section 07: Vision Front-End & Multimodal Projection Engine
   ============================================================ */
(function () {
  /* ------------------------------------------------------------
     1. Widget: 5-Stage Vision-to-Language Pipeline (#w-vision-pipe)
     ------------------------------------------------------------ */
  const visionPipe = $("visionPipe");
  const visionDetailCard = $("visionDetailCard");

  const VISION_STAGES = {
    "1": {
      name: "Stage 01 · Patchify & Spatial Tiling",
      tensor: "Input [B, C=3, H, W] → Patches [B, N_patches, 3×14×14 = 588]",
      desc: "Decomposes input RGB images into a regular 2D grid of non-overlapping 14×14 pixel patches. An initial 2D convolution projects 588 pixel values into a 1,536-dimensional linear embedding.",
      insight: "Dynamic native resolution support prevents geometric distortion and aspect-ratio skewing by tiling arbitrary aspect ratios into 224px coordinate blocks."
    },
    "2": {
      name: "Stage 02 · SigLIP Vision Transformer (ViT) Encoding",
      tensor: "Patches [B, N, 1536] → ViT Features [B, N, 1536]",
      desc: "A 30-layer Vision Transformer (16 attention heads, 6,144 FFN width) applies bidirectional self-attention across all image patches simultaneously, extracting semantic and fine-grained visual representations.",
      insight: "Trained using SigLIP pairwise sigmoid loss rather than CLIP softmax, eliminating the batch-size communication bottleneck and yielding sharper spatial features."
    },
    "3": {
      name: "Stage 03 · 2×2 Spatial Neighborhood Concatenation",
      tensor: "ViT Features [B, N, 1536] → Merged [B, N/4, 4×1536 = 6144]",
      desc: "Groups 2×2 adjacent spatial patch vectors into a single unified 6,144-dimensional representation. Compresses the total visual token count by exactly 75% (4× reduction).",
      insight: "Essential for long context budgets: an 800×600 screenshot shrinks from 2,494 raw patch tokens to just 624 merged tokens, enabling multi-image chat and hour-long video ingestion."
    },
    "4": {
      name: "Stage 04 · 2-Layer MLP Projector & Residual Alignment",
      tensor: "Merged [B, N/4, 6144] → Residual Stream [B, N/4, 5120]",
      desc: "A 2-layer MLP with SiLU activation transforms 6,144-dim vision vectors into the language model's 5,120-dim residual space: Linear(6144 → 6144) → SiLU → Linear(6144 → 5120).",
      insight: "Trained during multimodal SFT with precise output scaling to match the natural activation variance of text pretraining."
    },
    "5": {
      name: "Stage 05 · 3D MRoPE Coordinates & Residual Interleaving",
      tensor: "Residual Stream [B, N/4, 5120] with [T, H, W] Position IDs",
      desc: "Wraps visual tokens between `<|vision_start|>` and `<|vision_end|>` delimiters. 3D MRoPE assigns separate temporal, vertical, and horizontal rotary positional frequencies.",
      insight: "Enables the language model to understand native 2D layout geometry and video temporal continuity within the same rotary table."
    }
  };

  function renderVisionStage(stageNum) {
    if (!visionDetailCard) return;
    const info = VISION_STAGES[stageNum] || VISION_STAGES["1"];

    visionDetailCard.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px;border-bottom:1px solid var(--line);padding-bottom:6px">
        <span style="font-family:var(--display);font-size:15px;font-weight:700;color:var(--fg)">${info.name}</span>
        <span style="font-family:var(--mono);font-size:11px;color:var(--mint);font-weight:600">Stage ${stageNum} / 5</span>
      </div>
      <div style="background:var(--bg-2);padding:8px 12px;border-radius:6px;box-shadow:inset 0 0 0 1px var(--line);font-family:var(--mono);font-size:11px;color:var(--amber);margin-bottom:8px">
        ${info.tensor}
      </div>
      <div style="font-size:12.5px;color:var(--muted);line-height:1.5;margin-bottom:6px">
        ${info.desc}
      </div>
      <div style="font-size:11.5px;color:var(--faint);font-family:var(--mono)">
        <b>Architectural Key:</b> ${info.insight}
      </div>
    `;
  }

  if (visionPipe) {
    visionPipe.querySelectorAll(".stage").forEach(btn => {
      btn.addEventListener("click", () => {
        visionPipe.querySelectorAll(".stage").forEach(s => s.classList.remove("active"));
        btn.classList.add("active");
        renderVisionStage(btn.dataset.stage);
      });
    });
    enableTabs(visionPipe, ".stage");                              // div tabs: focus + Enter/Space
  }
  renderVisionStage("1");


  /* ------------------------------------------------------------
     2. Widget: Vision Token Economics Calculator (#w-vision-calc)
     ------------------------------------------------------------ */
  const visCalcReadout = $("visCalcReadout");
  const visPresetSel = $("visPresetSel");
  const visMergeToggle = $("visMergeToggle");

  const PRESETS = {
    "224x224x1":   { w: 224,  h: 224,  frames: 1,  name: "Standard Tile (224×224)" },
    "800x600x1":   { w: 800,  h: 600,  frames: 1,  name: "Web Screenshot (800×600)" },
    "1920x1080x1": { w: 1920, h: 1080, frames: 1,  name: "1080p Slide (1920×1080)" },
    "3840x2160x1": { w: 3840, h: 2160, frames: 1,  name: "4K High-Res Diagram (3840×2160)" },
    "448x448x16":  { w: 448,  h: 448,  frames: 16, name: "Short Video Clip (448×448 · 16f)" },
    "640x360x64":  { w: 640,  h: 360,  frames: 64, name: "Long Video Sequence (640×360 · 64f)" }
  };

  function renderVisionCalc() {
    if (!visCalcReadout) return;
    const presetKey = visPresetSel ? visPresetSel.value : "800x600x1";
    const useMerge = visMergeToggle ? visMergeToggle.value === "true" : true;

    const p = PRESETS[presetKey] || PRESETS["800x600x1"];
    const patchesPerFrame = Math.ceil(p.w / 14) * Math.ceil(p.h / 14);
    const rawTokens = patchesPerFrame * p.frames;
    const finalTokens = useMerge ? Math.ceil(rawTokens / 4) : rawTokens;

    // VRAM Embedding size (BF16 = 2 bytes * 5120 dim per token)
    const embeddingBytes = finalTokens * 5120 * 2;
    const embeddingMb = (embeddingBytes / (1024 * 1024)).toFixed(2);

    // Context budget usage of 262,144 tokens
    const contextPct = ((finalTokens / 262144) * 100).toFixed(2);

    visCalcReadout.innerHTML = `
      <div class="calc-stat-grid">
        <div class="cs-card">
          <div class="cs-num" style="color:var(--mint)">${finalTokens.toLocaleString()} Tokens</div>
          <div class="cs-lbl">${useMerge ? "Ingested Tokens (4× Merged)" : "Raw Patch Tokens"}</div>
        </div>
        <div class="cs-card">
          <div class="cs-num">${rawTokens.toLocaleString()}</div>
          <div class="cs-lbl">Raw 14×14 Patches</div>
        </div>
        <div class="cs-card">
          <div class="cs-num" style="color:var(--amber)">${embeddingMb} MB</div>
          <div class="cs-lbl">Residual VRAM Footprint</div>
        </div>
        <div class="cs-card">
          <div class="cs-num" style="color:var(--iris)">${contextPct}%</div>
          <div class="cs-lbl">Of 262k Context Budget</div>
        </div>
      </div>
      <div style="font-family:var(--mono);font-size:11px;color:var(--muted);line-height:1.5;border-top:1px solid var(--line);padding-top:8px">
        Input: <b>${p.name}</b> · Patches: <b>${Math.ceil(p.w/14)} × ${Math.ceil(p.h/14)}</b> per frame × ${p.frames} frame(s) · Compression: <b>${useMerge ? "4× Spatial Merge Active ✓" : "1:1 Raw Patches"}</b>.
      </div>
    `;
  }

  [visPresetSel, visMergeToggle].forEach(el => {
    if (el) {
      el.addEventListener("change", renderVisionCalc);
      el.addEventListener("input", renderVisionCalc);
    }
  });
  renderVisionCalc();
})();

/* ============================================================
   Section 08: Multi-Token Prediction & Speculative Engine
   ============================================================ */
(function () {
  /* ------------------------------------------------------------
     1. Widget: Speculative Decoding Simulator (#w-mtp)
     ------------------------------------------------------------ */
  const row = $("mtpRow");
  if (!row) return;
  const out = $("mtpOut");
  const K_EL = $("mtpK");
  const DOMAIN_SEL = $("mtpDomainSel");
  const AUTO_BTN = $("mtpAuto");

  // §08's acceptance formula is the single speedup convention — regenerate every
  // "~N.N×" option label from it so the dropdown can never drift from §08's prose.
  DOMAIN_SEL?.querySelectorAll("option").forEach(o => {
    const b = parseFloat(o.value);
    if (Number.isFinite(b)) o.textContent = o.textContent.replace(/~\d+(\.\d+)?\s*×/, `~${mtpSpeedup(b, 2).toFixed(2)}×`);
  });

  const words = [
    "def ", "calculate_loss", "(", "self", ", ", "hidden_states", ", ", "labels", "):", "\n    ",
    "logits ", "= ", "self", ".", "lm_head", "(", "hidden_states", ")", "\n    ",
    "shift_logits ", "= ", "logits", "[", "...", ", ", ":-1", ", ", ":", "]", ".", "contiguous", "()", "\n    ",
    "shift_labels ", "= ", "labels", "[", "...", ", ", "1:", "]", ".", "contiguous", "()", "\n    ",
    "return ", "self", ".", "loss_fct", "(", "shift_logits", ", ", "shift_labels", ")"
  ];
  let pos = 0, timer = null, accepted = 0, proposed = 0, steps = 0, totalTokens = 0;

  function push(cls, text) {
    const t = document.createElement("div");
    t.className = "tok " + cls;
    t.textContent = text.replace("\n", "\\n").trim() || "·";
    if (text.includes("\n")) t.style.opacity = "0.75";
    row.appendChild(t);
  }

  function step() {
    const beta = DOMAIN_SEL ? +DOMAIN_SEL.value : 0.78;
    const K = K_EL ? +K_EL.value : 2;

    if (pos >= words.length) {
      stop();
      setTimeout(reset, 2400);
      return;
    }
    steps++;

    // Draft K continuations from MTP heads
    const drafts = [];
    for (let k = 0; k < K && pos + 1 + k < words.length; k++) {
      drafts.push({ ok: Math.random() < beta, word: words[pos + 1 + k] });
      proposed++;
    }

    // Root target token (always generated)
    push("cur", words[pos]);
    totalTokens++;

    // Verify drafts sequentially
    let acc = 0;
    while (acc < drafts.length && drafts[acc].ok) acc++;

    for (let k = 0; k < acc; k++) {
      push("acc", drafts[k].word);
      accepted++;
      totalTokens++;
    }

    if (acc < drafts.length) {
      // Rejection: target resamples the first disagreement token
      push("rej", drafts[acc].word);
      const correction = words[pos + 1 + acc] || "";
      push("cur", correction);
      totalTokens++;
      pos += acc + 2; // Jump past accepted drafts + rejected + correction
    } else {
      pos += acc + 1;
    }

    const effRate = mtpEtoks(beta, K);
    const speedup = (totalTokens / Math.max(1, steps)).toFixed(2);

    out.innerHTML = `
      <div class="row"><span>Target Forward Passes</span><span class="v">${steps} passes</span></div>
      <div class="row"><span>Total Emitted Tokens</span><span class="v a">${totalTokens} tokens</span></div>
      <div class="row"><span>Draft Acceptance Rate</span><span class="v m">${accepted}/${proposed} = ${(accepted / Math.max(1, proposed) * 100).toFixed(0)}%</span></div>
      <div class="row"><span>Effective Speedup Multiplier</span><span class="v" style="color:var(--mint);font-size:13px"><b>${speedup}×</b> speedup</span></div>
      <div class="row"><span>Theoretical E[Tokens/Pass] @ β=${beta.toFixed(2)}, K=${K}</span><span class="v i">${effRate.toFixed(2)} tok/pass</span></div>
      <div style="color:var(--faint);font-size:11px;margin-top:6px;line-height:1.4">
        <span style="color:var(--mint)">■ Green</span> = Accepted MTP Drafts · <span style="color:var(--rose)">■ Red</span> = Rejected Tail · <span style="color:var(--amber)">■ Amber</span> = Target Root / Resampled Token
      </div>
    `;

    if (pos >= words.length) {
      stop();
      setTimeout(reset, 2800);
    }
  }

  function stop() {
    if (timer) { clearInterval(timer); timer = null; }
    AUTO_BTN.textContent = "play \u25b6";
    AUTO_BTN.setAttribute("aria-label", "Auto-run speculative decoding cycles");
    AUTO_BTN.classList.remove("on");
  }

  function reset() {
    stop();
    row.innerHTML = "";
    pos = 0; steps = 0; accepted = 0; proposed = 0; totalTokens = 0;
    out.innerHTML = `<div class="row" style="border:0"><span style="color:var(--faint)">press <b style="color:var(--amber)">step</b> or <b style="color:var(--amber)">play</b> to start MTP speculative drafting</span><span class="v"></span></div>`;
  }

  const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
  function play() {
    AUTO_BTN.textContent = "pause \u23f8";
    AUTO_BTN.setAttribute("aria-label", "Pause auto-run");
    AUTO_BTN.classList.add("on");
    timer = setInterval(step, 900);
  }
  $("mtpStep").addEventListener("click", step);
  AUTO_BTN.addEventListener("click", () => {
    if (timer) { stop(); return; }
    if (REDUCED) { step(); return; }               // reduced motion: single step, no autoplay loop
    play();
  });
  let autoPausedByTab = false;
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && timer) { stop(); autoPausedByTab = true; }
    else if (!document.hidden && autoPausedByTab && !timer) { autoPausedByTab = false; play(); }
  });
  $("mtpReset").addEventListener("click", reset);
  reset();


  /* ------------------------------------------------------------
     2. Widget: MTP Architectural Layout & Verification Tree (#w-mtp-tree)
     ------------------------------------------------------------ */
  const mtpTreeSvg = $("mtpTreeSvg");
  const mtpViewSeg = $("mtpViewSeg");
  const mtpTreeDepth = $("mtpTreeDepth");
  const mtpMaskTopologySeg = $("mtpMaskTopologySeg");
  const mtpMaskTableWrap = $("mtpMaskTableWrap");
  const mtpMaskExplainer = $("mtpMaskExplainer");
  const mtpVerifPromptText = $("mtpVerifPromptText");
  const mtpVerifStepBtn = $("mtpVerifStepBtn");
  const mtpVerifCandidates = $("mtpVerifCandidates");
  const mtpVerifComparison = $("mtpVerifComparison");
  const mtpStatWeight = $("mtpStatWeight");
  const mtpStatLatency = $("mtpStatLatency");
  const mtpStatVocab = $("mtpStatVocab");
  const mtpStatDist = $("mtpStatDist");

  // Dynamic Inspector elements
  const mtpInspectBadge = $("mtpInspectBadge");
  const mtpInspectTitle = $("mtpInspectTitle");
  const mtpInspectMath = $("mtpInspectMath");
  const mtpInspectDesc = $("mtpInspectDesc");
  const mtpInspectShape = $("mtpInspectShape");
  const mtpInspectDtype = $("mtpInspectDtype");

  let currentView = "arch";
  let depthK = 2;
  let selectedNode = "backbone";
  let maskTopo = "chain";
  let verifPresetIdx = 0;

  const MTP_NODES = {
    backbone: {
      badge: "TARGET BACKBONE",
      title: "Main Transformer Backbone (Layer 64)",
      math: "h_t = RMSNorm(Block₆₄(h_t⁽⁶³⁾)) ∈ ℝ^(B × 1 × 5120)",
      desc: "Produces the final 5120-dimensional hidden state h_t across all 64 layers (48 DeltaNet + 16 full-attention layers). This activation vector serves as the foundational root state for both standard generation and speculative branch expansion.",
      shape: "[Batch, 1, 5120]",
      dtype: "bfloat16 (Layer 64)"
    },
    lm_head: {
      badge: "VOCABULARY PROJECTION",
      title: "Shared Unembedding LM Head",
      math: "x_(t+1) ~ Softmax(W_u · RMSNorm(h_t)),  W_u ∈ ℝ^(248320 × 5120)",
      desc: "Tied or shared unembedding matrix mapping 5120-dim hidden states to 248,320 vocabulary logits. In Qwen 3.8, all MTP draft heads share this exact same LM Head weight matrix, avoiding 1.27B duplicated parameters (~2.6 GB VRAM savings).",
      shape: "[248320, 5120]",
      dtype: "bfloat16 (Shared)"
    },
    head_1: {
      badge: "MTP DRAFT HEAD (K=1)",
      title: "MTP Head 1 (Depth-1 Speculative Head)",
      math: "h̃_(t+1) = Block_MTP1([RMSNorm(h_t); Embed(x_(t+1))])",
      desc: "A dedicated hybrid transformer block (DeltaNet + RMSNorm) dedicated to predicting the 1-step lookahead token. It receives the concatenated backbone representation and embedding of the root token, projecting to draft token x̃_(t+2).",
      shape: "[Batch, 1, 5120]",
      dtype: "bfloat16 (~1.3B params)"
    },
    head_2: {
      badge: "MTP DRAFT HEAD (K=2)",
      title: "MTP Head 2 (Depth-2 Speculative Head)",
      math: "h̃_(t+2) = Block_MTP2([RMSNorm(h̃_(t+1)); Embed(x̃_(t+2))])",
      desc: "The second speculative head predicts the 2-step lookahead token x̃_(t+3). In Qwen 3.8-27B native configuration, K=2 allows proposing 3 candidate tokens simultaneously while keeping parameter overhead capped at +4.8% (~2.6 GB).",
      shape: "[Batch, 1, 5120]",
      dtype: "bfloat16 (~1.3B params)"
    },
    head_3: {
      badge: "MTP DRAFT HEAD (K=3)",
      title: "MTP Head 3 (Depth-3 Speculative Head)",
      math: "h̃_(t+3) = Block_MTP3([RMSNorm(h̃_(t+2)); Embed(x̃_(t+3))])",
      desc: "Extended depth-3 speculative head for ultra-high speedup on highly predictable domains (e.g. structured code syntax / JSON serialization). Proposes candidate x̃_(t+4).",
      shape: "[Batch, 1, 5120]",
      dtype: "bfloat16 (~1.3B params)"
    },
    verifier: {
      badge: "SINGLE-PASS VERIFIER",
      title: "Target Backbone Parallel Tree Verifier",
      math: "P(x_(t+1), x̃_(t+2), x̃_(t+3) | x_(≤t)) = Forward_Target(Batch, M_tree)",
      desc: "Executes one single forward pass across all proposed candidate tokens using a 2D causal tree attention mask. Verification latency is identical to standard 1-token generation because memory bandwidth reads weights only once for the entire candidate batch.",
      shape: "[Batch, K+1, 5120]",
      dtype: "bfloat16 (Target Weights)"
    }
  };

  const VERIF_PRESETS = [
    {
      domain: "Structured Python AST",
      prompt: "def compute_attention_scores(q, k, v):",
      tokens: [
        { role: "Root (t+1)", text: "\\n    ", prob: "98.4%", accept: true, reason: "Definite indentation" },
        { role: "Draft 1 (t+2)", text: "scores = ", prob: "94.2%", accept: true, reason: "Standard idiom" },
        { role: "Draft 2 (t+3)", text: "torch.matmul(", prob: "91.7%", accept: true, reason: "Matches AST prior" },
        { role: "Draft 3 (t+4)", text: "q, k.transpose(-1,-2))", prob: "88.5%", accept: true, reason: "Tensor matmul arg" }
      ]
    },
    {
      domain: "TypeScript Type Annotation",
      prompt: "type TransformerConfig = { hidden_size: ",
      tokens: [
        { role: "Root (t+1)", text: "number", prob: "96.1%", accept: true, reason: "Field type" },
        { role: "Draft 1 (t+2)", text: "; num_heads: ", prob: "92.8%", accept: true, reason: "Config property" },
        { role: "Draft 2 (t+3)", text: "string", prob: "31.4%", accept: false, correction: "number", reason: "Target corrects to 'number'" },
        { role: "Draft 3 (t+4)", text: "; num_layers", prob: "0.0%", accept: false, pruned: true, reason: "Pruned after divergence" }
      ]
    },
    {
      domain: "SQL Aggregation Query",
      prompt: "SELECT user_id, COUNT(id) AS total_orders",
      tokens: [
        { role: "Root (t+1)", text: " FROM ", prob: "99.1%", accept: true, reason: "SQL keyword" },
        { role: "Draft 1 (t+2)", text: "orders ", prob: "95.6%", accept: true, reason: "Source table" },
        { role: "Draft 2 (t+3)", text: "GROUP BY ", prob: "93.0%", accept: true, reason: "Aggregation clause" },
        { role: "Draft 3 (t+4)", text: "user_id;", prob: "94.8%", accept: true, reason: "Group key" }
      ]
    }
  ];

  function updateInspector(nodeKey) {
    selectedNode = nodeKey;
    const data = MTP_NODES[nodeKey] || MTP_NODES.backbone;
    if (mtpInspectBadge) mtpInspectBadge.textContent = data.badge;
    if (mtpInspectTitle) mtpInspectTitle.textContent = data.title;
    if (mtpInspectMath) mtpInspectMath.textContent = data.math;
    if (mtpInspectDesc) mtpInspectDesc.textContent = data.desc;
    if (mtpInspectShape) mtpInspectShape.textContent = data.shape;
    if (mtpInspectDtype) mtpInspectDtype.textContent = data.dtype;

    if (mtpTreeSvg) {
      mtpTreeSvg.querySelectorAll("g.clickable-node").forEach(g => {
        if (g.getAttribute("data-node") === nodeKey) {
          g.classList.add("selected");
        } else {
          g.classList.remove("selected");
        }
      });
    }
  }

  function renderMtpTreeSvg() {
    if (!mtpTreeSvg) return;
    const NS = "http://www.w3.org/2000/svg";
    mtpTreeSvg.innerHTML = "";

    const H = depthK === 3 ? 290 : 250;
    mtpTreeSvg.setAttribute("viewBox", `0 0 680 ${H}`);

    // Definitions
    const defs = document.createElementNS(NS, "defs");
    defs.innerHTML = `
      <marker id="mtpArrowHead" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
        <polygon points="0 0, 6 2, 0 4" fill="rgba(26,22,18,0.45)" />
      </marker>
      <marker id="mtpArrowMint" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
        <polygon points="0 0, 6 2, 0 4" fill="${PAL.mint}" />
      </marker>
      <marker id="mtpArrowAmber" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
        <polygon points="0 0, 6 2, 0 4" fill="${PAL.amber}" />
      </marker>
      <marker id="mtpArrowIris" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
        <polygon points="0 0, 6 2, 0 4" fill="${PAL.iris}" />
      </marker>
      <marker id="mtpArrowSky" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
        <polygon points="0 0, 6 2, 0 4" fill="${PAL.sky}" />
      </marker>
    `;
    mtpTreeSvg.appendChild(defs);

    function drawNode(key, x, y, w, h, title, sub, sub2, fill, stroke, textFill) {
      const g = document.createElementNS(NS, "g");
      g.setAttribute("class", "clickable-node" + (key === selectedNode ? " selected" : ""));
      g.setAttribute("data-node", key);
      g.setAttribute("tabindex", "0");
      g.setAttribute("role", "button");
      g.setAttribute("aria-label", title);

      const rect = document.createElementNS(NS, "rect");
      rect.setAttribute("x", x); rect.setAttribute("y", y);
      rect.setAttribute("width", w); rect.setAttribute("height", h);
      rect.setAttribute("rx", "6");
      rect.setAttribute("fill", fill);
      rect.setAttribute("stroke", stroke);
      rect.setAttribute("stroke-width", "1.2");
      g.appendChild(rect);

      const t1 = document.createElementNS(NS, "text");
      t1.setAttribute("x", x + w / 2); t1.setAttribute("y", y + 17);
      t1.setAttribute("text-anchor", "middle");
      t1.setAttribute("font-family", '"IBM Plex Mono", ui-monospace, monospace');
      t1.setAttribute("font-size", "10.5");
      t1.setAttribute("font-weight", "600");
      t1.setAttribute("fill", textFill || PAL.ink);
      t1.textContent = title;
      g.appendChild(t1);

      if (sub) {
        const t2 = document.createElementNS(NS, "text");
        t2.setAttribute("x", x + w / 2); t2.setAttribute("y", y + 32);
        t2.setAttribute("text-anchor", "middle");
        t2.setAttribute("font-family", '"IBM Plex Mono", ui-monospace, monospace');
        t2.setAttribute("font-size", "8.5");
        t2.setAttribute("fill", "rgba(26,22,18,0.65)");
        t2.textContent = sub;
        g.appendChild(t2);
      }

      if (sub2) {
        const t3 = document.createElementNS(NS, "text");
        t3.setAttribute("x", x + w / 2); t3.setAttribute("y", y + 46);
        t3.setAttribute("text-anchor", "middle");
        t3.setAttribute("font-family", '"IBM Plex Mono", ui-monospace, monospace');
        t3.setAttribute("font-size", "8");
        t3.setAttribute("font-weight", "600");
        t3.setAttribute("fill", textFill || PAL.ink);
        t3.textContent = sub2;
        g.appendChild(t3);
      }

      g.addEventListener("click", () => updateInspector(key));
      g.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          updateInspector(key);
        }
      });
      mtpTreeSvg.appendChild(g);
      return { x, y, w, h };
    }

    function drawLine(x1, y1, x2, y2, color, markerId) {
      const line = document.createElementNS(NS, "line");
      line.setAttribute("x1", x1); line.setAttribute("y1", y1);
      line.setAttribute("x2", x2); line.setAttribute("y2", y2);
      line.setAttribute("stroke", color || "rgba(26,22,18,0.3)");
      line.setAttribute("stroke-width", "1.5");
      if (markerId) line.setAttribute("marker-end", `url(#${markerId})`);
      mtpTreeSvg.appendChild(line);
    }

    // 1. Target Backbone
    const backboneY = depthK === 1 ? 65 : depthK === 2 ? 100 : 120;
    const bb = drawNode("backbone", 20, backboneY, 130, 68, "Main Backbone", "Layer 64 Hidden (h_t)", "d=5120 · 27.58B", "rgba(184,52,31,0.1)", PAL.amber, PAL.amber);

    // 2. Shared LM Head
    const lm = drawNode("lm_head", 185, 18, 150, 52, "Shared LM Head", "Softmax(W_u · h_t)", "→ Root x_{t+1} (248k)", "rgba(29,110,107,0.1)", PAL.mint, PAL.mint);

    // 3. MTP Head 1 (K=1)
    const h1 = drawNode("head_1", 185, 82, 150, 52, "MTP Head 1 (K=1)", "RMSNorm + Hybrid Block", "→ Draft x̃_{t+2}", "rgba(43,58,138,0.1)", PAL.iris, PAL.iris);

    // 4. MTP Head 2 (K=2)
    let h2 = null;
    if (depthK >= 2) {
      h2 = drawNode("head_2", 185, 146, 150, 52, "MTP Head 2 (K=2)", "RMSNorm + Hybrid Block", "→ Draft x̃_{t+3}", "rgba(186,117,23,0.1)", PAL.amber, PAL.amber);
    }

    // 5. MTP Head 3 (K=3)
    let h3 = null;
    if (depthK >= 3) {
      h3 = drawNode("head_3", 185, 210, 150, 52, "MTP Head 3 (K=3)", "RMSNorm + Hybrid Block", "→ Draft x̃_{t+4}", "rgba(43,122,158,0.1)", PAL.sky, PAL.sky);
    }

    // 6. Target Verifier Container
    const verifHeight = depthK === 1 ? 140 : depthK === 2 ? 195 : 245;
    const vG = document.createElementNS(NS, "g");
    vG.setAttribute("class", "clickable-node" + (selectedNode === "verifier" ? " selected" : ""));
    vG.setAttribute("data-node", "verifier");
    vG.setAttribute("tabindex", "0");
    vG.setAttribute("role", "button");
    vG.setAttribute("aria-label", "Single-Pass Target Verifier");

    const vRect = document.createElementNS(NS, "rect");
    vRect.setAttribute("x", "380"); vRect.setAttribute("y", "18");
    vRect.setAttribute("width", "280"); vRect.setAttribute("height", verifHeight);
    vRect.setAttribute("rx", "6");
    vRect.setAttribute("fill", "rgba(26,22,18,0.03)");
    vRect.setAttribute("stroke", "rgba(26,22,18,0.22)");
    vRect.setAttribute("stroke-width", "1.2");
    vG.appendChild(vRect);

    const vt1 = document.createElementNS(NS, "text");
    vt1.setAttribute("x", "520"); vt1.setAttribute("y", "38");
    vt1.setAttribute("text-anchor", "middle");
    vt1.setAttribute("font-family", '"IBM Plex Mono", ui-monospace, monospace');
    vt1.setAttribute("font-size", "10");
    vt1.setAttribute("font-weight", "700");
    vt1.setAttribute("letter-spacing", "0.05em");
    vt1.setAttribute("fill", PAL.ink);
    vt1.textContent = "SINGLE-PASS TARGET VERIFIER";
    vG.appendChild(vt1);

    const vt2 = document.createElementNS(NS, "text");
    vt2.setAttribute("x", "520"); vt2.setAttribute("y", "52");
    vt2.setAttribute("text-anchor", "middle");
    vt2.setAttribute("font-family", '"IBM Plex Mono", ui-monospace, monospace');
    vt2.setAttribute("font-size", "8");
    vt2.setAttribute("fill", "rgba(26,22,18,0.5)");
    vt2.textContent = "1 Target Forward Pass · 2D Tree Mask";
    vG.appendChild(vt2);

    // Inner slots in Verifier
    const slots = [
      { label: "Pos 0: Root x_{t+1}", color: PAL.mint },
      { label: "Pos 1: Draft x̃_{t+2} (Head 1)", color: PAL.iris },
      ...(depthK >= 2 ? [{ label: "Pos 2: Draft x̃_{t+3} (Head 2)", color: PAL.amber }] : []),
      ...(depthK >= 3 ? [{ label: "Pos 3: Draft x̃_{t+4} (Head 3)", color: PAL.sky }] : [])
    ];

    slots.forEach((s, idx) => {
      const sy = 66 + idx * 36;
      const sr = document.createElementNS(NS, "rect");
      sr.setAttribute("x", "395"); sr.setAttribute("y", sy);
      sr.setAttribute("width", "250"); sr.setAttribute("height", "28");
      sr.setAttribute("rx", "4");
      sr.setAttribute("fill", "#fff");
      sr.setAttribute("stroke", s.color);
      sr.setAttribute("stroke-width", "1");
      vG.appendChild(sr);

      const st = document.createElementNS(NS, "text");
      st.setAttribute("x", "405"); st.setAttribute("y", sy + 18);
      st.setAttribute("font-family", '"IBM Plex Mono", ui-monospace, monospace');
      st.setAttribute("font-size", "9");
      st.setAttribute("font-weight", "600");
      st.setAttribute("fill", s.color);
      st.textContent = s.label;
      vG.appendChild(st);

      const stScore = document.createElementNS(NS, "text");
      stScore.setAttribute("x", "635"); stScore.setAttribute("y", sy + 18);
      stScore.setAttribute("text-anchor", "end");
      stScore.setAttribute("font-family", '"IBM Plex Mono", ui-monospace, monospace');
      stScore.setAttribute("font-size", "8");
      stScore.setAttribute("fill", "rgba(26,22,18,0.6)");
      stScore.textContent = `P(x|ctx) ✓`;
      vG.appendChild(stScore);
    });

    vG.addEventListener("click", () => updateInspector("verifier"));
    vG.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        updateInspector("verifier");
      }
    });
    mtpTreeSvg.appendChild(vG);

    // Connector arrows from Backbone
    const bbMidY = bb.y + bb.h / 2;
    drawLine(150, bbMidY, 185, 44, "rgba(26,22,18,0.3)", "mtpArrowHead");
    drawLine(150, bbMidY, 185, 108, "rgba(26,22,18,0.3)", "mtpArrowHead");
    if (depthK >= 2) drawLine(150, bbMidY, 185, 172, "rgba(26,22,18,0.3)", "mtpArrowHead");
    if (depthK >= 3) drawLine(150, bbMidY, 185, 236, "rgba(26,22,18,0.3)", "mtpArrowHead");

    // Connector arrows into Verifier
    drawLine(335, 44, 395, 80, PAL.mint, "mtpArrowMint");
    drawLine(335, 108, 395, 116, PAL.iris, "mtpArrowIris");
    if (depthK >= 2) drawLine(335, 172, 395, 152, PAL.amber, "mtpArrowAmber");
    if (depthK >= 3) drawLine(335, 236, 395, 188, PAL.sky, "mtpArrowSky");
  }

  function renderMtpMaskTable() {
    if (!mtpMaskTableWrap) return;

    let headers = [];
    let fullLabels = [];
    let matrix = [];
    let descriptions = [];

    if (maskTopo === "chain") {
      headers = [
        "Prefix",
        "Root",
        "Draft 1",
        ...(depthK >= 2 ? ["Draft 2"] : []),
        ...(depthK >= 3 ? ["Draft 3"] : [])
      ];
      fullLabels = [
        "Prefix (x ≤ t)",
        "Root (x_{t+1})",
        "Draft 1 (x̃_{t+2})",
        ...(depthK >= 2 ? ["Draft 2 (x̃_{t+3})"] : []),
        ...(depthK >= 3 ? ["Draft 3 (x̃_{t+4})"] : [])
      ];

      const N = headers.length;
      for (let i = 0; i < N; i++) {
        const row = [];
        for (let j = 0; j < N; j++) {
          row.push(j <= i ? 1 : 0);
        }
        matrix.push(row);
      }

      descriptions = [
        "Prefix tokens attend strictly to prior context history.",
        "Root token x_{t+1} attends to Prefix history and self.",
        "Draft token x̃_{t+2} attends to Prefix, Root, and self (cannot see future drafts).",
        ...(depthK >= 2 ? ["Draft token x̃_{t+3} attends to Prefix, Root, Draft 1, and self."] : []),
        ...(depthK >= 3 ? ["Draft token x̃_{t+4} attends to the full linear speculative branch."] : [])
      ];
    } else {
      // Branching Tree Beam 2
      headers = [
        "Prefix",
        "Root",
        "1A",
        "1B",
        "2A",
        "2B"
      ];
      fullLabels = [
        "Prefix (x ≤ t)",
        "Root (x_{t+1})",
        "Draft 1A (Branch A)",
        "Draft 1B (Branch B)",
        "Draft 2A (Branch A)",
        "Draft 2B (Branch B)"
      ];
      matrix = [
        [1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0], // Sibling 1B sees Prefix+Root, but NOT 1A!
        [1, 1, 1, 0, 1, 0], // 2A descends from 1A
        [1, 1, 0, 1, 0, 1]  // 2B descends from 1B
      ];
      descriptions = [
        "Prefix tokens attend to prior context history.",
        "Root token attends to prefix history and self.",
        "Branch A (Depth 1) attends to Prefix, Root, and self.",
        "Branch B (Depth 1) attends to Prefix and Root, but CANNOT attend to Branch A (independent hypothesis).",
        "Branch A (Depth 2) attends to Prefix, Root, Draft 1A, and self.",
        "Branch B (Depth 2) attends to Prefix, Root, Draft 1B, and self (completely isolated from Branch A)."
      ];
    }

    let html = `<table class="mtp-mask-table"><thead><tr><th>Query ↓ \\ Key →</th>`;
    headers.forEach((h, idx) => {
      html += `<th title="${fullLabels[idx]}">${h}</th>`;
    });
    html += `</tr></thead><tbody>`;

    matrix.forEach((row, rIdx) => {
      html += `<tr data-row="${rIdx}"><th style="text-align:left" title="${fullLabels[rIdx]}">${headers[rIdx]}</th>`;
      row.forEach((val, cIdx) => {
        const isAllowed = val === 1;
        const cls = isAllowed ? "allowed" : "masked";
        const txt = isAllowed ? "1" : "0";
        html += `<td class="mtp-mask-cell ${cls}" data-row="${rIdx}" data-col="${cIdx}" title="Query: ${fullLabels[rIdx]} → Key: ${fullLabels[cIdx]} : ${isAllowed ? 'Allowed (0)' : 'Masked (-∞)'}">${txt}</td>`;
      });
      html += `</tr>`;
    });
    html += `</tbody></table>`;

    mtpMaskTableWrap.innerHTML = html;

    // Add interactivity to mask cells and rows
    mtpMaskTableWrap.querySelectorAll(".mtp-mask-cell").forEach(cell => {
      cell.addEventListener("mouseenter", () => {
        const r = parseInt(cell.getAttribute("data-row"), 10);
        const c = parseInt(cell.getAttribute("data-col"), 10);
        const isAllowed = cell.classList.contains("allowed");
        if (mtpMaskExplainer) {
          mtpMaskExplainer.innerHTML = `<strong>Query:</strong> ${fullLabels[r]} &nbsp;→&nbsp; <strong>Key:</strong> ${fullLabels[c]} : <span style="color:${isAllowed ? 'var(--mint)' : 'var(--amber)'}">${isAllowed ? '✓ Attending Allowed (0 dB)' : '✕ Masked (-∞ dB)'}</span>. ${descriptions[r]}`;
        }
      });
      cell.addEventListener("mouseleave", () => {
        if (mtpMaskExplainer) {
          mtpMaskExplainer.textContent = "Hover over any cell or token row in the matrix to see its valid attention receptive field.";
        }
      });
    });
  }

  function renderMtpVerifCycle() {
    if (!mtpVerifCandidates || !mtpVerifComparison) return;

    const preset = VERIF_PRESETS[verifPresetIdx % VERIF_PRESETS.length];
    if (mtpVerifPromptText) {
      mtpVerifPromptText.innerHTML = `<code>${preset.prompt}</code> <span style="font-size:10px;color:var(--faint);margin-left:6px">(${preset.domain})</span>`;
    }

    const visibleTokens = preset.tokens.slice(0, depthK + 1);
    let acceptedCount = 0;
    let diverged = false;

    let candHtml = "";
    visibleTokens.forEach((t, idx) => {
      let isAcc = t.accept && !diverged;
      if (isAcc) acceptedCount++;
      else if (!diverged) diverged = true;

      const cls = isAcc ? "accepted" : "rejected";
      const statusText = isAcc ? "✓ ACCEPTED" : (t.pruned ? "— PRUNED" : `✕ RESAMPLED (${t.correction || 'target'})`);
      const statusCls = isAcc ? "accept" : "reject";

      candHtml += `
        <div class="verif-candidate-row ${cls}">
          <span class="verif-role-tag">${t.role}</span>
          <span class="verif-token-text"><code>"${t.text}"</code></span>
          <span class="verif-prob-val">P = ${t.prob}</span>
          <span class="verif-status-badge ${statusCls}">${statusText}</span>
        </div>
      `;
    });
    mtpVerifCandidates.innerHTML = candHtml;

    // Comparison summary
    const totalTokensEvaluated = depthK + 1;
    const seqLatencyMs = (totalTokensEvaluated * 12.0).toFixed(1);
    const mtpLatencyMs = (12.6).toFixed(1);
    const speedup = (seqLatencyMs / mtpLatencyMs).toFixed(2);
    const acceptedTokensTotal = diverged ? acceptedCount + 1 : acceptedCount; // includes corrected target token if resampled

    mtpVerifComparison.innerHTML = `
      <div class="verif-cmp-card">
        <div class="cmp-title">Sequential Generation (${totalTokensEvaluated} passes)</div>
        <div class="cmp-stat">${seqLatencyMs} ms <span style="font-size:10px;color:var(--muted)">(${totalTokensEvaluated} × memory load)</span></div>
      </div>
      <div class="verif-cmp-card">
        <div class="cmp-title">MTP Single-Pass Tree (${acceptedTokensTotal} tokens accepted)</div>
        <div class="cmp-stat mint">${mtpLatencyMs} ms <span style="font-size:10px;color:var(--mint);font-weight:600">(${speedup}× faster)</span></div>
      </div>
    `;
  }

  function updateMtpDepth(newK) {
    depthK = newK;
    if (mtpStatWeight) {
      mtpStatWeight.textContent = newK === 1 ? "+2.4% (~1.3 GB)" : newK === 2 ? "+4.8% (~2.6 GB)" : "+7.2% (~3.9 GB)";
    }
    renderMtpTreeSvg();
    renderMtpMaskTable();
    renderMtpVerifCycle();
  }

  function setupMtpTreeWidget() {
    // View tabs
    if (mtpViewSeg) {
      mtpViewSeg.querySelectorAll("button").forEach(btn => {
        btn.addEventListener("click", () => {
          mtpViewSeg.querySelectorAll("button").forEach(b => {
            b.classList.remove("on");
            b.setAttribute("aria-pressed", "false");
          });
          btn.classList.add("on");
          btn.setAttribute("aria-pressed", "true");

          currentView = btn.getAttribute("data-view");
          const panelArch = $("mtpPanelArch");
          const panelMask = $("mtpPanelMask");
          const panelVerif = $("mtpPanelVerif");

          if (panelArch) panelArch.style.display = currentView === "arch" ? "block" : "none";
          if (panelMask) panelMask.style.display = currentView === "mask" ? "block" : "none";
          if (panelVerif) panelVerif.style.display = currentView === "verif" ? "block" : "none";
        });
      });
    }

    // Depth selector
    if (mtpTreeDepth) {
      mtpTreeDepth.addEventListener("change", (e) => {
        updateMtpDepth(parseInt(e.target.value, 10));
      });
    }

    // Mask topology toggle
    if (mtpMaskTopologySeg) {
      mtpMaskTopologySeg.querySelectorAll("button").forEach(btn => {
        btn.addEventListener("click", () => {
          mtpMaskTopologySeg.querySelectorAll("button").forEach(b => {
            b.classList.remove("on");
            b.setAttribute("aria-pressed", "false");
          });
          btn.classList.add("on");
          btn.setAttribute("aria-pressed", "true");
          maskTopo = btn.getAttribute("data-topo");
          renderMtpMaskTable();
        });
      });
    }

    // Verif cycle step button
    if (mtpVerifStepBtn) {
      mtpVerifStepBtn.addEventListener("click", () => {
        verifPresetIdx = (verifPresetIdx + 1) % VERIF_PRESETS.length;
        renderMtpVerifCycle();
      });
    }

    // Initial renders
    updateInspector("backbone");
    renderMtpTreeSvg();
    renderMtpMaskTable();
    renderMtpVerifCycle();
  }

  setupMtpTreeWidget();
})();

/* ============================================================
   Section 09: Data Pipeline & Synthesis Engine
   ============================================================ */
(function () {
  /* ------------------------------------------------------------
     1. Widget: 5-Stage Ingestion Funnel (#w-data-funnel)
     ------------------------------------------------------------ */
  const funnelPipe = $("funnelPipeline");
  const funnelDetailCard = $("funnelDetailCard");

  const FUNNEL_STAGES = {
    "1": {
      name: "Stage 01 · Raw Harvest & Web Crawl",
      volIn: "120+ Petabytes raw WARC / HTML / PDFs / Git Repositories",
      volOut: "45 Petabytes normalized raw documents",
      retention: "37.5% volume retention",
      heuristics: "Boilerplate stripping, HTML tag extraction, ad removal, MIME type validation, and UTF-8 normalization.",
      details: "Aggregates massive public web crawls (CommonCrawl, curated domains), ArXiv LaTeX sources, public GitHub code, and encyclopedic archives. Language classifier identifies 119 candidate languages."
    },
    "2": {
      name: "Stage 02 · Multimodal Layout-Aware OCR",
      volIn: "45 Petabytes unparsed PDFs, charts, and scanned math papers",
      volOut: "18 Petabytes structured markdown with LaTeX formulas",
      retention: "40.0% parsed text retention",
      heuristics: "Qwen2.5-VL layout segmentation, bounding box alignment, two-column reflow, and table-to-markdown conversion.",
      details: "Standard text parsers mangle complex STEM documents, tables, and mathematical proofs. A finetuned vision-language model parses multi-column layouts into clean, linear Markdown."
    },
    "3": {
      name: "Stage 03 · Quality Filtering, Dedup & Decontamination",
      volIn: "18 Petabytes structured text (~48 Trillion candidate tokens)",
      volOut: "22 Trillion high-density organic tokens",
      retention: "45.8% token retention",
      heuristics: "MinHash-LSH near-duplicate deduplication, perplexity classifier ensembles, repetition penalties, and 13-gram decontamination.",
      details: "Filters out low-entropy machine junk, SEO spam, and toxic content. 13-gram exact match scans permanently remove benchmark overlaps against GSM8K, MATH, HumanEval, and SWE-bench."
    },
    "4": {
      name: "Stage 04 · Model-in-the-Loop Verifier Synthesis",
      volIn: "22 Trillion organic foundation tokens",
      volOut: "+14 Trillion verified synthetic tokens (Total 36T)",
      retention: "+63.6% volume expansion via synthesis",
      heuristics: "Automated verification oracles: SymPy Computer Algebra System, Lean 4 provers, and Dockerized unit-test execution.",
      details: "Qwen2.5-Math and Qwen2.5-Coder synthesize missing textbook derivations, algorithmic solutions, and multi-turn agent tool traces. Only verified correct outputs enter pretraining shards."
    },
    "5": {
      name: "Stage 05 · Best-Fit Sequence Packing & Sharding",
      volIn: "36 Trillion verified clean tokens",
      volOut: "36 Trillion tokens packed into 4,096 & 262,144 shards",
      retention: "98.4% sequence bin packing fill rate",
      heuristics: "Best-fit bin packing, attention mask reset boundaries, position-ID resets, and deterministic shard shuffling.",
      details: "Eliminates padding token waste during massive distributed pretraining while strictly preventing attention leakage across document boundaries."
    }
  };

  function renderFunnelStage(stepNum) {
    if (!funnelDetailCard) return;
    const info = FUNNEL_STAGES[stepNum] || FUNNEL_STAGES["1"];

    funnelDetailCard.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px;border-bottom:1px solid var(--line);padding-bottom:6px">
        <span style="font-family:var(--display);font-size:15px;font-weight:700;color:var(--fg)">${info.name}</span>
        <span style="font-family:var(--mono);font-size:11px;color:var(--sky);font-weight:600">${info.retention}</span>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:8px;font-family:var(--mono);font-size:11px">
        <div style="background:var(--bg-2);padding:8px 12px;border-radius:6px;box-shadow:inset 0 0 0 1px var(--line)">
          <span style="color:var(--faint);text-transform:uppercase;font-size:9.5px;display:block;margin-bottom:2px">Input Volume</span>
          <span style="color:var(--fg);font-weight:600">${info.volIn}</span>
        </div>
        <div style="background:var(--bg-2);padding:8px 12px;border-radius:6px;box-shadow:inset 0 0 0 1px var(--line)">
          <span style="color:var(--faint);text-transform:uppercase;font-size:9.5px;display:block;margin-bottom:2px">Output Shard Yield</span>
          <span style="color:var(--mint);font-weight:600">${info.volOut}</span>
        </div>
      </div>
      <div style="font-size:12.5px;color:var(--muted);line-height:1.5;margin-bottom:6px">
        <b>Pipeline Strategy:</b> ${info.details}
      </div>
      <div style="font-size:11.5px;color:var(--faint);font-family:var(--mono)">
        <b>Filtering Rules:</b> ${info.heuristics}
      </div>
    `;
  }

  if (funnelPipe) {
    funnelPipe.querySelectorAll(".funnel-step").forEach(btn => {
      btn.addEventListener("click", () => {
        funnelPipe.querySelectorAll(".funnel-step").forEach(s => s.classList.remove("active"));
        btn.classList.add("active");
        renderFunnelStage(btn.dataset.step);
      });
    });
    enableTabs(funnelPipe, ".funnel-step");
  }
  renderFunnelStage("1");


  /* ------------------------------------------------------------
     2. Widget: Dynamic Corpus Mix Visualizer (#w-data-mix)
     ------------------------------------------------------------ */
  const mixStageSeg = $("mixStageSeg");
  const msbWeb = $("msbWeb");
  const msbCode = $("msbCode");
  const msbStem = $("msbStem");
  const msbMulti = $("msbMulti");
  const mixStats = $("mixStats");

  const STAGE_MIXES = {
    s1: {
      name: "Stage 1 · General Foundation Mix (30 Trillion Tokens)",
      web: 35, code: 30, stem: 20, multi: 15,
      tokens: { web: "10.5T", code: "9.0T", stem: "6.0T", multi: "4.5T" },
      desc: "Balanced distribution establishing broad language grounding across 119 natural languages, syntax mastery, and foundational world knowledge."
    },
    s2: {
      name: "Stage 2 · Reasoning & STEM Mix (+5 Trillion Tokens)",
      web: 15, code: 35, stem: 40, multi: 10,
      tokens: { web: "0.75T", code: "1.75T", stem: "2.0T", multi: "0.5T" },
      desc: "Heavily weighted towards verified synthetic mathematics (SymPy checked), competitive programming, and formal algorithmic reasoning."
    },
    s3: {
      name: "Stage 3 · Long-Context 262k Mix (~1 Trillion Tokens)",
      web: 25, code: 45, stem: 20, multi: 10,
      tokens: { web: "0.25T", code: "0.45T", stem: "0.20T", multi: "0.10T" },
      desc: "Full repository-level codebases, multi-document synthesis, and long-horizon needle-in-a-haystack tasks annealed up to 262k tokens."
    }
  };

  function renderCorpusMix(stageKey) {
    if (!mixStats) return;
    const mix = STAGE_MIXES[stageKey] || STAGE_MIXES.s1;

    if (msbWeb) msbWeb.style.width = `${mix.web}%`;
    if (msbCode) msbCode.style.width = `${mix.code}%`;
    if (msbStem) msbStem.style.width = `${mix.stem}%`;
    if (msbMulti) msbMulti.style.width = `${mix.multi}%`;

    mixStats.innerHTML = `
      <div style="font-family:var(--display);font-size:14px;font-weight:700;color:var(--fg);margin-bottom:6px">${mix.name}</div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(130px, 1fr));gap:8px;margin-bottom:8px;font-family:var(--mono);font-size:11px">
        <div style="background:var(--bg-2);padding:6px 10px;border-radius:4px">
          <span style="color:var(--sky)">Web &amp; Books:</span> <b>${mix.web}% (${mix.tokens.web})</b>
        </div>
        <div style="background:var(--bg-2);padding:6px 10px;border-radius:4px">
          <span style="color:var(--mint)">Code &amp; Repos:</span> <b>${mix.code}% (${mix.tokens.code})</b>
        </div>
        <div style="background:var(--bg-2);padding:6px 10px;border-radius:4px">
          <span style="color:var(--amber)">STEM &amp; Math:</span> <b>${mix.stem}% (${mix.tokens.stem})</b>
        </div>
        <div style="background:var(--bg-2);padding:6px 10px;border-radius:4px">
          <span style="color:var(--iris)">Multilingual:</span> <b>${mix.multi}% (${mix.tokens.multi})</b>
        </div>
      </div>
      <div style="font-size:12px;color:var(--muted);line-height:1.45">${mix.desc}</div>
    `;
  }

  if (mixStageSeg) {
    bindGroup(mixStageSeg, b => renderCorpusMix(b.dataset.stage));
  }
  renderCorpusMix("s1");


  /* ------------------------------------------------------------
     3. Widget: Sequence Packing & Mask Inspector (#w-data-packing)
     ------------------------------------------------------------ */
  const packingVisual = $("packingVisual");
  const packingModeSeg = $("packingModeSeg");
  let packingMode = "isolated";

  function renderPacking() {
    if (!packingVisual) return;

    if (packingMode === "isolated") {
      packingVisual.innerHTML = `
        <div class="pv-seq-track">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;font-family:var(--mono);font-size:11px;color:var(--muted)">
            <span>Packed 4,096-Token Training Sequence (3 Documents)</span>
            <span style="color:var(--mint);font-weight:600">Fill Rate: 99.8% · 8 Pad Tokens</span>
          </div>
          <div class="pv-track-bar">
            <div class="pv-doc d1" style="flex:1850">
              <span class="doc-lbl">Doc 1: Python AST</span>
              <span class="pos-lbl">1,850 tok · Pos 0–1,849 · Mask [1..1850]</span>
            </div>
            <div class="pv-doc d2" style="flex:1420">
              <span class="doc-lbl">Doc 2: Lean 4 Proof</span>
              <span class="pos-lbl">1,420 tok · Pos 0–1,419 · Mask Reset ✓</span>
            </div>
            <div class="pv-doc d3" style="flex:818">
              <span class="doc-lbl">Doc 3: Multilingual</span>
              <span class="pos-lbl">818 tok · Pos 0–817 · Mask Reset ✓</span>
            </div>
          </div>
        </div>
        <div class="pv-stats-grid">
          <div class="pv-stat-card">
            <span class="pv-stat-lbl">Attention FLOPs</span>
            <span class="pv-stat-val" style="color:var(--mint)">6.13M (Block-Diag: −63.5%)</span>
          </div>
          <div class="pv-stat-card">
            <span class="pv-stat-lbl">FlashAttention Varlen</span>
            <span class="pv-stat-val" style="color:var(--sky)">cu_seqlens = [0, 1850, 3270, 4088]</span>
          </div>
          <div class="pv-stat-card">
            <span class="pv-stat-lbl">RoPE Position Range</span>
            <span class="pv-stat-val" style="color:var(--fg)">Strict Pos 0 Reset at EOS</span>
          </div>
        </div>
        <div style="font-family:var(--mono);font-size:11.5px;color:var(--mint);line-height:1.5;background:rgba(29,110,107,0.06);padding:8px 12px;border-radius:6px;border-left:3px solid var(--mint)">
          ✓ <b>Correct Mask Isolation Active:</b> Attention matrix blocks cross-document attention using FlashAttention varlen paths. Position IDs reset to 0 at each EOS delimiter, preventing positional RoPE contamination across unrelated articles.
        </div>
      `;
    } else {
      packingVisual.innerHTML = `
        <div class="pv-seq-track">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;font-family:var(--mono);font-size:11px;color:var(--rose)">
            <span>Naive Concatenation (Attention Mask Ignored)</span>
            <span style="font-weight:600">Cross-Doc Contamination Bug ✗</span>
          </div>
          <div class="pv-track-bar">
            <div class="pv-doc d1 bleed" style="flex:1850">
              <span class="doc-lbl">Doc 1: Python AST</span>
              <span class="pos-lbl">1,850 tok · Pos 0–1,849</span>
            </div>
            <div class="pv-doc d2 bleed" style="flex:1420">
              <span class="doc-lbl">Doc 2: Lean 4 ⚠️</span>
              <span class="pos-lbl">1,420 tok · Attends to Doc 1</span>
            </div>
            <div class="pv-doc d3 bleed" style="flex:818">
              <span class="doc-lbl">Doc 3: Multilingual ⚠️</span>
              <span class="pos-lbl">818 tok · Attends to Doc 1 &amp; 2</span>
            </div>
          </div>
        </div>
        <div class="pv-stats-grid">
          <div class="pv-stat-card">
            <span class="pv-stat-lbl">Attention FLOPs</span>
            <span class="pv-stat-val" style="color:var(--rose)">16.78M (Full 4k: +174% Waste)</span>
          </div>
          <div class="pv-stat-card">
            <span class="pv-stat-lbl">Attention Leakage</span>
            <span class="pv-stat-val" style="color:var(--rose)">Unmasked (Cross-Doc Bleed)</span>
          </div>
          <div class="pv-stat-card">
            <span class="pv-stat-lbl">RoPE Position Range</span>
            <span class="pv-stat-val" style="color:var(--rose)">Spurious Drift Pos 0 → 4,087</span>
          </div>
        </div>
        <div style="font-family:var(--mono);font-size:11.5px;color:var(--rose);line-height:1.5;background:rgba(138,51,36,0.06);padding:8px 12px;border-radius:6px;border-left:3px solid var(--rose)">
          ✗ <b>Attention Leakage Bug:</b> Without attention mask resets, Doc 2 and Doc 3 attend directly into previous documents' tokens. Associative memory writes become corrupted with unrelated document history, and RoPE frequencies drift past the document's true length.
        </div>
      `;
    }
  }

  if (packingModeSeg) {
    bindGroup(packingModeSeg, b => { packingMode = b.dataset.mode; renderPacking(); });
  }
  renderPacking();
})();

/* ============================================================
   Section 10: Pretraining & Scaling Curriculum Engine
   ============================================================ */
(function () {
  /* ------------------------------------------------------------
     1. Interactive 3-Stage Pretraining Curriculum Tabs
     ------------------------------------------------------------ */
  const curr = $("pretrainCurriculum");
  if (curr) {
    curr.querySelectorAll(".cur-item").forEach(item => {
      item.addEventListener("click", () => {
        curr.querySelectorAll(".cur-item").forEach(c => c.classList.remove("active"));
        item.classList.add("active");
      });
    });
    enableTabs(curr, ".cur-item");
  }

  /* ------------------------------------------------------------
     2. Widget: Pretraining Loss Trajectory & Milestone Explorer (#w-pretrain-loss)
     ------------------------------------------------------------ */
  const lossSvg = $("lossSvg");
  const lossStats = $("lossStats");
  const lossTokenSlider = $("lossTokenSlider");
  const lossCurveToggle = $("lossCurveToggle");

  function renderLossTrajectory() {
    if (!lossSvg || !lossStats) return;
    const curTokens = +lossTokenSlider.value || 30;
    const trackMode = lossCurveToggle.value || "all";

    const NS = "http://www.w3.org/2000/svg";
    lossSvg.innerHTML = "";

    const W = 640, H = 260, P = { l: 48, r: 24, t: 24, b: 36 };
    const IW = W - P.l - P.r, IH = H - P.t - P.b;

    // X: 0 to 36T tokens, Y: Loss 0.8 to 3.5
    const fx = t => P.l + (t / 36) * IW;
    const fy = l => P.t + (1 - (l - 0.8) / (3.5 - 0.8)) * IH;

    // Grid lines & Axis labels
    [0, 10, 20, 30, 36].forEach(tVal => {
      const gx = fx(tVal);
      const l = document.createElementNS(NS, "line");
      l.setAttribute("class", "bp-grid");
      l.setAttribute("x1", gx); l.setAttribute("y1", P.t);
      l.setAttribute("x2", gx); l.setAttribute("y2", P.t + IH);
      lossSvg.appendChild(l);

      const txt = document.createElementNS(NS, "text");
      txt.setAttribute("class", "bp-tick");
      txt.setAttribute("x", gx); txt.setAttribute("y", H - 8);
      txt.setAttribute("text-anchor", "middle");
      txt.textContent = `${tVal}T`;
      lossSvg.appendChild(txt);
    });

    [1.0, 1.5, 2.0, 2.5, 3.0].forEach(lVal => {
      const gy = fy(lVal);
      const l = document.createElementNS(NS, "line");
      l.setAttribute("class", "bp-grid");
      l.setAttribute("x1", P.l); l.setAttribute("y1", gy);
      l.setAttribute("x2", P.l + IW); l.setAttribute("y2", gy);
      lossSvg.appendChild(l);

      const txt = document.createElementNS(NS, "text");
      txt.setAttribute("class", "bp-tick");
      txt.setAttribute("x", P.l - 8); txt.setAttribute("y", gy + 3);
      txt.setAttribute("text-anchor", "end");
      txt.textContent = lVal.toFixed(1);
      lossSvg.appendChild(txt);
    });

    // Stage boundary markers
    const s2X = fx(30);
    const s2Line = document.createElementNS(NS, "line");
    s2Line.setAttribute("x1", s2X); s2Line.setAttribute("y1", P.t);
    s2Line.setAttribute("x2", s2X); s2Line.setAttribute("y2", P.t + IH);
    s2Line.setAttribute("stroke", "rgba(184,52,31,0.3)");
    s2Line.setAttribute("stroke-dasharray", "4 3");
    lossSvg.appendChild(s2Line);

    const s2Lbl = document.createElementNS(NS, "text");
    s2Lbl.setAttribute("x", s2X + 4); s2Lbl.setAttribute("y", P.t + 14);
    s2Lbl.setAttribute("fill", PAL.amber);
    s2Lbl.setAttribute("font-size", "9.5");
    s2Lbl.setAttribute("font-family", '"IBM Plex Mono", ui-monospace, monospace');
    s2Lbl.textContent = "Stage 2 (STEM Restart)";
    lossSvg.appendChild(s2Lbl);

    // Compute Loss curves
    // 1. General language loss
    function getGenLoss(t) {
      return 1.42 + 1.65 * Math.pow(t + 0.5, -0.18);
    }
    // 2. STEM Math/Code loss
    function getStemLoss(t) {
      if (t < 30) {
        return 1.85 + 1.3 * Math.pow(t + 0.5, -0.12);
      } else {
        const dt = t - 30;
        return 2.15 * Math.exp(-dt * 0.22) + 1.15;
      }
    }
    // 3. MTP Aux Loss
    function getMtpLoss(t) {
      return 1.65 + 1.4 * Math.pow(t + 0.5, -0.15);
    }

    let dGen = "", dStem = "", dMtp = "";
    for (let t = 0; t <= 36.001; t += 0.25) {
      const gx = fx(t);
      dGen += (t === 0 ? "M" : "L") + gx + "," + fy(getGenLoss(t)) + " ";
      dStem += (t === 0 ? "M" : "L") + gx + "," + fy(getStemLoss(t)) + " ";
      dMtp += (t === 0 ? "M" : "L") + gx + "," + fy(getMtpLoss(t)) + " ";
    }

    if (trackMode === "all" || trackMode === "general") {
      const pGen = document.createElementNS(NS, "path");
      pGen.setAttribute("d", dGen); pGen.setAttribute("fill", "none");
      pGen.setAttribute("stroke", PAL.mint); pGen.setAttribute("stroke-width", "2.2");
      lossSvg.appendChild(pGen);
    }

    if (trackMode === "all" || trackMode === "stem") {
      const pStem = document.createElementNS(NS, "path");
      pStem.setAttribute("d", dStem); pStem.setAttribute("fill", "none");
      pStem.setAttribute("stroke", PAL.amber); pStem.setAttribute("stroke-width", "2.2");
      lossSvg.appendChild(pStem);
    }

    if (trackMode === "all" || trackMode === "mtp") {
      const pMtp = document.createElementNS(NS, "path");
      pMtp.setAttribute("d", dMtp); pMtp.setAttribute("fill", "none");
      pMtp.setAttribute("stroke", PAL.iris); pMtp.setAttribute("stroke-width", "1.8");
      pMtp.setAttribute("stroke-dasharray", "4 3");
      lossSvg.appendChild(pMtp);
    }

    // Scrubber Marker
    const curX = fx(curTokens);
    const sLine = document.createElementNS(NS, "line");
    sLine.setAttribute("x1", curX); sLine.setAttribute("y1", P.t);
    sLine.setAttribute("x2", curX); sLine.setAttribute("y2", P.t + IH);
    sLine.setAttribute("stroke", PAL.rose);
    sLine.setAttribute("stroke-width", "2");
    lossSvg.appendChild(sLine);

    const activeLoss = getGenLoss(curTokens);
    const activeStem = getStemLoss(curTokens);

    const sDot = document.createElementNS(NS, "circle");
    sDot.setAttribute("cx", curX); sDot.setAttribute("cy", fy(activeLoss));
    sDot.setAttribute("r", "5.5");
    sDot.setAttribute("fill", PAL.rose);
    sDot.setAttribute("stroke", PAL.ink);
    sDot.setAttribute("stroke-width", "2");
    lossSvg.appendChild(sDot);

    const stageName = curTokens < 30 ? "Stage 1 (General Foundation)" : (curTokens < 35 ? "Stage 2 (Reasoning & STEM)" : "Stage 3 (262k Long Context)");
    const pct = ((curTokens / 36) * 100).toFixed(1);

    lossStats.innerHTML = `
      <div class="rf-stat-box">
        <div class="rf-head">Curriculum Milestone @ ${curTokens.toFixed(1)}T Tokens (${pct}%)</div>
        <div class="rf-row"><span>Active Phase</span><span class="val amber" style="font-size:12px">${stageName}</span></div>
        <div class="rf-row"><span>General Cross-Entropy Loss</span><span class="val mint">${activeLoss.toFixed(3)} (PPL ~${Math.exp(activeLoss).toFixed(2)})</span></div>
        <div class="rf-row"><span>STEM / Code Loss</span><span class="val">${activeStem.toFixed(3)}</span></div>
        <div class="rf-row"><span>MTP Aux Loss</span><span class="val" style="color:var(--iris)">${getMtpLoss(curTokens).toFixed(3)}</span></div>
      </div>
      <div class="rf-stat-box" style="padding:10px 14px">
        <div style="font-family:var(--mono);font-size:10px;text-transform:uppercase;color:var(--faint);margin-bottom:4px">Milestone Insight</div>
        <div style="font-size:12px;color:var(--muted);line-height:1.45">
          ${curTokens < 30 
            ? "Phase 1 steadily compresses natural language grammar, factual world knowledge, and broad multilingual representations across 119 languages." 
            : (curTokens < 35 
              ? "Phase 2 LR restart rapidly steepens reasoning and competitive coding benchmarks without degradation to Stage 1 core representations." 
              : "Phase 3 anneals chunked WY recurrent scan kernels and YaRN RoPE scaling to lock in full 262k context length capability.")}
        </div>
      </div>
    `;
  }

  [lossTokenSlider, lossCurveToggle].forEach(el => {
    if (el) {
      el.addEventListener("input", renderLossTrajectory);
      el.addEventListener("change", renderLossTrajectory);
    }
  });
  renderLossTrajectory();
  if (lossTokenSlider?.type === "range") {
    const vt = () => lossTokenSlider.setAttribute("aria-valuetext", `${(+lossTokenSlider.value).toFixed(1)} trillion tokens`);
    lossTokenSlider.addEventListener("input", vt); vt();
  }


  /* ------------------------------------------------------------
     3. Widget: Cluster Compute & Supercomputing Budget Calculator (#w-pretrain-calc)
     ------------------------------------------------------------ */
  const calcReadout = $("calcReadout");
  const calcTokens = $("calcTokens");
  const calcGpuType = $("calcGpuType");
  const calcGpuCount = $("calcGpuCount");
  const calcMfu = $("calcMfu");
  const GPU_SPECS = {
    h100: { name: "NVIDIA H100 SXM5", tflops: 989, watts: 700 },
    b200: { name: "NVIDIA B200 NVL", tflops: 2250, watts: 1000 },
    h200: { name: "NVIDIA H200 SXM5", tflops: 989, watts: 700 },
    a100: { name: "NVIDIA A100 SXM4", tflops: 312,  watts: 400 }
  };

  function renderComputeCalc() {
    if (!calcReadout) return;
    const tokensT = +calcTokens.value || 36;
    const gpuKey = calcGpuType.value || "h100";
    const gpuCount = +calcGpuCount.value || 4096;
    const mfu = +calcMfu.value || 0.42;

    const gpu = GPU_SPECS[gpuKey];
    const totalTokens = tokensT * 1e12;
    const N = 27.58e9; // 27.58B parameters

    // Chinchilla compute FLOPs: 6 * N * D
    const totalFlops = 6 * N * totalTokens;
    const yottaFlops = totalFlops / 1e24;
    const exaFlopDays = totalFlops / (1e18 * 86400);

    // Cluster sustained compute in FLOPs/s
    const clusterFlopsPerSec = gpuCount * (gpu.tflops * 1e12) * mfu;
    const totalSeconds = totalFlops / clusterFlopsPerSec;
    const totalDays = totalSeconds / 86400;

    // Power & Energy
    const pue = 1.25;
    const clusterKw = (gpuCount * gpu.watts * pue) / 1000;
    const clusterMw = clusterKw / 1000;
    const totalMWh = (clusterMw * totalDays * 24);

    calcReadout.innerHTML = `
      <div class="calc-stat-grid">
        <div class="cs-card">
          <div class="cs-num">${yottaFlops.toFixed(2)} YFLOPs</div>
          <div class="cs-lbl">Total Compute Volume</div>
        </div>
        <div class="cs-card">
          <div class="cs-num">${exaFlopDays.toFixed(0)} EFD</div>
          <div class="cs-lbl">ExaFLOP-Days</div>
        </div>
        <div class="cs-card">
          <div class="cs-num" style="color:var(--mint)">${totalDays.toFixed(1)} Days</div>
          <div class="cs-lbl">Wall-Clock Duration</div>
        </div>
        <div class="cs-card">
          <div class="cs-num" style="color:var(--iris)">${totalMWh.toLocaleString(undefined, {maximumFractionDigits: 0})} MWh</div>
          <div class="cs-lbl">Energy (PUE 1.25)</div>
        </div>
      </div>
      <div style="font-family:var(--mono);font-size:11px;color:var(--muted);line-height:1.5;border-top:1px solid var(--line);padding-top:8px">
        Cluster throughput: <b>${(clusterFlopsPerSec / 1e18).toFixed(1)} Sustained ExaFLOPs/s</b> across ${gpuCount.toLocaleString()} × ${gpu.name} @ ${(mfu * 100).toFixed(0)}% MFU · Power Draw: <b>${clusterMw.toFixed(2)} MW</b>.
      </div>
    `;
  }

  [calcTokens, calcGpuType, calcGpuCount, calcMfu].forEach(el => {
    if (el) {
      el.addEventListener("change", renderComputeCalc);
      el.addEventListener("input", renderComputeCalc);
    }
  });
  renderComputeCalc();


  /* ------------------------------------------------------------
     4. Widget: Optimizer Dynamics & Beta2 Inspector (#w-optim-dyn)
     ------------------------------------------------------------ */
  const optimSvg = $("optimSvg");
  const optimStats = $("optimStats");
  const optimBetaSeg = $("optimBetaSeg");
  let curBeta2 = 0.95;

  function renderOptimDynamics() {
    if (!optimSvg || !optimStats) return;

    const NS = "http://www.w3.org/2000/svg";
    optimSvg.innerHTML = "";

    const W = 560, H = 220, P = { l: 48, r: 24, t: 20, b: 36 };
    const IW = W - P.l - P.r, IH = H - P.t - P.b;

    // Steps 0 to 100
    const fx = step => P.l + (step / 100) * IW;
    const fy = scale => P.t + (1 - scale) * IH;

    // Grid lines
    [0, 25, 50, 75, 100].forEach(sVal => {
      const gx = fx(sVal);
      const l = document.createElementNS(NS, "line");
      l.setAttribute("class", "bp-grid");
      l.setAttribute("x1", gx); l.setAttribute("y1", P.t);
      l.setAttribute("x2", gx); l.setAttribute("y2", P.t + IH);
      optimSvg.appendChild(l);

      const txt = document.createElementNS(NS, "text");
      txt.setAttribute("class", "bp-tick");
      txt.setAttribute("x", gx); txt.setAttribute("y", H - 8);
      txt.setAttribute("text-anchor", "middle");
      txt.textContent = `Step ${sVal}`;
      optimSvg.appendChild(txt);
    });

    [0.2, 0.4, 0.6, 0.8, 1.0].forEach(scaleVal => {
      const gy = fy(scaleVal);
      const l = document.createElementNS(NS, "line");
      l.setAttribute("class", "bp-grid");
      l.setAttribute("x1", P.l); l.setAttribute("y1", gy);
      l.setAttribute("x2", P.l + IW); l.setAttribute("y2", gy);
      optimSvg.appendChild(l);

      const txt = document.createElementNS(NS, "text");
      txt.setAttribute("class", "bp-tick");
      txt.setAttribute("x", P.l - 8); txt.setAttribute("y", gy + 3);
      txt.setAttribute("text-anchor", "end");
      txt.textContent = `${(scaleVal * 100).toFixed(0)}%`;
      optimSvg.appendChild(txt);
    });

    // Simulate 10x gradient spike at step 30
    let v_t = 1.0;
    let dPath = "";
    const halfLifeSteps = curBeta2 === 0.95 ? 14 : 693;

    for (let step = 0; step <= 100; step++) {
      let g_t = 1.0;
      if (step === 30) g_t = 10.0; // Gradient shock
      v_t = curBeta2 * v_t + (1 - curBeta2) * (g_t * g_t);
      const effectiveStepScale = Math.min(1.0, 1.0 / Math.sqrt(v_t));

      const gx = fx(step);
      const gy = fy(effectiveStepScale);
      dPath += (step === 0 ? "M" : "L") + gx + "," + gy + " ";
    }

    const pathEl = document.createElementNS(NS, "path");
    pathEl.setAttribute("d", dPath);
    pathEl.setAttribute("fill", "none");
    pathEl.setAttribute("stroke", curBeta2 === 0.95 ? PAL.mint : PAL.rose);
    pathEl.setAttribute("stroke-width", "2.5");
    optimSvg.appendChild(pathEl);

    // Shock annotation
    const shockX = fx(30);
    const sLine = document.createElementNS(NS, "line");
    sLine.setAttribute("x1", shockX); sLine.setAttribute("y1", P.t);
    sLine.setAttribute("x2", shockX); sLine.setAttribute("y2", P.t + IH);
    sLine.setAttribute("stroke", "rgba(138,51,36,0.4)");
    sLine.setAttribute("stroke-dasharray", "3 3");
    optimSvg.appendChild(sLine);

    optimStats.innerHTML = `
      <div class="rf-stat-box">
        <div class="rf-head">Optimizer Second-Moment Memory</div>
        <div class="rf-row"><span>Selected β₂ Parameter</span><span class="val amber">${curBeta2}</span></div>
        <div class="rf-row"><span>Variance Half-Life</span><span class="val ${curBeta2 === 0.95 ? "mint" : "rose"}">${halfLifeSteps} steps</span></div>
        <div class="rf-row"><span>Recovery from Gradient Shock</span><span class="val">${curBeta2 === 0.95 ? "Rapid (~14 steps) ✓" : "Severely Stalled (~693 steps) ✗"}</span></div>
      </div>
      <div style="font-size:12px;color:var(--muted);line-height:1.45;padding:4px 2px">
        ${curBeta2 === 0.95 
          ? "✓ Tuned β₂ = 0.95 flushes corrupted second-moment variance quickly, preventing stale damping on DeltaNet recurrent state updates." 
          : "⚠️ Standard β₂ = 0.999 keeps effective learning rate suppressed long after a transient gradient outlier, stalling hybrid state convergence."}
      </div>
    `;
  }

  if (optimBetaSeg) {
    bindGroup(optimBetaSeg, b => { curBeta2 = +b.dataset.beta; renderOptimDynamics(); });
  }
  renderOptimDynamics();
})();

/* ============================================================
   Section 11: Post-Training & Reasoning Alignment Engine
   ============================================================ */
(function () {
  /* ------------------------------------------------------------
     1. Interactive 4-Stage Post-Training Pipeline Tabs
     ------------------------------------------------------------ */
  const pipe = $("postTrainPipe");
  if (pipe) {
    pipe.querySelectorAll(".stage").forEach(st => {
      st.addEventListener("click", () => {
        pipe.querySelectorAll(".stage").forEach(s => s.classList.remove("active"));
        st.classList.add("active");
      });
    });
  }

  /* ------------------------------------------------------------
     2. Widget: Live GRPO Rollout & Policy Gradient Simulator (#w-grpo-sim)
     ------------------------------------------------------------ */
  const grpoRolloutsList = $("grpoRolloutsList");
  const grpoGroupStats = $("grpoGroupStats");
  const grpoPromptSel = $("grpoPromptSel");
  const grpoGroupSize = $("grpoGroupSize");
  const grpoTemp = $("grpoTemp");
  const grpoKlBeta = $("grpoKlBeta");

  const PROMPT_BANKS = {
    math: {
      title: "Math: Modular Inverse in Diophantine Equation",
      passRollouts: [
        "Extended Euclidean algorithm: 43 = 2(17) + 9, 17 = 1(9) + 8, 9 = 1(8) + 1. Reversing: 1 = 9 - (17 - 9) = 2(43 - 2(17)) - 17 = 2(43) - 5(17). Thus 17⁻¹ ≡ -5 ≡ 38 (mod 43). x ≡ 5 × 38 = 190 ≡ 18 (mod 43). Verification: 17(18) = 306 = 7(43) + 5. ✓",
        "Direct modular search: x = (5 + 43k)/17. For k=7: 5 + 301 = 306, 306 / 17 = 18 exactly. Minimal positive integer x = 18. ✓",
        "Fermat's Little Theorem: 17⁴¹ ≡ 1 (mod 43). Compute 17⁴¹ * 5 mod 43 via binary exponentiation... yields x = 18. ✓",
        "Linear congruence decomposition: 17x - 43y = 5. Bezout identity yields particular solution (18, 7). General solution x = 18 + 43n. Minimal positive x = 18. ✓"
      ],
      failRollouts: [
        "Arithmetic slip: 17⁻¹ computed as 28 instead of 38. x ≡ 5 × 28 = 140 ≡ 11 (mod 43). Verifier fails: 17(11) = 187 ≠ 5 mod 43. ✗",
        "Modulus confusion: Treated 43 as composite and factored into primes. Outputted x = 24. ✗",
        "Loop timeout / endless reasoning loop: Re-evaluating 17k without terminating. ✗",
        "Formatting violation: Did not box final integer answer. ✗"
      ]
    },
    code: {
      title: "Code: Memory-Leak Fix in Concurrency Thread-Pool",
      passRollouts: [
        "Identified unclosed channel in worker teardown loop. Added defer close(worker.jobs) and synced with sync.WaitGroup. All race tests pass with 0 leak. ✓",
        "Replaced naked goroutine spawns with bounded worker pool semaphore. Context cancellation cleanly drains pending tasks. Unit test assertions pass. ✓",
        "Fixed circular reference in task closure holding worker pointers in heap. Garbage collector frees buffers immediately. Verified with pprof. ✓",
        "Applied atomic pointer swap during pool resizing. Eliminates mutex contention deadlock under 10k req/s load test. ✓"
      ],
      failRollouts: [
        "Introduced mutex deadlock: Acquired lock inside defer before channel receive. Thread pool freezes on shutdown. ✗",
        "Memory leak worsened: Added buffer without bounds check, causing OOM panic under burst traffic. ✗",
        "Syntax error: Missing return type on closure signature. Compilation failed. ✗",
        "Race condition: Read worker status without atomic load. Data race detector triggered. ✗"
      ]
    },
    agent: {
      title: "Agent: 3-Step Tool-Calling with Retry Fallback",
      passRollouts: [
        "<think>API endpoint 503 returned. Need to back off and query secondary mirror.</think> <call:fetch_mirror(url='api.v2')/> -> status:200. Extracted schema and validated checksum. ✓",
        "<think>JSON response missing 'id' key. Inspecting raw payload...</think> Parsed nested payload under 'data.item.id'. Fallback succeeded. ✓",
        "<think>Rate limit 429 encountered.</think> Executed exponential jitter sleep (150ms). Retried and parsed response. ✓",
        "<think>Schema format mismatch.</think> Converted timestamp from ISO-8601 to epoch milliseconds. Assertion pass. ✓"
      ],
      failRollouts: [
        "<think>API 503 returned.</think> Retried in tight loop without delay 10 times, exhausting retry budget. ✗",
        "Hallucinated non-existent tool parameter 'force_override=True'. Tool call crashed with ValidationError. ✗",
        "Leaked internal scratchpad to user without producing structured JSON output. ✗",
        "Ignored error return code and hallucinated mock data. ✗"
      ]
    }
  };

  function renderGrpo() {
    if (!grpoRolloutsList || !grpoGroupStats) return;
    const taskKey = grpoPromptSel.value || "math";
    const G = +grpoGroupSize.value || 8;
    const temp = +grpoTemp.value || 0.7;
    const beta = +grpoKlBeta.value || 0.04;

    const bank = PROMPT_BANKS[taskKey];
    const passProb = Math.max(0.2, Math.min(0.85, 0.75 - (temp - 0.7) * 0.4));

    const rollouts = [];
    for (let i = 0; i < G; i++) {
      const isPass = Math.random() < passProb;
      const textPool = isPass ? bank.passRollouts : bank.failRollouts;
      const text = textPool[i % textPool.length];
      const r = isPass ? 1.0 : (Math.random() < 0.15 ? 0.2 : 0.0);
      rollouts.push({ id: `y_${i + 1}`, text, r, isPass });
    }

    const meanR = rollouts.reduce((acc, r) => acc + r.r, 0) / G;
    const varR = rollouts.reduce((acc, r) => acc + Math.pow(r.r - meanR, 2), 0) / G;
    const stdR = Math.sqrt(varR);

    // Compute standardized advantages
    rollouts.forEach(r => {
      r.adv = (r.r - meanR) / (stdR + 1e-5);
    });

    const estKl = 0.025 * temp * temp;
    const lossGrpo = -(rollouts.reduce((acc, r) => acc + Math.min(r.adv, 1.2 * r.adv), 0) / G) + beta * estKl;

    grpoGroupStats.innerHTML = `
      Group Mean μ: <b>${meanR.toFixed(2)}</b> · Std σ: <b>${stdR.toFixed(2)}</b> · KL: <b>${estKl.toFixed(3)}</b> · J_GRPO: <b style="color:var(--mint)">${lossGrpo.toFixed(3)}</b>
    `;

    grpoRolloutsList.innerHTML = rollouts.map(r => {
      const isPos = r.adv >= 0;
      return `
        <div class="grpo-card ${isPos ? "pos" : "neg"}">
          <div class="grpo-id">${r.id}</div>
          <div class="grpo-text">${r.text}</div>
          <div class="grpo-meta">
            <span class="grpo-reward-badge ${r.isPass ? "pass" : "fail"}">Reward r = ${r.r.toFixed(2)}</span>
            <span class="grpo-adv ${isPos ? "pos" : "neg"}">Adv A = ${isPos ? "+" : ""}${r.adv.toFixed(2)} ${isPos ? "▲ Boost" : "▼ Suppress"}</span>
          </div>
        </div>
      `;
    }).join("");
  }

  [grpoPromptSel, grpoGroupSize, grpoTemp, grpoKlBeta].forEach(sel => {
    if (sel) {
      sel.addEventListener("change", renderGrpo);
      sel.addEventListener("input", renderGrpo);
    }
  });
  renderGrpo();


  /* ------------------------------------------------------------
     3. Widget: Upgraded Thinking-Budget Explorer (#w-effort)
     ------------------------------------------------------------ */
  const effortSvg = $("effortSvg");
  const effortOut = $("effortOut");
  const effortSel = $("effortSel");
  const effortDiff = $("effortDiff");
  const effortPreserve = $("effortPreserve");

  function renderEffort() {
    if (!effortSvg || !effortOut) return;
    const effort = +effortSel.value;
    const diff = +effortDiff.value;
    const preserve = effortPreserve.value === "true";
    if (effortSel.type === "range") effortSel.setAttribute("aria-valuetext", `effort level ${effort} of 3`);
    if (effortDiff.type === "range") effortDiff.setAttribute("aria-valuetext", `${diff}% task difficulty`);

    const NS = "http://www.w3.org/2000/svg";
    effortSvg.innerHTML = "";

    const W = 720, H = 260, ox = 56, oy = 215, pw = 620, ph = 175;
    const X = f => ox + (f / 3) * pw;
    const Ya = a => oy - (a / 100) * ph;
    const Yt = t => oy - (t / 16384) * ph;

    // Grid lines & Axes
    const l1 = document.createElementNS(NS, "line");
    l1.setAttribute("x1", ox); l1.setAttribute("y1", oy);
    l1.setAttribute("x2", ox + pw); l1.setAttribute("y2", oy);
    l1.setAttribute("stroke", "rgba(26,22,18,.15)");
    effortSvg.appendChild(l1);

    const l2 = document.createElementNS(NS, "line");
    l2.setAttribute("x1", ox); l2.setAttribute("y1", oy - ph);
    l2.setAttribute("x2", ox); l2.setAttribute("y2", oy);
    l2.setAttribute("stroke", "rgba(26,22,18,.15)");
    effortSvg.appendChild(l2);

    [[0, "off (0 tok)"], [1, "low (1k)"], [2, "med (4k)"], [3, "xhigh (16k)"]].forEach(([f, s]) => {
      const t = document.createElementNS(NS, "text");
      t.setAttribute("x", X(f)); t.setAttribute("y", oy + 18);
      t.setAttribute("text-anchor", "middle");
      t.setAttribute("fill", PAL.muted);
      t.setAttribute("font-size", "10");
      t.setAttribute("font-family", '"IBM Plex Mono", ui-monospace, monospace');
      t.textContent = s;
      effortSvg.appendChild(t);
    });

    function getAcc(eff, d) {
      const base = 25 + 0.65 * d;
      const gain = eff === 0 ? 0.35 : (eff === 1 ? 0.72 : (eff === 2 ? 0.91 : 0.98));
      return Math.min(99.4, base * gain);
    }
    function getToks(eff) {
      return eff === 0 ? 0 : (eff === 1 ? 1024 : (eff === 2 ? 4096 : 16384));
    }

    // Drawn token curve interpolates the same budgets getToks reports,
    // so the plotted value at each effort level matches the readout exactly.
    const TOK_BUDGETS = [0, 1024, 4096, 16384];
    const simToks = f => {
      const lo = Math.min(2, Math.floor(f));
      return TOK_BUDGETS[lo] + (TOK_BUDGETS[lo + 1] - TOK_BUDGETS[lo]) * (f - lo);
    };

    let da = "", dt = "";
    for (let f = 0; f <= 3.001; f += 0.05) {
      const simulatedAcc = getAcc(f, diff);
      da += (f === 0 ? "M" : "L") + X(f) + "," + Ya(simulatedAcc) + " ";
      dt += (f === 0 ? "M" : "L") + X(f) + "," + Yt(Math.min(16384, simToks(f))) + " ";
    }

    const accP = document.createElementNS(NS, "path");
    accP.setAttribute("d", da);
    accP.setAttribute("fill", "none");
    accP.setAttribute("stroke", PAL.mint);
    accP.setAttribute("stroke-width", "2.5");
    effortSvg.appendChild(accP);

    const tokP = document.createElementNS(NS, "path");
    tokP.setAttribute("d", dt);
    tokP.setAttribute("fill", "none");
    tokP.setAttribute("stroke", PAL.iris);
    tokP.setAttribute("stroke-width", "2");
    tokP.setAttribute("stroke-dasharray", "5 4");
    effortSvg.appendChild(tokP);

    const curAcc = getAcc(effort, diff);
    const curToks = getToks(effort);

    const dot = document.createElementNS(NS, "circle");
    dot.setAttribute("cx", X(effort)); dot.setAttribute("cy", Ya(curAcc));
    dot.setAttribute("r", "6");
    dot.setAttribute("fill", PAL.amber);
    dot.setAttribute("stroke", PAL.ink);
    dot.setAttribute("stroke-width", "2");
    effortSvg.appendChild(dot);

    const tokDot = document.createElementNS(NS, "circle");
    tokDot.setAttribute("cx", X(effort)); tokDot.setAttribute("cy", Yt(curToks));
    tokDot.setAttribute("r", "5");
    tokDot.setAttribute("fill", PAL.iris);
    effortSvg.appendChild(tokDot);

    const recoveryRate = preserve ? Math.min(96, curAcc * 1.08) : Math.max(40, curAcc * 0.82);

    effortOut.innerHTML = `
      <div class="row"><span>Expected Benchmark Pass@1 @ Difficulty ${diff}%</span><span class="v m" style="font-size:14px;font-weight:700">${curAcc.toFixed(1)}%</span></div>
      <div class="row"><span>Deliberation Thinking Tokens Allocated</span><span class="v i">${curToks.toLocaleString()} tokens</span></div>
      <div class="row"><span>Agent Multi-Turn Self-Correction Probability (preserve_thinking=${preserve})</span><span class="v a">${recoveryRate.toFixed(1)}%</span></div>
      <div class="row"><span>Marginal Accuracy Left on Table vs xhigh</span><span class="v" style="color:var(--faint)">${(getAcc(3, diff) - curAcc).toFixed(1)} pts</span></div>
      <div style="color:var(--faint);font-size:11.5px;margin-top:8px;line-height:1.45">
        ${effort === 0 
          ? "Direct instant mode active: Zero latency overhead, best for chat & simple retrieval." 
          : `Thinking mode active: Subgoal decomposition allocates ~${curToks.toLocaleString()} tokens of test-time compute.`}
      </div>
    `;
  }

  [effortSel, effortDiff, effortPreserve].forEach(el => {
    if (el) {
      el.addEventListener("input", renderEffort);
      el.addEventListener("change", renderEffort);
    }
  });
  renderEffort();


  /* ------------------------------------------------------------
     4. Widget: Chain-of-Thought & Agentic Trace Visualizer (#w-cot-trace)
     ------------------------------------------------------------ */
  const cotTraceViewer = $("cotTraceViewer");
  const traceScenarioSeg = $("traceScenarioSeg");
  let curScenario = "math_backtrack";

  const TRACES = {
    math_backtrack: [
      {
        role: "user",
        text: "Find all real solutions to: $\\sqrt{x + 3} - \\sqrt{x - 2} = 1$."
      },
      {
        role: "assistant",
        think: `1. Domain check: x + 3 >= 0 => x >= -3, and x - 2 >= 0 => x >= 2. Intersection domain is x in [2, inf).
2. Isolate radicals: sqrt(x + 3) = 1 + sqrt(x - 2).
3. Square both sides: x + 3 = 1 + 2*sqrt(x - 2) + (x - 2).
4. Simplify algebra: x + 3 = x - 1 + 2*sqrt(x - 2) => 4 = 2*sqrt(x - 2) => sqrt(x - 2) = 2.
5. Wait, let me double check sign: 3 - (-1) = 4, correct!
6. Square again: x - 2 = 4 => x = 6.
7. Verification test: sqrt(6 + 3) - sqrt(6 - 2) = sqrt(9) - sqrt(4) = 3 - 2 = 1. Matches right hand side exactly!`,
        output: "The only real solution is <strong>x = 6</strong>."
      }
    ],
    agent_swe: [
      {
        role: "user",
        text: "Fix flaky test `test_websocket_heartbeat_timeout` failing intermittently on CI."
      },
      {
        role: "assistant",
        think: `The issue reports a race condition where the heartbeat ticker fires before the mock client socket registers the handshake.
Let's inspect the test file and connection teardown.`,
        toolCall: "run_command(CommandLine='pytest tests/test_ws.py -k test_websocket_heartbeat_timeout --count=5')",
        toolObs: "FAILED tests/test_ws.py::test_websocket_heartbeat_timeout (2/5 failed: AssertionError: heartbeat not received within 50ms)",
        thinkCont: `Root cause confirmed: The test uses a hardcoded 50ms sleep instead of waiting for the connection state condition.
Let's modify the test to use an event-driven 'wait_for_condition(timeout=2.0)'.`,
        toolCall2: "replace_file_content(TargetFile='tests/test_ws.py', Instruction='Replace fixed sleep with event waiter')",
        toolObs2: "File edited successfully.",
        output: "Fixed the flaky test by replacing the brittle 50ms sleep with an explicit condition waiter on `client.is_connected`. All 5 verification test runs now pass reliably."
      }
    ],
    direct_bypass: [
      {
        role: "user",
        text: "What is the capital of Australia?"
      },
      {
        role: "assistant",
        output: "The capital of Australia is **Canberra**."
      }
    ]
  };

  function renderCotTrace() {
    if (!cotTraceViewer) return;
    const steps = TRACES[curScenario] || TRACES.math_backtrack;

    cotTraceViewer.innerHTML = steps.map(step => {
      if (step.role === "user") {
        return `
          <div class="cot-step">
            <span class="cot-role">User</span>
            <div class="cot-bubble" style="background:var(--bg-1)">${step.text}</div>
          </div>
        `;
      } else {
        let innerHtml = "";
        if (step.think) {
          innerHtml += `<div class="cot-think-block"><b>&lt;think&gt;</b><br>${step.think.replace(/\n/g, "<br>")}</div>`;
        }
        if (step.toolCall) {
          innerHtml += `<div class="cot-tool-block"><b>Tool Call:</b> <code>${step.toolCall}</code></div>`;
        }
        if (step.toolObs) {
          innerHtml += `<div class="cot-obs-block"><b>Tool Output:</b> ${step.toolObs}</div>`;
        }
        if (step.thinkCont) {
          innerHtml += `<div class="cot-think-block"><b>&lt;think&gt; (Self-Reflection)</b><br>${step.thinkCont.replace(/\n/g, "<br>")}</div>`;
        }
        if (step.toolCall2) {
          innerHtml += `<div class="cot-tool-block"><b>Tool Call:</b> <code>${step.toolCall2}</code></div>`;
        }
        if (step.toolObs2) {
          innerHtml += `<div class="cot-obs-block"><b>Tool Output:</b> ${step.toolObs2}</div>`;
        }
        if (step.output) {
          innerHtml += `<div style="margin-top:8px">${step.output}</div>`;
        }
        return `
          <div class="cot-step">
            <span class="cot-role" style="color:var(--iris)">Qwen 3.8-27B Assistant</span>
            <div class="cot-bubble">${innerHtml}</div>
          </div>
        `;
      }
    }).join("");
    katexRender(cotTraceViewer);                                  // user turns carry inline math
  }

  if (traceScenarioSeg) {
    bindGroup(traceScenarioSeg, b => { curScenario = b.dataset.scen; renderCotTrace(); });
  }
  renderCotTrace();
})();



/* ============================================================
   W10, W11, W12 — Section 12: Inference & Optimizations Engine
   ============================================================ */
(function () {
  /* ------------------------------------------------------------
     1. Widget: Upgraded VRAM Planner (#w-quant)
     ------------------------------------------------------------ */
  const quantOut = $("quantOut");
  const vsbWeights = $("vsbWeights");
  const vsbKv = $("vsbKv");
  const vsbDeltanet = $("vsbDeltanet");
  const vsbOverhead = $("vsbOverhead");
  const vramTotalLabel = $("vramTotalLabel");

  function renderQuant() {
    if (!quantOut) return;
    const weightsGB = +$("quantPrec").value;
    const T = +$("quantCtx").value;
    const batch = Math.max(1, +$("quantBatch").value || 1);
    const budget = +$("quantGpu").value;
    const kvB = +$("quantKv").value;

    // KV: 16 GQA layers · 2 (K & V) · 4 heads · 256 dim · kvB bytes/token/layer
    const kvBytes = 2 * 4 * 256 * 16 * kvB * T * batch;
    const kvGB = kvBytes / 1e9;

    // DeltaNet: 48 layers · 48 heads · 128 · 128 · 2 bytes (BF16 state) · batch
    const dnStateBytes = 48 * 48 * 128 * 128 * 2 * batch;
    const dnStateGB = dnStateBytes / 1e9;

    // Runtime Activation & CUDA context overhead
    const overheadGB = 1.5 + (0.1 * batch);
    const totalGB = weightsGB + kvGB + dnStateGB + overheadGB;
    const fits = totalGB <= budget;

    // Update stacked visual memory bar
    const maxBar = Math.max(budget, totalGB);
    if (vsbWeights) vsbWeights.style.width = `${(weightsGB / maxBar) * 100}%`;
    if (vsbKv) vsbKv.style.width = `${(kvGB / maxBar) * 100}%`;
    if (vsbDeltanet) vsbDeltanet.style.width = `${Math.max(1.5, (dnStateGB / maxBar) * 100)}%`;
    if (vsbOverhead) vsbOverhead.style.width = `${(overheadGB / maxBar) * 100}%`;

    if (vramTotalLabel) {
      vramTotalLabel.innerHTML = `<span style="color:${fits ? "var(--mint)" : "var(--rose)"}">${totalGB.toFixed(2)} GB</span> / ${budget} GB`;
    }

    // Precision name derives from the CFG.total footprints (BF16 55.2 / FP8 27.6 / NVFP4 13.8),
    // not from hardcoded option strings — stays correct if the weights total ever moves.
    const precTable = [
      [CFG.total * 2 / 1e9, "BF16"],
      [CFG.total / 1e9, "FP8"],
      [CFG.total / 2e9, "NVFP4"],
      [16, "Q4_K_M"]
    ];
    const precHit = precTable.find(([gb]) => Math.abs(gb - weightsGB) < 0.75);
    const precName = precHit ? precHit[1] : "Custom";
    const kvName = { "2": "BF16", "1": "FP8", "0.5": "NVFP4" }[String(kvB)] || "Custom";

    quantOut.innerHTML = `
      <div class="row"><span>1. Model Weights (${precName} Quantized)</span><span class="v">${weightsGB.toFixed(1)} GB</span></div>
      <div class="row"><span>2. GQA KV-Cache (16 layers × ${batch} seq @ ${T.toLocaleString()} tok in ${kvName})</span><span class="v a">${kvGB < 1 ? (kvGB * 1000).toFixed(0) + " MB" : kvGB.toFixed(2) + " GB"}</span></div>
      <div class="row"><span>3. DeltaNet Recurrent States (48 layers, fixed matrix)</span><span class="v m">${(dnStateGB * 1000).toFixed(1)} MB</span></div>
      <div class="row"><span>4. CUDA Driver &amp; Runtime Activation Buffers</span><span class="v">${overheadGB.toFixed(2)} GB</span></div>
      <div class="row" style="border-top:1px solid var(--line-strong);margin-top:6px;padding-top:10px">
        <span><b>Total Serving Footprint vs ${budget} GB Target GPU</b></span>
        <span class="big" style="color:${fits ? "var(--mint)" : "var(--rose)"}">
          ${totalGB.toFixed(2)} GB ${fits ? `· Fits comfortably (${(budget - totalGB).toFixed(1)} GB headroom ✓)` : `· Over budget by ${(totalGB - budget).toFixed(1)} GB ✗`}
        </span>
      </div>
      <div style="color:var(--faint);font-size:11.5px;margin-top:8px;line-height:1.45">
        ${fits 
          ? "✓ Configuration verified. High throughput serving ready with zero out-of-memory hazard." 
          : "⚠️ Remedy overflow: (1) Switch KV Cache dtype from BF16 to FP8/NVFP4, (2) Step down weights precision to NVFP4, or (3) Lower concurrency."}
      </div>
    `;
  }

  ["quantPrec", "quantCtx", "quantBatch", "quantGpu", "quantKv"].forEach(id => {
    const el = $(id);
    if (el) {
      el.addEventListener("input", renderQuant);
      el.addEventListener("change", renderQuant);
    }
  });
  renderQuant();


  /* ------------------------------------------------------------
     2. Widget: Hardware Roofline & Tok/s Simulator (#w-roofline)
     ------------------------------------------------------------ */
  const rfSvg = $("rooflineSvg");
  const rfStats = $("rooflineStats");
  const rfHwSelect = $("rfHardware");
  const rfQuantSelect = $("rfQuant");
  const rfMtpSelect = $("rfMtp");

  // Regenerate β option labels from §08's acceptance formula so the dropdown
  // always agrees with the simulator (labels ship as static "~N.N tok/step").
  rfMtpSelect?.querySelectorAll("option").forEach(o => {
    const b = parseFloat(o.value);
    if (Number.isFinite(b)) o.textContent = o.textContent.replace(/~\d+(\.\d+)?/, `~${mtpEtoks(b, 2).toFixed(2)}`);
  });

  const HW_PROFILES = {
    "5090":  { name: "NVIDIA RTX 5090", vram: 32, bw: 1792, tflops: 1800, bus: "GDDR7 512-bit" },
    "4090":  { name: "NVIDIA RTX 4090", vram: 24, bw: 1008, tflops: 660,  bus: "GDDR6X 384-bit" },
    "m4max": { name: "Apple M4 Max",    vram: 128, bw: 546,  tflops: 120,  bus: "Unified LPDDR5X" },
    "a6000": { name: "RTX 6000 Ada",   vram: 48, bw: 960,   tflops: 750,  bus: "GDDR6 384-bit" },
    "h100":  { name: "NVIDIA H100 SXM", vram: 80, bw: 3350,  tflops: 2000, bus: "HBM3 5120-bit" },
    "b200":  { name: "NVIDIA B200 NVL", vram: 192, bw: 8000, tflops: 4500, bus: "HBM3e 8192-bit" }
  };

  function renderRoofline() {
    if (!rfSvg || !rfStats) return;
    const hwKey = rfHwSelect.value || "5090";
    const weightFootprintGB = +rfQuantSelect.value || 13.8;   // NVFP4 footprint from CFG.total
    const mtpBeta = +rfMtpSelect.value || 0.8;

    const hw = HW_PROFILES[hwKey];
    const baselineTokS = hw.bw / weightFootprintGB;
    // §08's acceptance convention: net speedup S ≈ E[tokens/pass] / (1 + γ_verify)
    const effectiveTokS = baselineTokS * mtpSpeedup(mtpBeta, 2);
    const latencyPerTokMs = 1000 / effectiveTokS;
    const ttftShortMs = (weightFootprintGB / hw.bw) * 1000 + 12;

    // Operational Intensity for single batch decode (2 FLOP × params in B / GB)
    const decodeIntensity = (2 * 27.58) / weightFootprintGB; // ~3.56 FLOP/Byte at NVFP4 footprint

    const NS = "http://www.w3.org/2000/svg";
    rfSvg.innerHTML = "";

    const W = 560, H = 260, P = { l: 48, r: 24, t: 24, b: 40 };
    const IW = W - P.l - P.r, IH = H - P.t - P.b;

    // Log-log scales: X = Operational Intensity (0.1 to 1000), Y = Performance (1 to 5000 TFLOPs)
    const minX = Math.log10(0.2), maxX = Math.log10(1000);
    const minY = Math.log10(1), maxY = Math.log10(10000);

    const fx = v => P.l + ((Math.log10(Math.max(0.2, v)) - minX) / (maxX - minX)) * IW;
    const fy = v => P.t + (1 - (Math.log10(Math.max(1, v)) - minY) / (maxY - minY)) * IH;

    // Grid lines
    [1, 10, 100, 1000].forEach(xVal => {
      const gx = fx(xVal);
      const l = document.createElementNS(NS, "line");
      l.setAttribute("class", "bp-grid");
      l.setAttribute("x1", gx); l.setAttribute("y1", fy(1));
      l.setAttribute("x2", gx); l.setAttribute("y2", fy(10000));
      rfSvg.appendChild(l);

      const t = document.createElementNS(NS, "text");
      t.setAttribute("class", "bp-tick");
      t.setAttribute("x", gx); t.setAttribute("y", fy(1) + 16);
      t.setAttribute("text-anchor", "middle");
      t.textContent = `${xVal}`;
      rfSvg.appendChild(t);
    });

    [10, 100, 1000, 5000].forEach(yVal => {
      const gy = fy(yVal);
      const l = document.createElementNS(NS, "line");
      l.setAttribute("class", "bp-grid");
      l.setAttribute("x1", fx(0.2)); l.setAttribute("y1", gy);
      l.setAttribute("x2", fx(1000)); l.setAttribute("y2", gy);
      rfSvg.appendChild(l);

      const t = document.createElementNS(NS, "text");
      t.setAttribute("class", "bp-tick");
      t.setAttribute("x", P.l - 8); t.setAttribute("y", gy + 3);
      t.setAttribute("text-anchor", "end");
      t.textContent = `${yVal}T`;
      rfSvg.appendChild(t);
    });

    // Roofline ceiling line (Diagonal BW limit -> Horizontal Compute Peak)
    // Knee = arithmetic intensity where ai·BW/1000 reaches the compute peak
    const kneePoint = hw.tflops * 1000 / hw.bw;
    const kneeX = fx(kneePoint);
    const kneeY = fy(hw.tflops);

    const roofPath = document.createElementNS(NS, "path");
    const dStr = `M ${fx(0.2)},${fy(0.2 * hw.bw / 1000)} L ${kneeX},${kneeY} L ${fx(1000)},${kneeY}`;
    roofPath.setAttribute("d", dStr);
    roofPath.setAttribute("fill", "none");
    roofPath.setAttribute("stroke", PAL.amber);
    roofPath.setAttribute("stroke-width", "2.5");
    rfSvg.appendChild(roofPath);

    // Shaded Memory Bound Region
    const memShade = document.createElementNS(NS, "polygon");
    memShade.setAttribute("points", `${fx(0.2)},${fy(1)} ${fx(0.2)},${fy(0.2 * hw.bw / 1000)} ${kneeX},${kneeY} ${kneeX},${fy(1)}`);
    memShade.setAttribute("fill", "rgba(184,52,31,0.06)");
    rfSvg.appendChild(memShade);

    // Current Decode Operating Point Dot (Memory Bound)
    const opX = fx(decodeIntensity);
    const opY = fy(decodeIntensity * hw.bw / 1000);

    const halo = document.createElementNS(NS, "circle");
    halo.setAttribute("cx", opX); halo.setAttribute("cy", opY);
    halo.setAttribute("r", "12");
    halo.setAttribute("fill", "rgba(29,110,107,0.25)");
    rfSvg.appendChild(halo);

    const dot = document.createElementNS(NS, "circle");
    dot.setAttribute("cx", opX); dot.setAttribute("cy", opY);
    dot.setAttribute("r", "5.5");
    dot.setAttribute("fill", PAL.mint);
    dot.setAttribute("stroke", PAL.ink);
    dot.setAttribute("stroke-width", "2");
    rfSvg.appendChild(dot);

    const dotLbl = document.createElementNS(NS, "text");
    dotLbl.setAttribute("x", opX + 12);
    dotLbl.setAttribute("y", opY + 4);
    dotLbl.setAttribute("fill", PAL.mint);
    dotLbl.setAttribute("font-size", "10");
    dotLbl.setAttribute("font-weight", "700");
    dotLbl.setAttribute("font-family", '"IBM Plex Mono", ui-monospace, monospace');
    dotLbl.textContent = `Batch=1 Decode (${decodeIntensity.toFixed(1)} FLOP/B)`;
    rfSvg.appendChild(dotLbl);

    // Roofline text captions
    const capX = document.createElementNS(NS, "text");
    capX.setAttribute("class", "bp-cap");
    capX.setAttribute("x", P.l + IW / 2); capX.setAttribute("y", H - 6);
    capX.setAttribute("text-anchor", "middle");
    capX.textContent = "Operational Intensity (FLOPs / Byte, log scale) →";
    rfSvg.appendChild(capX);

    const capY = document.createElementNS(NS, "text");
    capY.setAttribute("class", "bp-cap");
    capY.setAttribute("transform", `rotate(-90 12 ${P.t + IH / 2})`);
    capY.setAttribute("x", 12); capY.setAttribute("y", P.t + IH / 2);
    capY.setAttribute("text-anchor", "middle");
    capY.textContent = "Attainable Performance (TFLOPs) →";
    rfSvg.appendChild(capY);

    // Stats Sidebar
    rfStats.innerHTML = `
      <div class="rf-stat-box">
        <div class="rf-head">Hardware Serving Ceiling</div>
        <div class="rf-row"><span>Memory Bandwidth</span><span class="val amber">${hw.bw} GB/s</span></div>
        <div class="rf-row"><span>Baseline Generation</span><span class="val">${baselineTokS.toFixed(1)} tok/s</span></div>
        <div class="rf-row"><span>MTP Speculative Speed</span><span class="val mint" style="font-size:14px;font-weight:700">${effectiveTokS.toFixed(1)} tok/s</span></div>
        <div class="rf-row"><span>Inter-Token Latency</span><span class="val">${latencyPerTokMs.toFixed(1)} ms</span></div>
        <div class="rf-row"><span>Time to First Token (est)</span><span class="val">${ttftShortMs.toFixed(0)} ms</span></div>
      </div>

      <div class="rf-stat-box" style="padding:10px 14px">
        <div style="font-family:var(--mono);font-size:10px;text-transform:uppercase;color:var(--faint);margin-bottom:4px">Roofline Takeaway</div>
        <div style="font-size:12px;color:var(--muted);line-height:1.45">
          Single-stream generation operates firmly inside the <b>memory-bandwidth-bound slope</b> (${decodeIntensity.toFixed(1)} vs ${kneePoint.toFixed(0)} FLOP/B knee). MTP speculative draft heads multiply effective speed by <b>${mtpSpeedup(mtpBeta, 2).toFixed(2)}×</b> (§08 acceptance model, β=${mtpBeta.toFixed(2)}, γ_verify=0.15) without increasing weight memory traffic.
        </div>
      </div>
    `;
  }

  [rfHwSelect, rfQuantSelect, rfMtpSelect].forEach(sel => {
    if (sel) sel.addEventListener("change", renderRoofline);
  });
  renderRoofline();


  /* ------------------------------------------------------------
     3. Widget: Chunked Prefill & Batch Scheduler (#w-batch-sim)
     ------------------------------------------------------------ */
  const schedPipeline = $("schedPipeline");
  const schedModeSeg = $("schedModeSeg");
  let schedMode = "continuous";

  function renderScheduler() {
    if (!schedPipeline) return;

    if (schedMode === "continuous") {
      schedPipeline.innerHTML = `
        <div class="sched-track">
          <span class="sched-seq-name">Req 1 (128k Doc)</span>
          <div class="sched-blocks">
            <div class="sched-chunk prefill" style="flex:2" title="Prefill Chunk 1 (16k)">Chunk 1/4</div>
            <div class="sched-chunk prefill" style="flex:2" title="Prefill Chunk 2 (16k)">Chunk 2/4</div>
            <div class="sched-chunk prefill" style="flex:2" title="Prefill Chunk 3 (16k)">Chunk 3/4</div>
            <div class="sched-chunk prefill" style="flex:2" title="Prefill Chunk 4 (16k)">Chunk 4/4</div>
            <div class="sched-chunk decode" style="flex:1" title="Autoregressive Decode Step">Dec</div>
            <div class="sched-chunk decode" style="flex:1">Dec</div>
          </div>
        </div>
        <div class="sched-track">
          <span class="sched-seq-name">Req 2 (Chat 2k)</span>
          <div class="sched-blocks">
            <div class="sched-chunk prefill" style="flex:1.5" title="Instant Prefill (2k)">Prefill</div>
            <div class="sched-chunk decode" style="flex:1">Dec</div>
            <div class="sched-chunk decode" style="flex:1">Dec</div>
            <div class="sched-chunk decode" style="flex:1">Dec</div>
            <div class="sched-chunk decode" style="flex:1">Dec</div>
            <div class="sched-chunk decode" style="flex:1">Dec</div>
          </div>
        </div>
        <div class="sched-track">
          <span class="sched-seq-name">Req 3 (Agent SWE)</span>
          <div class="sched-blocks">
            <div class="sched-chunk decode" style="flex:1">Dec</div>
            <div class="sched-chunk decode" style="flex:1">Dec</div>
            <div class="sched-chunk prefill" style="flex:1.5" title="Tool Observation (4k)">Tool Call</div>
            <div class="sched-chunk decode" style="flex:1">Dec</div>
            <div class="sched-chunk decode" style="flex:1">Dec</div>
            <div class="sched-chunk decode" style="flex:1">Dec</div>
          </div>
        </div>
        <div style="font-family:var(--mono);font-size:11px;color:var(--mint);margin-top:6px;display:flex;justify-content:space-between">
          <span>✓ Zero Pipeline Bubbles · Tensor Core Utilization: <b>84%</b></span>
          <span>Max Inter-Token Jitter: <b>&lt;18 ms</b></span>
        </div>
      `;
    } else {
      // Naive batching
      schedPipeline.innerHTML = `
        <div class="sched-track">
          <span class="sched-seq-name">Req 1 (128k Doc)</span>
          <div class="sched-blocks">
            <div class="sched-chunk prefill" style="flex:6" title="Monolithic 128k Prefill Stall (3.8s)">Full Monolithic Prefill (3.8s GPU Lock)</div>
            <div class="sched-chunk decode" style="flex:1">Dec</div>
            <div class="sched-chunk decode" style="flex:1">Dec</div>
          </div>
        </div>
        <div class="sched-track">
          <span class="sched-seq-name">Req 2 (Chat 2k)</span>
          <div class="sched-blocks">
            <div class="sched-chunk bubble" style="flex:6" title="Pipeline Bubble (Blocked waiting for Req 1)">Blocked / Stalled (3.8s Jitter)</div>
            <div class="sched-chunk prefill" style="flex:1">Prefill</div>
            <div class="sched-chunk decode" style="flex:1">Dec</div>
          </div>
        </div>
        <div class="sched-track">
          <span class="sched-seq-name">Req 3 (Agent SWE)</span>
          <div class="sched-blocks">
            <div class="sched-chunk bubble" style="flex:6" title="Pipeline Bubble (Blocked)">Blocked / Stalled</div>
            <div class="sched-chunk decode" style="flex:1">Dec</div>
            <div class="sched-chunk decode" style="flex:1">Dec</div>
          </div>
        </div>
        <div style="font-family:var(--mono);font-size:11px;color:var(--rose);margin-top:6px;display:flex;justify-content:space-between">
          <span>✗ Severe Latency Bubble · Tensor Core Utilization: <b>31%</b></span>
          <span>Inter-Token Latency Spike: <b>3,800 ms</b></span>
        </div>
      `;
    }
  }

  if (schedModeSeg) {
    bindGroup(schedModeSeg, b => { schedMode = b.dataset.mode; renderScheduler(); });
  }
  renderScheduler();
})();

/* ============================================================
   W11 & W12 — Section 14: PyTorch Code Studio & Tensor Tracer
   ============================================================ */
(function () {
  /* ------------------------------------------------------------
     1. Widget: Live Tensor Dimension & Memory Tracer (#w-code-tracer)
     ------------------------------------------------------------ */
  const tracerFlow = $("tracerFlow");
  const tracerStats = $("tracerStats");
  const traceBatchSelect = $("traceBatch");
  const traceCtxSelect = $("traceCtx");
  const traceDtypeSelect = $("traceDtype");
  const traceModuleSelect = $("traceModule");

  if (!tracerFlow || !tracerStats) return;

  // decimal units — the site's locked convention (GB = 1e9 bytes, matching HF/prose figures)
  function formatBytes(bytes) {
    if (bytes < 1e3) return `${bytes.toFixed(0)} B`;
    if (bytes < 1e6) return `${(bytes / 1e3).toFixed(1)} KB`;
    if (bytes < 1e9) return `${(bytes / 1e6).toFixed(2)} MB`;
    return `${(bytes / 1e9).toFixed(2)} GB`;
  }

  function formatFlops(flops) {
    if (flops < 1e9) return `${(flops / 1e6).toFixed(1)} MFLOPs`;
    if (flops < 1e12) return `${(flops / 1e9).toFixed(2)} GFLOPs`;
    return `${(flops / 1e12).toFixed(2)} TFLOPs`;
  }

  function renderTracer() {
    const B = +traceBatchSelect.value || 1;
    const T = +traceCtxSelect.value || 2048;
    const b = +traceDtypeSelect.value || 2;
    const mod = traceModuleSelect.value || "deltanet";

    const D = 5120;
    const D_FF = 17408;
    const VOCAB = 248320;

    let cards = [];
    let stats = {
      params: 0,
      flops: 0,
      actMemory: 0,
      stateMemory: 0,
      throughputTokS: 0
    };

    if (mod === "deltanet") {
      // 1 Gated DeltaNet layer
      const qk_heads = 16, v_heads = 48, dh = 128;
      const in_proj_dim = (qk_heads * 2 + v_heads) * dh; // 10,240
      const out_proj_dim = v_heads * dh; // 6,144

      const in_bytes = B * T * D * b;
      const qkv_bytes = B * T * in_proj_dim * b;
      const beta_bytes = B * T * v_heads * b;
      const state_bytes = B * v_heads * dh * dh * b;
      const out_bytes = B * T * D * b;

      stats.params = (D * in_proj_dim) + (D * v_heads) + (v_heads * dh * D);
      stats.flops = (2 * B * T * D * in_proj_dim) + (3 * B * T * v_heads * dh * dh) + (2 * B * T * out_proj_dim * D);
      stats.actMemory = in_bytes + qkv_bytes + beta_bytes + out_bytes;
      stats.stateMemory = state_bytes;

      cards = [
        {
          op: "1. Residual Input Vector (x)",
          tag: "Activation",
          shape: `[${B}, ${T}, 5120]`,
          bytes: formatBytes(in_bytes),
          math: "Input residual hidden stream from preceding RMSNorm (d_model = 5120)"
        },
        {
          op: "2. Fused Q, K, V Linear Projection (W_in)",
          tag: "GEMM Projection",
          shape: `[${B}, ${T}, 10240]`,
          bytes: formatBytes(qkv_bytes),
          math: `Q, K: <code>[${B}, ${T}, 16, 128]</code> · V: <code>[${B}, ${T}, 48, 128]</code>`
        },
        {
          op: "3. L2 Head Normalization & Write Gate (β_t)",
          tag: "Normalization + Gating",
          shape: `[${B}, ${T}, 48]`,
          bytes: formatBytes(beta_bytes),
          math: `L2-norm on Q,K; Q/K repeated 3× to 48 heads; β_t = 2σ(xW_β + b_β) ∈ (0, 2)`
        },
        {
          op: "4. Associative Recurrent State Matrix (S_t)",
          tag: "State Memory",
          shape: `[${B}, 48, 128, 128]`,
          highlight: true,
          bytes: formatBytes(state_bytes),
          math: `Delta-rule matrix update: <code>S_t = α_t S_{t-1} − β_t (S_{t-1} k_t − v_t) k_tᵀ</code>`
        },
        {
          op: "5. Associative Readout & Output Projection (W_out)",
          tag: "Output GEMM",
          shape: `[${B}, ${T}, 5120]`,
          bytes: formatBytes(out_bytes),
          math: `y_t = S_tᵀ q_t ∈ <code>[${B}, ${T}, 6144]</code> → Linear Proj to <code>[${B}, ${T}, 5120]</code>`
        }
      ];
    } else if (mod === "attention") {
      // 1 Gated Full Attention layer (GQA 24:4)
      const q_heads = 24, kv_heads = 4, dh = 256;
      const q_dim = q_heads * dh; // 6,144
      const kv_dim = kv_heads * dh; // 1,024

      const in_bytes = B * T * D * b;
      const q_bytes = B * T * q_dim * b;
      const kv_bytes = 2 * B * T * kv_dim * b;
      const gate_bytes = B * T * q_dim * b;
      const attn_map_bytes = B * q_heads * T * Math.min(T, 4096) * b;
      const out_bytes = B * T * D * b;

      stats.params = (D * q_dim) + (2 * D * kv_dim) + (D * q_dim) + (q_dim * D);
      stats.flops = (2 * B * T * D * (q_dim + 2 * kv_dim + q_dim)) + (4 * B * q_heads * T * T * dh) + (2 * B * T * q_dim * D);
      stats.actMemory = in_bytes + q_bytes + kv_bytes + gate_bytes + out_bytes;
      stats.stateMemory = kv_bytes; // Unbounded growing KV buffer

      cards = [
        {
          op: "1. Residual Input Vector (x)",
          tag: "Activation",
          shape: `[${B}, ${T}, 5120]`,
          bytes: formatBytes(in_bytes),
          math: "Input residual stream from RMSNorm(x)"
        },
        {
          op: "2. Q, K, V & Gate Projections",
          tag: "Linear GEMM",
          shape: `Q: [${B}, ${T}, 24, 256] · KV: [${B}, ${T}, 4, 256]`,
          bytes: formatBytes(q_bytes + kv_bytes + gate_bytes),
          math: `Grouped-Query Attention (24 query heads : 4 KV heads = 6:1 sharing)`
        },
        {
          op: "3. 3D MRoPE Rotary Phase Embedding",
          tag: "Coordinate Rotation",
          shape: `[${B}, ${T}, 24, 64] rotary`,
          bytes: formatBytes(q_bytes * (64 / 256)),
          math: `Rotates head channels 0..63 across <code>[11, 11, 10]</code> sections; channels 64..255 pass unmodified`
        },
        {
          op: "4. Autoregressive KV-Cache Buffer",
          tag: "Unbounded Buffer",
          shape: `[${B}, 4, ${T}, 256] × 2 (K & V)`,
          highlight: true,
          bytes: formatBytes(kv_bytes),
          math: `Grows linearly with sequence length T: <code>2 · 4 · 256 · T · dtype</code>`
        },
        {
          op: "5. FlashAttention SDPA & Output σ-Gate",
          tag: "Attention + Gating",
          shape: `[${B}, ${T}, 5120]`,
          bytes: formatBytes(out_bytes),
          math: `Output = <code>(SDPA(Q, K_rep, V_rep) ⊙ σ(x W_g)) @ W_out</code>`
        }
      ];
    } else if (mod === "swiglu") {
      // 1 SwiGLU FFN block
      const in_bytes = B * T * D * b;
      const gate_up_bytes = 2 * B * T * D_FF * b;
      const out_bytes = B * T * D * b;

      stats.params = (2 * D * D_FF) + (D_FF * D);
      stats.flops = (2 * B * T * D * 2 * D_FF) + (3 * B * T * D_FF) + (2 * B * T * D_FF * D);
      stats.actMemory = in_bytes + gate_up_bytes + out_bytes;
      stats.stateMemory = 0;

      cards = [
        {
          op: "1. Pre-FFN Residual Input",
          tag: "Activation",
          shape: `[${B}, ${T}, 5120]`,
          bytes: formatBytes(in_bytes),
          math: "Post-mixer residual stream passed through RMSNorm"
        },
        {
          op: "2. Parallel Gate & Up Projections (W_gate, W_up)",
          tag: "GEMM Projection",
          shape: `[${B}, ${T}, 17408] × 2`,
          highlight: true,
          bytes: formatBytes(gate_up_bytes),
          math: `Expansion ratio 3.4 × 5120 = 17,408 intermediate hidden dimension (padded to a multiple of 2,048)`
        },
        {
          op: "3. Swish (SiLU) Activation & Elementwise Hadamard Product",
          tag: "Non-Linearity",
          shape: `[${B}, ${T}, 17408]`,
          bytes: formatBytes(B * T * D_FF * b),
          math: `Intermediate tensor: <code>h_ffn = SiLU(x @ W_gate) ⊙ (x @ W_up)</code>`
        },
        {
          op: "4. Down-Projection to Residual Bus (W_down)",
          tag: "Down GEMM",
          shape: `[${B}, ${T}, 5120]`,
          bytes: formatBytes(out_bytes),
          math: `Final projection: <code>h_ffn @ W_down</code> added to residual stream`
        }
      ];
    } else if (mod === "mtp") {
      // MTP Speculative Prediction Module
      const K = 2;
      const in_bytes = B * T * D * b;
      const proj_bytes = K * B * T * D * b;
      const logits_bytes = K * B * T * VOCAB * b;

      // K draft blocks (one d×d projection each) + ONE shared d·V vocab head —
      // the MTP heads reuse the backbone LM head rather than owning private copies.
      stats.params = K * (D * D) + D * VOCAB;
      stats.flops = K * ((2 * B * T * D * D) + (2 * B * T * D * VOCAB));
      stats.actMemory = in_bytes + proj_bytes + logits_bytes;
      stats.stateMemory = 0;

      cards = [
        {
          op: "1. Final Backbone Hidden States (h_64)",
          tag: "Backbone State",
          shape: `[${B}, ${T}, 5120]`,
          bytes: formatBytes(in_bytes),
          math: "Latent output from 64th super-block layer"
        },
        {
          op: "2. MTP Draft Head Projections (Depth k=1, k=2)",
          tag: "Draft GEMM",
          shape: `[${B}, ${T}, 2, 5120]`,
          bytes: formatBytes(proj_bytes),
          math: `Predicts +1 and +2 future token draft representations simultaneously`
        },
        {
          op: "3. Untied Vocabulary Classification Logits",
          tag: "Speculative Logits",
          shape: `[${B}, ${T}, 2, 248320]`,
          highlight: true,
          bytes: formatBytes(logits_bytes),
          math: `Parallel draft logits evaluated against verification threshold in a single forward step`
        }
      ];
    } else if (mod === "superblock") {
      // 1 Hybrid Super-Block = 3 DeltaNet + 1 Attention + 4 FFNs
      const dn_state_bytes = 3 * (B * 48 * 128 * 128 * b);
      const attn_kv_bytes = 2 * B * 4 * T * 256 * b;
      const total_params = (3 * (5120 * 10240 + 5120 * 48 + 6144 * 5120)) +
                           (5120 * 6144 + 2 * 5120 * 1024 + 5120 * 6144 + 6144 * 5120) +
                           (4 * (2 * 5120 * 17408 + 17408 * 5120));

      stats.params = total_params;
      stats.flops = total_params * 2 * B * T;
      stats.actMemory = B * T * 5120 * b * 8;
      stats.stateMemory = dn_state_bytes + attn_kv_bytes;

      cards = [
        {
          op: "Layer 4k+0: Gated DeltaNet #1 + SwiGLU",
          tag: "Linear Attention",
          shape: `State: [${B}, 48, 128, 128]`,
          bytes: formatBytes(B * 48 * 128 * 128 * b),
          math: "Sub-quadratic prefill + constant-space recurrence"
        },
        {
          op: "Layer 4k+1: Gated DeltaNet #2 + SwiGLU",
          tag: "Linear Attention",
          shape: `State: [${B}, 48, 128, 128]`,
          bytes: formatBytes(B * 48 * 128 * 128 * b),
          math: "Associative memory retrieval with learned forgetting"
        },
        {
          op: "Layer 4k+2: Gated DeltaNet #3 + SwiGLU",
          tag: "Linear Attention",
          shape: `State: [${B}, 48, 128, 128]`,
          bytes: formatBytes(B * 48 * 128 * 128 * b),
          math: "Deep syntactic & linguistic token tracking"
        },
        {
          op: "Layer 4k+3: Gated Full Attention + SwiGLU",
          tag: "Full Attention",
          shape: `KV Cache: [${B}, 4, ${T}, 256]`,
          highlight: true,
          bytes: formatBytes(attn_kv_bytes),
          math: "Exact global cross-token routing with 3D MRoPE positions"
        }
      ];
    }

    // Render flow cards
    tracerFlow.innerHTML = cards.map((c, i) => `
      <div class="trace-card">
        <div class="trace-card-head">
          <span class="trace-op">${c.op}</span>
          <span class="trace-tag">${c.tag}</span>
        </div>
        <div class="trace-tensor">
          <span class="trace-shape ${c.highlight ? "highlight" : ""}">${c.shape}</span>
          <span class="trace-bytes">${c.bytes}</span>
        </div>
        <div class="trace-math">${c.math}</div>
      </div>
      ${i < cards.length - 1 ? `<div class="trace-connector">↓</div>` : ""}
    `).join("");

    // Render statistics sidebar
    tracerStats.innerHTML = `
      <div class="tracer-stat-box">
        <div class="tsb-head">Subsystem Resource Profile</div>
        <div class="tsb-row"><span>Parameters</span><span class="val amber">${(stats.params / 1e6).toFixed(1)} M</span></div>
        <div class="tsb-row"><span>Forward FLOPs</span><span class="val">${formatFlops(stats.flops)}</span></div>
        <div class="tsb-row"><span>Peak Activation RAM</span><span class="val">${formatBytes(stats.actMemory)}</span></div>
        <div class="tsb-row"><span>Persistent State RAM</span><span class="val mint">${formatBytes(stats.stateMemory)}</span></div>
      </div>

      <div class="tracer-stat-box">
        <div class="tsb-head">Serving Hardware Context</div>
        <div class="tsb-row"><span>Batch Size (B)</span><span class="val">${B}</span></div>
        <div class="tsb-row"><span>Context Tokens (T)</span><span class="val">${T.toLocaleString()}</span></div>
        <div class="tsb-row"><span>Precision Dtype</span><span class="val">${b === 2 ? "BF16 (16-bit)" : b === 1 ? "FP8 (8-bit)" : "FP32 (32-bit)"}</span></div>
        <div class="tsb-row"><span>Architecture Ratio</span><span class="val amber">3:1 Hybrid</span></div>
      </div>

      <div class="tracer-stat-box" style="padding:10px 14px">
        <div style="font-family:var(--mono);font-size:10px;text-transform:uppercase;color:var(--faint);margin-bottom:4px">Key Tensor Insight</div>
        <div style="font-size:12px;color:var(--muted);line-height:1.45">
          ${mod === "deltanet" 
            ? "DeltaNet's associative matrix <code>S</code> remains strictly constant in size (<b>" + formatBytes(stats.stateMemory) + "</b>) regardless of whether sequence length T is 1 or 262,144 tokens."
            : mod === "attention"
            ? "Full Attention's KV cache scales linearly with T (<b>" + formatBytes(stats.stateMemory) + "</b>), requiring the 3:1 hybrid design to bound total footprint."
            : "SwiGLU provides a wide 3.4× expansion for superior multi-step reasoning capabilities with zero recurrent state overhead."}
        </div>
      </div>
    `;
  }

  [traceBatchSelect, traceCtxSelect, traceDtypeSelect, traceModuleSelect].forEach(sel => {
    if (sel) sel.addEventListener("change", renderTracer);
  });
  renderTracer();


  /* ------------------------------------------------------------
     2. Widget: PyTorch Implementation Studio (#w-code-lab)
     ------------------------------------------------------------ */
  // Python sources live beside the studio markup as <script type="text/plain" data-py="…">
  // blocks (_parts/05_bench_code_end.html). text/plain is never executed by browsers and
  // needs no fetch(), so file:// keeps working; textContent is exact because the Python
  // body and each closing tag sit flush-left (bootstrap strips one edge newline per side).
  const CODE_FILES = {
    full: {
      name: "qwen38_full.py",
      desc: "Complete self-contained hybrid architecture with 16×[ΔΔΔA] super-blocks, SwiGLU, and MTP head.",
      meta: "342 lines · Python 3.10+ · PyTorch 2.0+"
    },
    deltanet: {
      name: "01_deltanet.py",
      desc: "Gated DeltaNet linear mixer with L2 normalization and associative delta-rule state updates.",
      meta: "68 lines · Linear Attention §04"
    },
    attention: {
      name: "02_attention.py",
      desc: "Gated Attention with Grouped-Query Attention (GQA), 3D MRoPE, and Sigmoid output gate.",
      meta: "66 lines · Full Attention §05"
    },
    swiglu: {
      name: "03_swiglu_block.py",
      desc: "RMSNorm, SwiGLU FFN with 17,408 hidden dimension, and the 16×[ΔΔΔA] hybrid super-block motif.",
      meta: "55 lines · FFN & Residual Bus §06"
    },
    mtp: {
      name: "04_mtp_head.py",
      desc: "Multi-Token Prediction draft heads for parallel speculative decoding.",
      meta: "24 lines · Speculative Heads §08"
    },
    verify: {
      name: "05_run_verify.py",
      desc: "Executable test harness with shape assertions, parameter counting, and forward inference timing.",
      meta: "46 lines · Verification Script"
    }
  };
  document.querySelectorAll('script[type="text/plain"][data-py]').forEach(el => {
    const f = CODE_FILES[el.dataset.py];
    if (!f) return;
    f.code = el.textContent.replace(/^\n/, "").replace(/\n$/, "");
  });

  let activeFile = "full";
  let showAnnotations = true;

  const codeTabsContainer = $("codeStudioTabs");
  const codeContentEl = $("codeStudioContent");
  const cfFilenameEl = $("cfFilename");
  const cfDescEl = $("cfDesc");
  const cfMetaEl = $("cfMeta");
  const btnToggleAnnot = $("btnToggleAnnot");
  const annotStateEl = $("annotState");
  const btnCopyCode = $("btnCopyCode");
  const copyBtnText = $("copyBtnText");
  const btnDownloadCode = $("btnDownloadCode");

  function highlightPython(code) {
    return tokenizePython(code, showAnnotations);
  }

  // Tokenize a Python source string to a flat text + tokens map, then re-emit.
  // This avoids regex-pass chaining where later passes see HTML emitted by
  // earlier passes (e.g. `class` inside `<span class="k">…</span>`).
  function tokenizePython(src, showAnns) {
    let text = src.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

    // 1. Pull out docstrings / strings / comments / annotation badges into
    //    sentinel placeholders so the kw/fn/num passes below don't re-touch them.
    //    Placeholder uses non-word, non-digit characters only.
    const tokens = []; // [{ open, body }]
    // Encode id as a base-26 letter sequence so the number regex never matches
    // digits inside the placeholder token.
    const id2s = (n) => String.fromCharCode(97 + (n % 26)) + (n >= 26 ? id2s(Math.floor(n / 26) - 1) : "");
    const place = (open, body) => {
        const id = tokens.length;
        tokens.push({ open, body });
        return `{{§${id2s(id)}§}}`;
    };
    text = text.replace(/("""[\s\S]*?""")/g, (m) => place('<span class="s">', m));
    text = text.replace(/("(\\.|[^"\\])*"|'(\\.|[^'\\])*')/g, (m) => place('<span class="s">', m));
    text = text.replace(/(#[^\n]*)/g, (m) => place('<span class="c">', m));
    if (showAnns) {
        text = text.replace(/(\(eq\.\s*[\d–\-,\s]+\))/g, (m) => place('<span class="cm-badge amber">', m));
        text = text.replace(/(§0\d)/g, (m) => place('<span class="cm-badge mint">', m));
    }

    // 2. Keywords / functions / numbers — apply on flat source. The {{§N§}}
    //    sentinel uses non-word characters so neither kw nor num regex touches it.
    const kws = ["class", "def", "return", "import", "from", "super", "self", "for", "in", "range", "if", "else", "is", "not", "None", "True", "False", "with", "as", "assert", "pass"];
    kws.forEach(kw => {
        const re = new RegExp(`\\b(${kw})\\b`, "g");
        text = text.replace(re, '<span class="k">$1</span>');
    });
    text = text.replace(/\.([a-zA-Z_]\w*)\(/g, '.<span class="f">$1</span>(');
    text = text.replace(/\bdef\s+([a-zA-Z_]\w*)/g, 'def <span class="f">$1</span>');
    text = text.replace(/\bclass\s+([a-zA-Z_]\w*)/g, 'class <span class="f">$1</span>');
    text = text.replace(/\b(\d+(\.\d+)?(e[+-]?\d+)?)\b/g, '<span class="n">$1</span>');

    // 3. Restore placeholders. Match the letter-encoded id of length 1+ and reverse.
    const s2id = (s) => {
        let n = 0;
        for (const ch of [...s].reverse()) n = n * 26 + (ch.charCodeAt(0) - 96);
        return n - 1;
    };
    text = text.replace(/\{\{§([a-z]+)§\}\}/g, (_m, idStr) => {
        const t = tokens[s2id(idStr)];
        return t.open + t.body + '</span>';
    });

    return text;
}

  function renderCodeStudio() {
    const file = CODE_FILES[activeFile];
    if (!file || !codeContentEl) return;

    if (cfFilenameEl) cfFilenameEl.textContent = file.name;
    if (cfDescEl) cfDescEl.textContent = file.desc;
    if (cfMetaEl) cfMetaEl.textContent = file.meta;

    codeContentEl.innerHTML = highlightPython(file.code);

    if (codeTabsContainer) {
      codeTabsContainer.querySelectorAll(".code-tab").forEach(tab => {
        const isCur = tab.dataset.file === activeFile;
        tab.classList.toggle("on", isCur);
        tab.setAttribute("aria-selected", String(isCur));
      });
    }
  }

  if (codeTabsContainer) {
    codeTabsContainer.addEventListener("click", e => {
      const tab = e.target.closest(".code-tab");
      if (!tab || tab.dataset.file === activeFile) return;
      activeFile = tab.dataset.file;
      renderCodeStudio();
    });
  }

  if (btnToggleAnnot) {
    btnToggleAnnot.addEventListener("click", () => {
      showAnnotations = !showAnnotations;
      if (annotStateEl) annotStateEl.textContent = showAnnotations ? "ON" : "OFF";
      renderCodeStudio();
    });
  }

  if (btnCopyCode) {
    const origLabel = copyBtnText ? copyBtnText.textContent : "Copy code";
    btnCopyCode.addEventListener("click", () => {
      const file = CODE_FILES[activeFile];
      if (!file) return;
      navigator.clipboard.writeText(file.code).then(() => {
        if (copyBtnText) copyBtnText.textContent = "Code copied ✓";
        btnCopyCode.classList.add("primary");
        setTimeout(() => {
          if (copyBtnText) copyBtnText.textContent = origLabel;
          btnCopyCode.classList.remove("primary");
        }, 2000);
      }).catch(() => {
        if (copyBtnText) copyBtnText.textContent = "Copy failed — select the code and copy manually.";
        setTimeout(() => { if (copyBtnText) copyBtnText.textContent = origLabel; }, 2600);
      });
    });
  }

  if (btnDownloadCode) {
    btnDownloadCode.addEventListener("click", () => {
      const file = CODE_FILES[activeFile];
      if (!file) return;
      const blob = new Blob([file.code], { type: "text/x-python;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = file.name;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    });
  }

  renderCodeStudio();

  // KaTeX render pass — runs once after scripts load.
  // ponytail: scope to <main> to skip nav/footer; delimiters match § math.
  // ponytail: also render bare <div class="math"> blocks as display math
  // (the page has 13 such blocks holding unwrapped TeX).
  function katexReady(){ return !!(window.renderMathInElement && window.katex); }
  function runKatex(){
    if(!katexReady()) return;
    const root = document.querySelector("main") || document.body;
    katexRender(root);                       // shared helper — same config as dynamic re-renders
    // Render bare .math divs as display math (skip ones already rendered
    // by auto-render, and skip the synthetic sample math block at .w-out).
    document.querySelectorAll(".math").forEach((el) => {
      if (el.querySelector(".katex")) return;
      if (el.closest(".w-out")) return;
      const tex = el.textContent.trim();
      if (!tex) return;
      try {
        window.katex.render(tex, el, { displayMode: true, throwOnError: false });
      } catch (_) {}
    });
  }
  if(katexReady()) runKatex();
  else window.addEventListener("load", runKatex);
})();
