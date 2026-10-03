# Qwen 3.8-27B — The Architecture, From Scratch

A self-contained static explainer site that derives Alibaba's **Qwen 3.8-27B**
(August 2026, Apache 2.0) from first principles: architecture mathematics, data
pipeline, training curriculum, post-training RL, and inference optimizations,
with interactive widgets, SVG diagrams, and animated charts. No build step,
no framework, no npm.

**Read it live:** https://atandra2000.github.io/qwen3.8-guide/

Part of a from-scratch deep learning portfolio. See
[atandra2000](https://github.com/atandra2000) for the accompanying model
implementations (GPT-2 → hybrid DeltaNet/attention stacks) in raw PyTorch.

## Run locally

```bash
cd qwen38-site
python3 -m http.server 8137
# open http://localhost:8137
```

Any static file server works (`npx serve`, `php -S`, nginx…). Opening
`index.html` directly from disk also works, because there are no fetch() calls.

## Deploy

| Host | Command |
|------|---------|
| GitHub Pages | **Live:** https://atandra2000.github.io/qwen3.8-guide/ (served from branch `main`, root `/`) |
| Vercel | `vercel --prod` (framework preset: Other) |
| Netlify | drag-and-drop the folder onto app.netlify.com/drop |
| Cloudflare Pages | connect repo, build command: none, output dir: `/` |
| S3 | `aws s3 sync . s3://<bucket>/ --cache-control "max-age=86400"` |

## Structure

```
qwen38-site/
├── index.html          # the whole page (assembled from _parts/)
├── css/styles.css      # design system (research-paper dark, two type registers)
├── js/app.js           # 36 interactive widgets, all computed client-side
└── _parts/             # source sections; edit here, then re-assemble:
                         cat _parts/0*.html > index.html
```

## Design system

Two registers, one system: **serif carries the argument** (STIX Two Text
masthead + headings and body prose, one family with hierarchy by weight as in
journal typesetting), **mono carries the evidence** (IBM Plex Mono for math, tables,
code, widget readouts, labels). Color is semantic, not
decorative: amber = DeltaNet/primary, mint = attention/vision, iris =
training/RL, sky = data pipeline, rose = warnings. The hero is a datasheet-style
spec sheet, not stat cards.

## Content map

| § | Section | Key math / artifacts |
|---|---------|---------------------|
| 01 | Orientation + parameter audit | full dataflow diagram (Fig. 1), live param accounting → 27.58B ≈ 55.2 GB BF16 |
| 02 | Tokenizer & input contract | interactive BPE token segmenter, multilingual fertility matrix, special token protocol catalog |
| 03 | Hybrid backbone | 16×[3Δ+1A] periodic super-block, 64-layer memory accumulator, motif visualizer, Pareto matrix |
| 04 | Gated DeltaNet | eq.(3)–(4): delta rule + α/β gates, native associative memory lab, chunked WY scan visualizer |
| 05 | Gated Attention | eq.(5)–(7): GQA 6:1 KV cache calculator, 3D MRoPE [11,11,10] phase visualizer, output σ-gate noise suppressor |
| 06 | FFN & residual stream | eq.(8)–(10): SwiGLU 17.11B parameter accounting, activation explorer, residual bus Pre-RMSNorm variance tracker |
| 07 | Vision front-end | interactive 5-stage multimodal pipeline, image/video token budget calculator, SigLIP pairwise loss, ViT parameter accounting |
| 08 | Multi-Token Prediction | eq.(12)–(13), interactive speculative decoding engine, single-pass verification tree, MTP vs draft model matrix |
| 09 | Data pipeline | interactive 5-stage data funnel, dynamic curriculum mix explorer, best-fit sequence packing visualizer, verifier synthesis matrix |
| 10 | Pretraining | interactive 3-stage curriculum, loss trajectory milestone scrubber, cluster compute & budget calculator, AdamW β₂ fast-forgetting simulator |
| 11 | Post-training | interactive 4-stage pipeline, critic-free GRPO group advantage simulator, test-time compute explorer, agentic CoT trace viewer |
| 12 | Optimizations | stacked VRAM budget planner, hardware roofline & tok/s simulator, chunked prefill scheduler, microscaling inspector |
| 13 | Benchmarks | multi-axis capability radar, parity scatter, comparative bar waterfall, Pareto efficiency frontier, master benchmark index |
| 14 | From scratch, as code | interactive tensor dimension tracer, multi-tab PyTorch studio, downloadable .py scripts, mixer contract matrix |
| 15 | Synthesis | six takeaways |

## Sourcing honesty

- **Architecture specs** (64 layers, 48Δ/16A split, head counts, dims, vocab,
  context, YaRN params): official HF model card
  [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B).
- **Benchmarks**: vendor-published numbers, labeled as such on the page.
- **Training/data methodology**: follows the Qwen3 technical report
  ([arXiv:2505.09388](https://arxiv.org/abs/2505.09388)), the most recent
  *published* recipe. The 3.8 report had not been released when this was
  written; scale figures are marked as such.
- **Parameter audit assumptions**: the §01 widget counts every projection once
  and includes the untied LM head. The MTP module (≈1.32B) and vision tower
  (≈0.9B) are estimates, so toggle them off/on in the widget. With both included:
  27.58B ≈ 55.2 GB BF16 (decimal GB, ÷1e9, matching HF and prose figures).
- Every derived number (param counts, KV cache sizes, FLOPs) is recomputed
  client-side from the config so the page can't drift from its own math.

## Tests

Static checks for `_parts/`↔`index.html` assembly integrity, stylesheet
contract, parameter-math regression, and accessibility fallbacks:

```bash
npm test        # node test-site.mjs
```

## License

MIT. See [LICENSE](LICENSE).
