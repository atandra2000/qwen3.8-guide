# CLAUDE.md

Static explainer site for the Qwen 3.8-27B architecture. No build step, no npm install, no framework. It has 36 interactive widgets. Live at https://atandra2000.github.io/qwen3.8-guide/

## Commands

```bash
npm test          # node test-site.mjs
npm run serve     # python3 -m http.server 8137
```

## Layout

| Path | Holds |
|---|---|
| `index.html` | Page shell |
| `js/`, `css/` | Behaviour and styling |
| `_parts/` | Page fragments |
| `test-site.mjs` | Site checks |

## Notes

- Keep the site dependency-free. Do not add npm packages or a bundler.
- `AUDIT-2026-08-24.md` is a past audit record. Do not edit it.
- Use `codegraph explore "<query>"` before you grep. The index is in `.codegraph/`.
