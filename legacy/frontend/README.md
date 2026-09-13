# PointsX — frontend (Vercel)

Static SPA for the PointsX body-measurement demo. Talks to the backend
running on Hugging Face Spaces (separate repo: `Pointx-backend`).

```
browser ⇄ Vercel (this repo)  →  HF Space (Pointx-backend)  →  S3 bucket
```

## Repo layout

```
static/                       ← the SPA (HTML/JS/CSS)
  index.html
  config.js                   ← runtime API base, rewritten at deploy
  css/                        css/
  js/                         js/
  data/                       data/
scripts/
  build_static.mjs            Vercel build step
vercel.json                   build + rewrite rules
.env.example                  local preview env vars
```

The SPA itself only needs **one** runtime config value:

```js
// static/config.js
window.POINTSX_API_BASE = "https://<your-space>.hf.space";
```

At deploy `scripts/build_static.mjs` overwrites that file with the value of
the `POINTSX_API_BASE` env var configured on Vercel. The compiled output
lands in `public/`, which Vercel serves.

## Local preview

You don't need Node or Vercel CLI to test — any static server pointed at
`static/` works, but `config.js` will be the dev placeholder:

```powershell
# Quick check (Python's built-in static server)
cd static
python -m http.server 5173
# Open http://localhost:5173/ — calls POINTSX_API_BASE from config.js
```

For an end-to-end Vercel-style preview:

```powershell
npm i -g vercel
copy .env.example .env
notepad .env                 # set POINTSX_API_BASE
vercel dev                   # runs the build script, serves public/
```

## Deploy

1. Import this repo into Vercel as a new project.
   - **Framework preset**: Other
   - **Build command**: `node scripts/build_static.mjs`
   - **Output directory**: `public`
2. Settings → **Environment Variables** → add:
   - `POINTSX_API_BASE` = `https://<your-username>-<space-name>.hf.space`
   (Production, Preview, Development — same value for all three.)
3. Trigger a deploy. The Vercel URL serves the SPA; every API call goes
   to the Space.
4. Add your custom domain under **Domains** when ready.

## Locking down

While testing, the backend accepts any origin (`CORS_ALLOW_ORIGINS=*`).
Once the Vercel URL or your custom domain is stable, restrict the
backend by setting on the HF Space:

```
CORS_ALLOW_ORIGINS=https://your-domain.example,https://your-app.vercel.app
```

## Notes

- The build script copies `static/index.html` to `public/index.html` AND
  the whole `static/` tree to `public/static/`. This mirrors the path
  layout the SPA's own HTML/JS expect (`/static/...`), so nothing needs
  to be rewritten beyond `config.js`.
- No JS bundler / TypeScript / framework — plain ES modules.
- For production hardening: add a `headers` block to `vercel.json` for
  cache-control + a CSP. Out of scope for the scientific demo.
