# Quality Goals 2026-06-06

## Sources Reviewed

- web.dev Core Web Vitals: LCP, INP, CLS and 75th percentile thresholds.
- TanStack Query docs: stale data, refetch behavior, placeholder data, retry behavior.
- FastAPI docs: HTTPException handling and TestClient-based smoke tests.
- OWASP Authentication and Session Management cheat sheets: authenticated session/token handling and reauthentication guidance.
- Vite docs: only `VITE_*` env values are exposed to browser bundles; they must not contain secrets.

## Product Goals

- First visit should show the application shell quickly instead of a long blocking splash.
- Article list loading should preserve layout and show skeletons or stale data while fetching.
- Empty, error, and unauthorized states should show a clear next action.
- Admin actions must not be reachable without server-side authentication.
- Deployment readiness should be checked by one local preflight command.

## Metrics And Gates

- Core Web Vitals target: p75 LCP <= 2.5s, INP <= 200ms, CLS <= 0.1.
- Local API target: `/api/articles`, `/api/articles/{id}`, `/api/admin/stats`, `/health` p95 <= 800ms under normal DB conditions.
- Security target: `/api/admin/*` returns 401 for missing/wrong password and 503 if `ADMIN_PASSWORD` is not configured.
- Data payload target: article list APIs must not fetch or return `full_text`, embeddings, or trust rationale fields.
- Observability target: backend responses include `Server-Timing`; frontend reports lightweight client performance events.
- Verification gate: `python scripts/preflight.py` passes before deployment.

## Implemented Changes

- Shortened splash duration and avoided blocking the main list on secondary queries.
- Added query stale-time defaults, placeholder data, skeletons, error CTAs, and empty-state CTAs.
- Added admin password verification on the backend and memory-only admin authentication on the frontend.
- Added backend `/health`, request timing logs, and admin smoke tests.
- Reduced article list and admin stats DB payloads.
- Added thumbnail caching and event log redaction.
- Added `scripts/preflight.py` for local quality checks.

## Remaining Backlog

- Replace deprecated `google.generativeai` with the newer Google Gen AI SDK.
- Add durable background job records beyond process memory.
- Add Playwright UI regression tests for main, detail, admin login, empty, and error states.
- Add real production monitoring aggregation for `client_performance` events.
