# LLM playground

Learning project: change a model setting in a web UI and see the effect on the answer, tokens and latency. Every send is stored in SQLite.

## Stack
Python 3.11 (venv in `env/`), FastAPI + uvicorn, SQLite (`sqlite3` from the standard library), `requests` (no provider SDKs), React + Vite (JavaScript). Providers: Mistral and Gemini.

## Run
- Setup: `env\Scripts\activate`, then `pip install -r requirements.txt`; copy `.env.example` to `.env` and add the keys.
- Backend: `uvicorn backend.main:app --reload` (port 8000, docs at `/docs`).
- Frontend: `cd frontend`, `npm install`, `npm run dev` (port 5173, proxies `/api` to the backend).
- Replay an experiment: `python scripts/replay.py --experiment temperature --provider mistral --n 5`.
- Report: `python scripts/report.py`.
- Phase 1 terminal chat: `python chat.py --provider mistral`.

## Structure
- `backend/llm.py`: the only module that talks to providers; returns one common result.
- `backend/db.py`: the only module that touches SQLite (`data/runs.db`, table `runs`).
- `backend/main.py`: FastAPI routes; glue between the UI, `llm.py` and `db.py`.
- `scripts/`: `replay.py` (N runs per setting into `runs`), `report.py` (report from `runs`).
- `frontend/`: React UI; it only calls `/api/*`.
- `report/`: the mini-report and the evidence rows.
- `chat.py`, `experiments.py`, `questions.json`, `results.json`: phase 1, left untouched.

## Conventions
- Plain, readable code; every module must be explainable by the author. No clever abstractions.
- The UI never calls a provider; only `llm.py` does.
- A parameter left empty is not sent (provider default) and is stored as NULL.
- Failed sends are saved in `runs` with the `error` column.
- Keep the frontend, backend and database responsibilities separate.

## Rules
- Never commit API keys or `.env`. Keys live only in `.env`; never print or log them.
- Error messages go through `redact()` before they are shown or stored.
- Do not read or edit `.env` (denied in `.claude/settings.json`).
- Commit in small steps; push `master` to `origin main` after each commit.
