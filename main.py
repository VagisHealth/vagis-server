"""
Vagis backend — agent relay + metric ingestion + enrollment-code registry.

    Vagis app  ->  POST /chat    (holds the key)   ->  Anthropic API
    Vagis app  ->  POST /ingest  (code, mode, CSV)  ->  Postgres
    You        ->  POST /admin/researchers          ->  issues an RP code
    Researcher ->  POST /portal/subjects            ->  issues an SE code
    App        ->  POST /portal/validate            ->  checks an SE code is real

Enrollment code scheme (no hyphens):
    Researcher : RP + 3 digits                       e.g. RP014
    Subject    : SE + rp(3) + user(4) + tail(3) = 12 e.g. SE0140001Q4J
The embedded RP digits let a researcher's portal show only their own subjects.
The random tail (unambiguous alphabet, no O/0/I/1/L) means a mistyped code fails
cleanly instead of silently matching another real subject.

Endpoints:
    GET  /health              -- server status
    POST /chat                -- personal agent relay
    POST /ingest              -- upload a cumulative CSV for one code + mode
    GET  /ingest/list         -- list uploads (verification / portal read)
    POST /admin/researchers   -- (admin) issue an RP code for a researcher
    GET  /admin/researchers   -- (admin) list researchers
    POST /portal/subjects     -- (researcher) issue the next SE code under their RP
    GET  /portal/subjects     -- (researcher) list their own subjects
    POST /portal/validate     -- (app) confirm an SE code exists, return its RP
"""

from __future__ import annotations

import csv
import io
import os
import secrets
from typing import Any, Optional

import anthropic
import psycopg2
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Configuration  (all via environment variables -- nothing secret in the code)
# --------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Shared secret the APP sends (Authorization: Bearer ...). Ships in the app, so
# treat it as "keeps randoms out", not as a high-value secret.
VAGIS_APP_TOKEN = os.environ.get("VAGIS_APP_TOKEN", "")

# ADMIN secret only YOU know. Protects RP-code issuing. Never ships in the app.
# Generate one with:  python -c "import secrets; print(secrets.token_urlsafe(32))"
VAGIS_ADMIN_TOKEN = os.environ.get("VAGIS_ADMIN_TOKEN", "")

MODEL = os.environ.get("VAGIS_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = int(os.environ.get("VAGIS_MAX_TOKENS", "1024"))

DATABASE_URL = os.environ.get("DATABASE_URL", "")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# Unambiguous alphabet for the random tail: no O, 0, I, 1, L.
CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

app = FastAPI(title="Vagis Agent Server")


# --------------------------------------------------------------------------
# Enrollment code helpers  (pure functions -- unit tested)
# --------------------------------------------------------------------------
def make_rp_code(seq: int) -> str:
    if not (1 <= seq <= 999):
        raise ValueError("RP sequence out of range (1-999)")
    return f"RP{seq:03d}"


def make_se_code(rp_seq: int, user_seq: int) -> str:
    if not (1 <= rp_seq <= 999):
        raise ValueError("RP sequence out of range (1-999)")
    if not (1 <= user_seq <= 9999):
        raise ValueError("user sequence out of range (1-9999)")
    tail = "".join(secrets.choice(CODE_ALPHABET) for _ in range(3))
    return f"SE{rp_seq:03d}{user_seq:04d}{tail}"


def parse_rp_code(code: str) -> Optional[dict[str, Any]]:
    code = (code or "").strip().upper()
    if len(code) != 5 or not code.startswith("RP"):
        return None
    digits = code[2:5]
    if not digits.isdigit():
        return None
    return {"rp_code": code, "rp_seq": int(digits)}


def parse_se_code(code: str) -> Optional[dict[str, Any]]:
    code = (code or "").strip().upper()
    if len(code) != 12 or not code.startswith("SE"):
        return None
    rp_digits, user_digits, tail = code[2:5], code[5:9], code[9:12]
    if not rp_digits.isdigit() or not user_digits.isdigit():
        return None
    if any(c not in CODE_ALPHABET for c in tail):
        return None
    return {
        "se_code": code,
        "rp_code": f"RP{rp_digits}",
        "rp_seq": int(rp_digits),
        "user_seq": int(user_digits),
        "tail": tail,
    }


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------
CREATE_UPLOADS_SQL = """
CREATE TABLE IF NOT EXISTS metric_uploads (
    id              SERIAL PRIMARY KEY,
    enrollment_code TEXT        NOT NULL,
    mode            TEXT        NOT NULL,
    filename        TEXT,
    csv_text        TEXT        NOT NULL,
    row_count       INTEGER,
    uploaded_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (enrollment_code, mode)
);
"""

CREATE_RESEARCHERS_SQL = """
CREATE TABLE IF NOT EXISTS researchers (
    rp_code    TEXT PRIMARY KEY,
    rp_seq     INTEGER NOT NULL UNIQUE,
    name       TEXT,
    email      TEXT,
    secret     TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

CREATE_SUBJECTS_SQL = """
CREATE TABLE IF NOT EXISTS subjects (
    se_code    TEXT PRIMARY KEY,
    rp_code    TEXT NOT NULL REFERENCES researchers(rp_code),
    user_seq   INTEGER NOT NULL,
    label      TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (rp_code, user_seq)
);
"""

UPSERT_UPLOAD_SQL = """
INSERT INTO metric_uploads (enrollment_code, mode, filename, csv_text, row_count, uploaded_at)
VALUES (%s, %s, %s, %s, %s, now())
ON CONFLICT (enrollment_code, mode)
DO UPDATE SET filename    = EXCLUDED.filename,
              csv_text    = EXCLUDED.csv_text,
              row_count   = EXCLUDED.row_count,
              uploaded_at = now()
RETURNING uploaded_at;
"""

LIST_UPLOADS_SQL = """
SELECT enrollment_code, mode, filename, row_count, uploaded_at
FROM metric_uploads
{where}
ORDER BY enrollment_code, mode;
"""


def db_connect():
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="Database not configured.")
    try:
        return psycopg2.connect(DATABASE_URL)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Database connection failed: {type(e).__name__}")


def ensure_tables(cur) -> None:
    cur.execute(CREATE_RESEARCHERS_SQL)  # referenced by subjects, create first
    cur.execute(CREATE_SUBJECTS_SQL)
    cur.execute(CREATE_UPLOADS_SQL)


@app.on_event("startup")
def init_db() -> None:
    if not DATABASE_URL:
        return
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
        conn.close()
    except Exception as e:
        print(f"[startup] could not initialise database: {type(e).__name__}: {e}")


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------
def check_app_auth(authorization: str | None) -> None:
    """App-level shared token (Authorization: Bearer <VAGIS_APP_TOKEN>)."""
    if not VAGIS_APP_TOKEN:
        raise HTTPException(status_code=500, detail="Server token not configured.")
    if authorization != f"Bearer {VAGIS_APP_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized.")


def check_admin_auth(authorization: str | None) -> None:
    """Admin token, only you (Authorization: Bearer <VAGIS_ADMIN_TOKEN>)."""
    if not VAGIS_ADMIN_TOKEN:
        raise HTTPException(status_code=500, detail="Admin token not configured.")
    if authorization != f"Bearer {VAGIS_ADMIN_TOKEN}":
        raise HTTPException(status_code=401, detail="Admin unauthorized.")


def authenticate_researcher(cur, rp_code: str, secret: str) -> dict[str, Any]:
    """Confirm rp_code + secret match a real researcher; return their row."""
    parsed = parse_rp_code(rp_code)
    if not parsed:
        raise HTTPException(status_code=400, detail="Malformed RP code.")
    cur.execute(
        "SELECT rp_code, rp_seq, secret FROM researchers WHERE rp_code = %s",
        (parsed["rp_code"],),
    )
    row = cur.fetchone()
    if not row or not secret or not secrets.compare_digest(row[2], secret):
        raise HTTPException(status_code=401, detail="Invalid researcher credentials.")
    return {"rp_code": row[0], "rp_seq": row[1]}


# --------------------------------------------------------------------------
# Request / response shapes
# --------------------------------------------------------------------------
class Turn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    mode: str = ""
    date: str = ""
    metrics: dict[str, dict[str, Any]] = Field(default_factory=dict)
    history_summary: str = ""
    conversation: list[Turn] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str


class IssueResearcherRequest(BaseModel):
    name: str = ""
    email: str = ""


class IssueSubjectRequest(BaseModel):
    rp_code: str
    secret: str
    label: str = ""          # optional private note, e.g. "pilot subject 3"


class ValidateRequest(BaseModel):
    se_code: str


# --------------------------------------------------------------------------
# System prompt
# --------------------------------------------------------------------------
def render_metrics(metrics: dict[str, dict[str, Any]]) -> str:
    if not metrics:
        return "No structured metrics were provided for this session."
    lines: list[str] = []
    for section, values in metrics.items():
        lines.append(f"## {section}")
        if isinstance(values, dict):
            for label, value in values.items():
                lines.append(f"- {label}: {value}")
        else:
            lines.append(f"- {values}")
        lines.append("")
    return "\n".join(lines).strip()


def build_system_prompt(req: ChatRequest) -> str:
    metrics_block = render_metrics(req.metrics)
    history_block = (
        req.history_summary.strip()
        if req.history_summary.strip()
        else "No prior-session summary was provided."
    )
    return f"""You are the personal health assistant inside the Vagis app. You help \
the user understand their own autonomic nervous system data, recorded from a smart \
ring, in plain and accessible language.

What you are:
- An educational tool that explains what the user's own numbers mean and what \
generally influences them.
- Grounded in the user's actual session data, shown below. Refer to their specific \
values when you answer -- be concrete, not generic.

What you are not:
- You are not a diagnostic or medical device. You do not diagnose conditions, \
interpret data as evidence of any specific disease, or recommend treatments, \
medication changes, or procedures.
- If the user describes symptoms, asks whether something is wrong with them, or asks \
a clinical question, explain the relevant physiology in general terms and suggest \
they discuss it with their physician. Do not speculate about diagnoses.

GROUND EVERYTHING IN THE DATA SHOWN BELOW -- THIS IS THE MOST IMPORTANT RULE:
- The metrics under THIS SESSION are the only metrics this app produces for the \
current view. Discuss ONLY these metrics and the general physiology behind them.
- If the user asks about a metric, score, or feature that is NOT in the data below, \
do not invent one. Say plainly that it isn't part of what the app shows for this \
session, and offer to discuss the metrics that ARE present instead. Never make up a \
metric name, a number, a formula, a threshold, or a normal range that is not given \
to you here.
- Never state a specific value for any metric unless that exact value appears in the \
data below. If you don't have a number, say you don't have it rather than estimating.
- It is always better to say "I don't have that" than to guess. Confident-sounding \
invention is the worst outcome and must be avoided.

EXPLAINING METRICS:
- You CAN and SHOULD explain, in plain language, what each metric shown below \
measures and what generally influences it -- this is one of your main jobs.
- Explain the concept and what it reflects about the body. Do NOT reveal or speculate \
about the internal calculation, formula, frequency bands, thresholds, or algorithm \
behind a Vagis metric. If asked how a metric is computed, describe what it represents \
and why it matters, not the math.
- When you explain a metric, connect it to the user's actual value for it where one \
is shown.

How to respond:
- Keep answers concise -- a few sentences unless the user asks for more detail.
- Use plain language. Define a term the first time you use it.
- Be warm and direct. The user is the expert on their own body and how they feel.
- Only discuss the data and physiology. If asked something unrelated, gently steer \
back to their health data.

--- THIS SESSION ---
Mode: {req.mode or "(not specified)"}
Date: {req.date or "(not specified)"}

{metrics_block}

--- RECENT HISTORY ---
{history_block}
"""


# --------------------------------------------------------------------------
# Endpoints: health + chat
# --------------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL,
        "anthropic_key_set": bool(ANTHROPIC_API_KEY),
        "app_token_set": bool(VAGIS_APP_TOKEN),
        "admin_token_set": bool(VAGIS_ADMIN_TOKEN),
        "database_url_set": bool(DATABASE_URL),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, authorization: str | None = Header(default=None)) -> ChatResponse:
    check_app_auth(authorization)
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="Anthropic key not configured.")
    if not req.conversation:
        raise HTTPException(status_code=400, detail="No conversation provided.")

    system_prompt = build_system_prompt(req)
    messages = [{"role": t.role, "content": t.content} for t in req.conversation]

    try:
        message = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=messages,
        )
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"Anthropic error: {e.status_code}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {type(e).__name__}")

    reply = "".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    ).strip()
    if not reply:
        raise HTTPException(status_code=502, detail="Empty reply from model.")

    return ChatResponse(reply=reply)


# --------------------------------------------------------------------------
# Endpoints: ingestion
# --------------------------------------------------------------------------
def count_csv_rows(text: str) -> int:
    reader = csv.reader(io.StringIO(text))
    n = sum(1 for _ in reader)
    return max(0, n - 1)


def subject_exists(cur, se_code: str) -> bool:
    cur.execute("SELECT 1 FROM subjects WHERE se_code = %s", (se_code,))
    return cur.fetchone() is not None


@app.post("/ingest")
async def ingest(
    enrollment_code: str = Form(...),
    mode: str = Form(...),
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Store one cumulative CSV for a subject + mode. Re-upload replaces prior."""
    check_app_auth(authorization)

    parsed = parse_se_code(enrollment_code)
    if not parsed:
        raise HTTPException(
            status_code=400,
            detail="enrollment_code must be a valid SE code (e.g. SE0140001Q4J).",
        )
    code = parsed["se_code"]

    mode_clean = (mode or "").strip().lower()
    if not mode_clean:
        raise HTTPException(status_code=400, detail="mode is required.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")
    try:
        csv_text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 text CSV.")

    row_count = count_csv_rows(csv_text)

    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            # The code must have been issued through the registry first.
            if not subject_exists(cur, code):
                raise HTTPException(
                    status_code=404,
                    detail="Unknown enrollment code. It must be issued before uploading.",
                )
            cur.execute(
                UPSERT_UPLOAD_SQL,
                (code, mode_clean, file.filename, csv_text, row_count),
            )
            uploaded_at = cur.fetchone()[0]
    finally:
        conn.close()

    return {
        "status": "ok",
        "enrollment_code": code,
        "mode": mode_clean,
        "filename": file.filename,
        "row_count": row_count,
        "uploaded_at": uploaded_at.isoformat(),
    }


@app.get("/ingest/list")
def ingest_list(
    enrollment_code: str | None = None,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    check_app_auth(authorization)
    params: tuple[Any, ...] = ()
    where = ""
    if enrollment_code:
        where = "WHERE enrollment_code = %s"
        params = (enrollment_code.strip().upper(),)

    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            cur.execute(LIST_UPLOADS_SQL.format(where=where), params)
            rows = cur.fetchall()
    finally:
        conn.close()

    items = [
        {
            "enrollment_code": r[0],
            "mode": r[1],
            "filename": r[2],
            "row_count": r[3],
            "uploaded_at": r[4].isoformat() if r[4] else None,
        }
        for r in rows
    ]
    return {"count": len(items), "items": items}


# --------------------------------------------------------------------------
# Endpoints: registry (admin issues RP, researcher issues SE, app validates SE)
# --------------------------------------------------------------------------
@app.post("/admin/researchers")
def issue_researcher(
    req: IssueResearcherRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """ADMIN: create the next researcher. Returns their RP code and secret.
    Give the researcher BOTH -- the secret is how they issue subjects and, later,
    log into their portal. Store it safely; it is shown once here."""
    check_admin_auth(authorization)
    secret = secrets.token_urlsafe(24)

    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            cur.execute("SELECT COALESCE(MAX(rp_seq), 0) FROM researchers;")
            next_seq = cur.fetchone()[0] + 1
            if next_seq > 999:
                raise HTTPException(status_code=409, detail="RP capacity reached (999).")
            rp_code = make_rp_code(next_seq)
            cur.execute(
                "INSERT INTO researchers (rp_code, rp_seq, name, email, secret) "
                "VALUES (%s, %s, %s, %s, %s);",
                (rp_code, next_seq, req.name or None, req.email or None, secret),
            )
    finally:
        conn.close()

    return {"status": "ok", "rp_code": rp_code, "secret": secret,
            "name": req.name, "email": req.email}


@app.get("/admin/researchers")
def list_researchers(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """ADMIN: list researchers (no secrets returned)."""
    check_admin_auth(authorization)
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            cur.execute(
                "SELECT r.rp_code, r.name, r.email, r.created_at, "
                "(SELECT COUNT(*) FROM subjects s WHERE s.rp_code = r.rp_code) "
                "FROM researchers r ORDER BY r.rp_seq;"
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return {
        "count": len(rows),
        "items": [
            {"rp_code": r[0], "name": r[1], "email": r[2],
             "created_at": r[3].isoformat() if r[3] else None, "subjects": r[4]}
            for r in rows
        ],
    }


@app.post("/portal/subjects")
def issue_subject(
    req: IssueSubjectRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """RESEARCHER: issue the next SE code under their own RP. Auth = rp_code + secret.
    (The Authorization header still carries the app token, so the endpoint isn't open.)"""
    check_app_auth(authorization)

    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            researcher = authenticate_researcher(cur, req.rp_code, req.secret)
            cur.execute(
                "SELECT COALESCE(MAX(user_seq), 0) FROM subjects WHERE rp_code = %s;",
                (researcher["rp_code"],),
            )
            next_user = cur.fetchone()[0] + 1
            if next_user > 9999:
                raise HTTPException(status_code=409, detail="Subject capacity reached (9999).")
            se_code = make_se_code(researcher["rp_seq"], next_user)
            cur.execute(
                "INSERT INTO subjects (se_code, rp_code, user_seq, label) "
                "VALUES (%s, %s, %s, %s);",
                (se_code, researcher["rp_code"], next_user, req.label or None),
            )
    finally:
        conn.close()

    return {"status": "ok", "se_code": se_code, "rp_code": researcher["rp_code"],
            "user_seq": next_user}


@app.get("/portal/subjects")
def list_subjects(
    rp_code: str,
    secret: str,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """RESEARCHER: list their own subjects (auth = rp_code + secret as query params)."""
    check_app_auth(authorization)
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            researcher = authenticate_researcher(cur, rp_code, secret)
            cur.execute(
                "SELECT se_code, user_seq, label, created_at FROM subjects "
                "WHERE rp_code = %s ORDER BY user_seq;",
                (researcher["rp_code"],),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return {
        "rp_code": researcher["rp_code"],
        "count": len(rows),
        "items": [
            {"se_code": r[0], "user_seq": r[1], "label": r[2],
             "created_at": r[3].isoformat() if r[3] else None}
            for r in rows
        ],
    }


@app.post("/portal/validate")
def validate_subject(
    req: ValidateRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """APP: confirm an SE code is well-formed AND issued. Returns its RP if valid.
    The app calls this when a user enters their code, before storing it."""
    check_app_auth(authorization)
    parsed = parse_se_code(req.se_code)
    if not parsed:
        return {"valid": False, "reason": "malformed"}

    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            cur.execute(
                "SELECT rp_code FROM subjects WHERE se_code = %s;",
                (parsed["se_code"],),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        return {"valid": False, "reason": "not_issued"}
    return {"valid": True, "se_code": parsed["se_code"], "rp_code": row[0]}


# --------------------------------------------------------------------------
# Admin page  (private tool, NO JavaScript -- plain HTML forms that POST)
# --------------------------------------------------------------------------
# GET  /admin              -> the page (token + create form + list form)
# POST /admin/ui/create    -> creates a researcher, shows the result page
# POST /admin/ui/list      -> shows the researcher list page
# The admin token is a normal form field; the server checks it. Because these are
# real <form> submits, they work even if the browser blocks scripts.
def _admin_style() -> str:
    return """
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 760px; margin: 0 auto; padding: 24px; color: #1a1a1a; background: #fafafa; }
  h1 { font-size: 22px; font-weight: 600; margin: 0 0 4px; }
  .sub { color: #666; font-size: 14px; margin: 0 0 24px; }
  .card { background: #fff; border: 1px solid #e4e4e4; border-radius: 12px; padding: 20px; margin-bottom: 20px; }
  .card h2 { font-size: 16px; font-weight: 600; margin: 0 0 14px; }
  label { display: block; font-size: 13px; color: #444; margin: 12px 0 4px; }
  input { width: 100%; padding: 10px 12px; font-size: 15px; border: 1px solid #d0d0d0;
          border-radius: 8px; background: #fff; }
  button { margin-top: 16px; padding: 10px 18px; font-size: 15px; font-weight: 500; color: #fff;
           background: #0f6e56; border: none; border-radius: 8px; cursor: pointer; }
  button.secondary { background: #444; }
  .result { margin: 0 0 20px; padding: 16px; border-radius: 8px; background: #e1f5ee; border: 1px solid #9fe1cb; }
  .result .row { display: flex; justify-content: space-between; padding: 6px 0; font-size: 15px; }
  .result .k { color: #085041; font-weight: 500; }
  .result .v { font-family: ui-monospace, Menlo, monospace; font-size: 16px; }
  .warn { color: #854f0b; font-size: 13px; margin-top: 10px; }
  .err { margin: 0 0 20px; padding: 16px; border-radius: 8px; background: #fcebeb; border: 1px solid #f7c1c1; color: #a32d2d; }
  table { width: 100%; border-collapse: collapse; margin-top: 6px; font-size: 14px; }
  th, td { text-align: left; padding: 8px 6px; border-bottom: 1px solid #eee; }
  th { color: #666; font-weight: 500; font-size: 12px; text-transform: uppercase; }
  td.mono { font-family: ui-monospace, Menlo, monospace; }
  .muted { color: #999; font-size: 13px; }
  a.back { display: inline-block; margin-top: 8px; color: #0f6e56; font-size: 14px; }
</style>
"""


def _admin_page(token: str = "", banner: str = "") -> str:
    tok_val = (token or "").replace('"', "&quot;")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vagis Admin</title>{_admin_style()}</head>
<body>
  <h1>Vagis Admin</h1>
  <p class="sub">Create researcher accounts and view the list. Private tool.</p>
  {banner}
  <div class="card">
    <h2>Create a researcher</h2>
    <form method="post" action="/admin/ui/create">
      <label>Admin token</label>
      <input name="token" type="password" placeholder="Paste your VAGIS_ADMIN_TOKEN" value="{tok_val}" autocomplete="off">
      <label>Name (optional)</label>
      <input name="name" type="text" placeholder="Dr. Jane Smith">
      <label>Email (optional)</label>
      <input name="email" type="text" placeholder="jane@example.com">
      <button type="submit">Create researcher</button>
    </form>
  </div>
  <div class="card">
    <h2>Researchers</h2>
    <form method="post" action="/admin/ui/list">
      <label>Admin token</label>
      <input name="token" type="password" placeholder="Paste your VAGIS_ADMIN_TOKEN" value="{tok_val}" autocomplete="off">
      <button type="submit" class="secondary">Show list</button>
    </form>
  </div>
</body></html>"""


@app.get("/admin", response_class=HTMLResponse)
def admin_page() -> HTMLResponse:
    return HTMLResponse(_admin_page())


@app.post("/admin/ui/create", response_class=HTMLResponse)
def admin_ui_create(
    token: str = Form(""),
    name: str = Form(""),
    email: str = Form(""),
) -> HTMLResponse:
    # Trim whitespace so a stray space in the token never blocks you.
    if token.strip() != (VAGIS_ADMIN_TOKEN or "").strip() or not VAGIS_ADMIN_TOKEN:
        banner = '<div class="err">Admin token did not match. Check it and try again.</div>'
        return HTMLResponse(_admin_page(token, banner))

    secret = secrets.token_urlsafe(24)
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            cur.execute("SELECT COALESCE(MAX(rp_seq), 0) FROM researchers;")
            next_seq = cur.fetchone()[0] + 1
            if next_seq > 999:
                return HTMLResponse(_admin_page(token, '<div class="err">RP capacity reached (999).</div>'))
            rp_code = make_rp_code(next_seq)
            cur.execute(
                "INSERT INTO researchers (rp_code, rp_seq, name, email, secret) VALUES (%s,%s,%s,%s,%s);",
                (rp_code, next_seq, name or None, email or None, secret),
            )
    finally:
        conn.close()

    banner = (
        '<div class="result">'
        f'<div class="row"><span class="k">Researcher code (RP)</span><span class="v">{rp_code}</span></div>'
        f'<div class="row"><span class="k">Secret</span><span class="v">{secret}</span></div>'
        '<div class="warn">Email BOTH to the researcher and save them now &mdash; '
        'the list never shows the secret again.</div></div>'
    )
    return HTMLResponse(_admin_page(token, banner))


@app.post("/admin/ui/list", response_class=HTMLResponse)
def admin_ui_list(token: str = Form("")) -> HTMLResponse:
    if token.strip() != (VAGIS_ADMIN_TOKEN or "").strip() or not VAGIS_ADMIN_TOKEN:
        return HTMLResponse(_admin_page(token, '<div class="err">Admin token did not match.</div>'))

    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            cur.execute(
                "SELECT r.rp_code, r.name, r.email, r.created_at, "
                "(SELECT COUNT(*) FROM subjects s WHERE s.rp_code = r.rp_code) "
                "FROM researchers r ORDER BY r.rp_seq;"
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        table = '<p class="muted">No researchers yet.</p>'
    else:
        body = "".join(
            f'<tr><td class="mono">{r[0]}</td><td>{r[1] or ""}</td><td>{r[2] or ""}</td>'
            f'<td>{r[4]}</td><td>{r[3].isoformat()[:10] if r[3] else ""}</td></tr>'
            for r in rows
        )
        table = (
            '<table><thead><tr><th>RP code</th><th>Name</th><th>Email</th>'
            '<th>Subjects</th><th>Created</th></tr></thead><tbody>' + body + '</tbody></table>'
        )
    banner = f'<div class="card"><h2>Researchers ({len(rows)})</h2>{table}' \
             f'<a class="back" href="/admin">&larr; Back</a></div>'
    return HTMLResponse(_admin_page(token, banner))
