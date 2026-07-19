"""
Vagis backend — agent relay + two-system data pipeline (research + clinical).

Two parallel systems, one server, told apart by code prefix:

                        RESEARCH                    CLINICAL (physician)
  Provider code         RES001                      PHY001
  Person code           SE0010001K3P (subject)      PT0010001K3P (patient)
  Data retention        persistent (study data)     ephemeral (auto-purged 48h)
  Stored in             research_uploads            clinical_holds
  Governed by           study protocol + consent    individual review, no keep

The person-code prefix routes the data: SE -> persistent research store;
PT -> ephemeral clinical hold that self-deletes 48h after upload. The two live
in separate tables so clinical data physically cannot land in the persistent
store.

Endpoints (foundation):
  GET  /health                 -- status
  POST /chat                   -- agent relay (unchanged)
  POST /ingest                 -- app uploads a CSV; routed by code prefix
  POST /portal/validate        -- app checks an SE or PT code is real
  GET  /admin                  -- create providers (research or clinical)
  GET  /portal                 -- provider login (RES -> research, PHY -> clinical)
"""

from __future__ import annotations

import asyncio
import csv
import io
import os
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import anthropic
import psycopg2
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
VAGIS_APP_TOKEN = os.environ.get("VAGIS_APP_TOKEN", "")
VAGIS_ADMIN_TOKEN = os.environ.get("VAGIS_ADMIN_TOKEN", "")
MODEL = os.environ.get("VAGIS_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = int(os.environ.get("VAGIS_MAX_TOKENS", "1024"))
DATABASE_URL = os.environ.get("DATABASE_URL", "")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# Clinical (PT) uploads self-delete this many hours after they arrive.
CLINICAL_HOLD_HOURS = int(os.environ.get("VAGIS_CLINICAL_HOLD_HOURS", "48"))

# Unambiguous alphabet for the random tail: no O, 0, I, 1, L.
CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"

# Prefixes that define the two systems.
PROVIDER_PREFIX = {"research": "RES", "clinical": "PHY"}   # 3 chars each
PERSON_PREFIX   = {"research": "SE",  "clinical": "PT"}    # 2 chars each
KIND_BY_PROVIDER_PREFIX = {v: k for k, v in PROVIDER_PREFIX.items()}
KIND_BY_PERSON_PREFIX   = {v: k for k, v in PERSON_PREFIX.items()}

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
app = FastAPI(title="Vagis Server")


# --------------------------------------------------------------------------
# Enrollment code scheme  (pure functions -- unit tested)
# --------------------------------------------------------------------------
# Provider: <PREFIX 3> + <seq 3>              e.g. RES001 / PHY001
# Person  : <PREFIX 2> + <provider 3> + <person 4> + <tail 3> = 12
#           e.g. SE0010001K3P (research) / PT0010001K3P (clinical)
def make_provider_code(kind: str, seq: int) -> str:
    if kind not in PROVIDER_PREFIX:
        raise ValueError(f"unknown kind {kind!r}")
    if not (1 <= seq <= 999):
        raise ValueError("provider sequence out of range (1-999)")
    return f"{PROVIDER_PREFIX[kind]}{seq:03d}"


def make_person_code(kind: str, provider_seq: int, person_seq: int) -> str:
    if kind not in PERSON_PREFIX:
        raise ValueError(f"unknown kind {kind!r}")
    if not (1 <= provider_seq <= 999):
        raise ValueError("provider sequence out of range (1-999)")
    if not (1 <= person_seq <= 9999):
        raise ValueError("person sequence out of range (1-9999)")
    tail = "".join(secrets.choice(CODE_ALPHABET) for _ in range(3))
    return f"{PERSON_PREFIX[kind]}{provider_seq:03d}{person_seq:04d}{tail}"


def parse_provider_code(code: str) -> Optional[dict[str, Any]]:
    code = (code or "").strip().upper()
    if len(code) != 6:
        return None
    prefix, digits = code[:3], code[3:6]
    if prefix not in KIND_BY_PROVIDER_PREFIX or not digits.isdigit():
        return None
    return {"provider_code": code, "kind": KIND_BY_PROVIDER_PREFIX[prefix], "seq": int(digits)}


def parse_person_code(code: str) -> Optional[dict[str, Any]]:
    code = (code or "").strip().upper()
    if len(code) != 12:
        return None
    prefix = code[:2]
    if prefix not in KIND_BY_PERSON_PREFIX:
        return None
    prov_digits, person_digits, tail = code[2:5], code[5:9], code[9:12]
    if not prov_digits.isdigit() or not person_digits.isdigit():
        return None
    if any(c not in CODE_ALPHABET for c in tail):
        return None
    kind = KIND_BY_PERSON_PREFIX[prefix]
    return {
        "person_code": code,
        "kind": kind,
        "provider_code": f"{PROVIDER_PREFIX[kind]}{prov_digits}",
        "provider_seq": int(prov_digits),
        "person_seq": int(person_digits),
        "tail": tail,
    }


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
# providers: both research (RES) and clinical (PHY), told apart by `kind`.
CREATE_PROVIDERS_SQL = """
CREATE TABLE IF NOT EXISTS providers (
    provider_code TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,
    seq           INTEGER NOT NULL,
    name          TEXT,
    email         TEXT,
    secret        TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (kind, seq)
);
"""

# persons: subjects (SE, research) and patients (PT, clinical).
CREATE_PERSONS_SQL = """
CREATE TABLE IF NOT EXISTS persons (
    person_code   TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,
    provider_code TEXT NOT NULL REFERENCES providers(provider_code),
    person_seq    INTEGER NOT NULL,
    label         TEXT,
    email         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (provider_code, person_seq)
);
"""

# research_uploads: PERSISTENT. One row per (subject, mode); re-upload replaces.
CREATE_RESEARCH_UPLOADS_SQL = """
CREATE TABLE IF NOT EXISTS research_uploads (
    id           SERIAL PRIMARY KEY,
    person_code  TEXT NOT NULL,
    mode         TEXT NOT NULL,
    filename     TEXT,
    csv_text     TEXT NOT NULL,
    row_count    INTEGER,
    uploaded_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (person_code, mode)
);
"""

# clinical_holds: EPHEMERAL. Same shape plus expires_at; purged after it passes.
CREATE_CLINICAL_HOLDS_SQL = """
CREATE TABLE IF NOT EXISTS clinical_holds (
    id           SERIAL PRIMARY KEY,
    person_code  TEXT NOT NULL,
    mode         TEXT NOT NULL,
    filename     TEXT,
    csv_text     TEXT NOT NULL,
    row_count    INTEGER,
    uploaded_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at   TIMESTAMPTZ NOT NULL,
    UNIQUE (person_code, mode)
);
"""

UPSERT_RESEARCH_SQL = """
INSERT INTO research_uploads (person_code, mode, filename, csv_text, row_count, uploaded_at)
VALUES (%s, %s, %s, %s, %s, now())
ON CONFLICT (person_code, mode)
DO UPDATE SET filename=EXCLUDED.filename, csv_text=EXCLUDED.csv_text,
              row_count=EXCLUDED.row_count, uploaded_at=now()
RETURNING uploaded_at;
"""

UPSERT_CLINICAL_SQL = """
INSERT INTO clinical_holds (person_code, mode, filename, csv_text, row_count, uploaded_at, expires_at)
VALUES (%s, %s, %s, %s, %s, now(), %s)
ON CONFLICT (person_code, mode)
DO UPDATE SET filename=EXCLUDED.filename, csv_text=EXCLUDED.csv_text,
              row_count=EXCLUDED.row_count, uploaded_at=now(), expires_at=EXCLUDED.expires_at
RETURNING uploaded_at, expires_at;
"""


def db_connect():
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="Database not configured.")
    try:
        return psycopg2.connect(DATABASE_URL)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Database connection failed: {type(e).__name__}")


def ensure_tables(cur) -> None:
    cur.execute(CREATE_PROVIDERS_SQL)   # referenced by persons, first
    cur.execute(CREATE_PERSONS_SQL)
    cur.execute(CREATE_RESEARCH_UPLOADS_SQL)
    cur.execute(CREATE_CLINICAL_HOLDS_SQL)


def purge_expired(cur) -> int:
    """Delete clinical holds whose window has passed. Returns rows removed."""
    cur.execute("DELETE FROM clinical_holds WHERE expires_at < now();")
    return cur.rowcount


@app.on_event("startup")
def init_db() -> None:
    if not DATABASE_URL:
        return
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            purge_expired(cur)
        conn.close()
    except Exception as e:
        print(f"[startup] db init failed: {type(e).__name__}: {e}")


# Background sweeper: purge expired clinical holds even with no traffic.
def _purge_loop() -> None:
    while True:
        time.sleep(900)  # every 15 minutes
        if not DATABASE_URL:
            continue
        try:
            conn = psycopg2.connect(DATABASE_URL)
            with conn, conn.cursor() as cur:
                n = purge_expired(cur)
            conn.close()
            if n:
                print(f"[purge] removed {n} expired clinical hold(s)")
        except Exception as e:
            print(f"[purge] sweep failed: {type(e).__name__}: {e}")


@app.on_event("startup")
def start_purge_thread() -> None:
    if DATABASE_URL:
        threading.Thread(target=_purge_loop, daemon=True).start()


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------
def check_app_auth(authorization: str | None) -> None:
    if not VAGIS_APP_TOKEN:
        raise HTTPException(status_code=500, detail="Server token not configured.")
    if authorization != f"Bearer {VAGIS_APP_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized.")


def check_admin_auth(authorization: str | None) -> None:
    if not VAGIS_ADMIN_TOKEN:
        raise HTTPException(status_code=500, detail="Admin token not configured.")
    if authorization != f"Bearer {VAGIS_ADMIN_TOKEN}":
        raise HTTPException(status_code=401, detail="Admin unauthorized.")


def authenticate_provider(cur, provider_code: str, secret: str) -> Optional[dict[str, Any]]:
    """Return provider dict if code+secret match, else None."""
    parsed = parse_provider_code(provider_code)
    if not parsed:
        return None
    cur.execute("SELECT provider_code, kind, seq, name, secret FROM providers WHERE provider_code = %s;",
                (parsed["provider_code"],))
    row = cur.fetchone()
    if not row or not secret or not secrets.compare_digest(row[4], secret):
        return None
    return {"provider_code": row[0], "kind": row[1], "seq": row[2], "name": row[3]}


# --------------------------------------------------------------------------
# Chat models + system prompt  (unchanged from prior version)
# --------------------------------------------------------------------------
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
# Health + chat
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
        "clinical_hold_hours": CLINICAL_HOLD_HOURS,
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
# Ingestion  (routed by person-code prefix: SE -> persistent, PT -> ephemeral)
# --------------------------------------------------------------------------
def count_csv_rows(text: str) -> int:
    reader = csv.reader(io.StringIO(text))
    n = sum(1 for _ in reader)
    return max(0, n - 1)


def person_exists(cur, person_code: str, kind: str) -> bool:
    cur.execute("SELECT 1 FROM persons WHERE person_code = %s AND kind = %s;", (person_code, kind))
    return cur.fetchone() is not None


@app.post("/ingest")
async def ingest(
    enrollment_code: str = Form(...),
    mode: str = Form(...),
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Store one cumulative CSV. The code prefix routes it:
    SE -> research_uploads (persistent). PT -> clinical_holds (ephemeral, 48h)."""
    check_app_auth(authorization)

    parsed = parse_person_code(enrollment_code)
    if not parsed:
        raise HTTPException(status_code=400,
            detail="enrollment_code must be a valid SE or PT code.")
    code = parsed["person_code"]
    kind = parsed["kind"]

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
            purge_expired(cur)
            if not person_exists(cur, code, kind):
                raise HTTPException(status_code=404,
                    detail="Unknown enrollment code. It must be issued before uploading.")
            if kind == "research":
                cur.execute(UPSERT_RESEARCH_SQL,
                            (code, mode_clean, file.filename, csv_text, row_count))
                uploaded_at = cur.fetchone()[0]
                return {
                    "status": "ok", "system": "research", "enrollment_code": code,
                    "mode": mode_clean, "row_count": row_count,
                    "uploaded_at": uploaded_at.isoformat(), "retention": "persistent",
                }
            else:  # clinical -> ephemeral hold
                expires = datetime.now(timezone.utc) + timedelta(hours=CLINICAL_HOLD_HOURS)
                cur.execute(UPSERT_CLINICAL_SQL,
                            (code, mode_clean, file.filename, csv_text, row_count, expires))
                uploaded_at, expires_at = cur.fetchone()
                return {
                    "status": "ok", "system": "clinical", "enrollment_code": code,
                    "mode": mode_clean, "row_count": row_count,
                    "uploaded_at": uploaded_at.isoformat(),
                    "expires_at": expires_at.isoformat(),
                    "retention": f"ephemeral ({CLINICAL_HOLD_HOURS}h)",
                }
    finally:
        conn.close()


class ValidateRequest(BaseModel):
    enrollment_code: str


@app.post("/portal/validate")
def validate_person(req: ValidateRequest,
                    authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """App checks an SE or PT code is well-formed AND issued. Returns its provider."""
    check_app_auth(authorization)
    parsed = parse_person_code(req.enrollment_code)
    if not parsed:
        return {"valid": False, "reason": "malformed"}

    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            cur.execute("SELECT provider_code, kind FROM persons WHERE person_code = %s;",
                        (parsed["person_code"],))
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        return {"valid": False, "reason": "not_issued"}
    return {"valid": True, "enrollment_code": parsed["person_code"],
            "provider_code": row[0], "system": row[1]}


# --------------------------------------------------------------------------
# Admin JSON endpoints (create/list providers of either kind)
# --------------------------------------------------------------------------
class IssueProviderRequest(BaseModel):
    kind: str            # "research" or "clinical"
    name: str = ""
    email: str = ""


def _issue_provider(cur, kind: str, name: str, email: str) -> dict[str, Any]:
    if kind not in PROVIDER_PREFIX:
        raise HTTPException(status_code=400, detail="kind must be 'research' or 'clinical'.")
    secret = secrets.token_urlsafe(24)
    cur.execute("SELECT COALESCE(MAX(seq), 0) FROM providers WHERE kind = %s;", (kind,))
    next_seq = cur.fetchone()[0] + 1
    if next_seq > 999:
        raise HTTPException(status_code=409, detail="Provider capacity reached (999).")
    code = make_provider_code(kind, next_seq)
    cur.execute(
        "INSERT INTO providers (provider_code, kind, seq, name, email, secret) "
        "VALUES (%s,%s,%s,%s,%s,%s);",
        (code, kind, next_seq, name or None, email or None, secret))
    return {"provider_code": code, "kind": kind, "secret": secret}


@app.post("/admin/providers")
def issue_provider(req: IssueProviderRequest,
                   authorization: str | None = Header(default=None)) -> dict[str, Any]:
    check_admin_auth(authorization)
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            out = _issue_provider(cur, req.kind, req.name, req.email)
    finally:
        conn.close()
    return {"status": "ok", **out}


def _issue_person(cur, provider: dict[str, Any], label: str, email: str) -> dict[str, Any]:
    kind = provider["kind"]
    cur.execute("SELECT COALESCE(MAX(person_seq), 0) FROM persons WHERE provider_code = %s;",
                (provider["provider_code"],))
    next_person = cur.fetchone()[0] + 1
    if next_person > 9999:
        raise HTTPException(status_code=409, detail="Person capacity reached (9999).")
    code = make_person_code(kind, provider["seq"], next_person)
    cur.execute("INSERT INTO persons (person_code, kind, provider_code, person_seq, label, email) "
                "VALUES (%s,%s,%s,%s,%s,%s);",
                (code, kind, provider["provider_code"], next_person, label or None, email or None))
    return {"person_code": code, "person_seq": next_person}

# --------------------------------------------------------------------------
# Web pages  (NO JavaScript -- plain HTML forms)
# --------------------------------------------------------------------------
from html import escape as _esc
from urllib.parse import quote as _q


def _style() -> str:
    return """
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 980px; margin: 0 auto; padding: 24px; color: #1a1a1a; background: #fafafa; }
  h1 { font-size: 22px; font-weight: 600; margin: 0 0 4px; }
  h2 { font-size: 16px; font-weight: 600; margin: 0 0 14px; }
  .sub { color: #666; font-size: 14px; margin: 0 0 22px; }
  .card { background: #fff; border: 1px solid #e4e4e4; border-radius: 12px; padding: 20px; margin-bottom: 20px; }
  label { display: block; font-size: 13px; color: #444; margin: 12px 0 4px; }
  input, select { width: 100%; padding: 10px 12px; font-size: 15px; border: 1px solid #d0d0d0; border-radius: 8px; background: #fff; }
  button { margin-top: 10px; padding: 9px 16px; font-size: 14px; font-weight: 500; color: #fff;
           background: #0f6e56; border: none; border-radius: 8px; cursor: pointer; }
  button.secondary { background: #444; }
  button.small { padding: 6px 12px; font-size: 13px; margin: 0; }
  form.inline { display: inline; margin: 0; }
  .result { margin: 0 0 20px; padding: 16px; border-radius: 8px; background: #e1f5ee; border: 1px solid #9fe1cb; }
  .result .row { display: flex; justify-content: space-between; padding: 4px 0; font-size: 15px; }
  .result .k { color: #085041; font-weight: 500; }
  .result .v { font-family: ui-monospace, Menlo, monospace; font-size: 16px; }
  .warn { color: #854f0b; font-size: 13px; margin-top: 8px; }
  .err { margin: 0 0 20px; padding: 14px 16px; border-radius: 8px; background: #fcebeb; border: 1px solid #f7c1c1; color: #a32d2d; }
  table { width: 100%; border-collapse: collapse; margin-top: 6px; font-size: 14px; }
  th, td { text-align: left; padding: 9px 10px; border-bottom: 1px solid #eee; white-space: nowrap; }
  th { color: #666; font-weight: 600; font-size: 12px; text-transform: uppercase; background: #f4f4f4; position: sticky; top: 0; }
  td.mono, .mono { font-family: ui-monospace, Menlo, monospace; }
  .muted { color: #999; font-size: 13px; }
  .tablewrap { overflow-x: auto; border: 1px solid #eee; border-radius: 8px; }
  .backbtn { display: inline-block; margin-top: 8px; color: #0f6e56; font-size: 14px; background: none; padding: 0; border: none; cursor: pointer; }
  .pill { display: inline-block; font-size: 11px; color: #0f6e56; background: #e1f5ee; border-radius: 5px; padding: 2px 8px; margin-left: 6px; }
  .flag { display: inline-block; font-size: 11px; color: #7a3b00; background: #ffe6c7; border-radius: 5px; padding: 2px 8px; margin-left: 6px; font-weight: 600; }
  .subrow { display: flex; align-items: center; justify-content: space-between; padding: 12px 0; border-bottom: 1px solid #f0f0f0; }
  .subrow:last-child { border-bottom: none; }
  .badge { display:inline-block; font-size:11px; font-weight:600; padding:2px 9px; border-radius:6px; }
  .badge.res { background:#e1f5ee; color:#085041; }
  .badge.phy { background:#e6eefc; color:#1c458f; }
</style>
"""


def _hidden(provider_code: str, key: str) -> str:
    return (f'<input type="hidden" name="provider_code" value="{_esc(provider_code)}">'
            f'<input type="hidden" name="key" value="{_esc(key)}">')


def _mailto(email: str, subject: str, body: str, text: str) -> str:
    if not email:
        return ""
    href = f"mailto:{_q(email)}?subject={_q(subject)}&body={_q(body)}"
    return (f'<a href="{href}" style="display:inline-block;margin-top:10px;padding:9px 16px;'
            f'background:#0f6e56;color:#fff;border-radius:8px;font-size:14px;font-weight:500;'
            f'text-decoration:none;">{_esc(text)}</a>')


def _mode_label(m: str) -> str:
    return {"sleep": "Sleep", "rest": "Rest", "stand": "Stand", "breathwork": "Breathwork"}.get(m, m.capitalize())


# ---- Admin page ----------------------------------------------------------
def _admin_page(token: str = "", banner: str = "") -> str:
    tok = _esc(token)
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vagis Admin</title>{_style()}</head><body>
  <h1>Vagis Admin</h1>
  <p class="sub">Create provider accounts for the research and clinical systems.</p>
  {banner}
  <div class="card">
    <h2>Create a provider</h2>
    <form method="post" action="/admin/ui/create">
      <label>Admin token</label>
      <input name="token" type="password" placeholder="Your VAGIS_ADMIN_TOKEN" value="{tok}" autocomplete="off">
      <label>System</label>
      <select name="kind">
        <option value="research">Research  (RES &mdash; persistent study data)</option>
        <option value="clinical">Clinical  (PHY &mdash; ephemeral, 48h)</option>
      </select>
      <label>Name (optional)</label>
      <input name="name" type="text" placeholder="Dr. Jane Smith">
      <label>Provider email (required)</label>
      <input name="email" type="text" placeholder="jane@example.com">
      <button type="submit">Create provider</button>
    </form>
  </div>
  <div class="card">
    <h2>Providers</h2>
    <form method="post" action="/admin/ui/list">
      <label>Admin token</label>
      <input name="token" type="password" placeholder="Your VAGIS_ADMIN_TOKEN" value="{tok}" autocomplete="off">
      <button type="submit" class="secondary">Show list</button>
    </form>
  </div>
</body></html>"""


@app.get("/admin", response_class=HTMLResponse)
def admin_page() -> HTMLResponse:
    return HTMLResponse(_admin_page())


@app.post("/admin/ui/create", response_class=HTMLResponse)
def admin_ui_create(token: str = Form(""), kind: str = Form("research"),
                    name: str = Form(""), email: str = Form("")) -> HTMLResponse:
    if token.strip() != (VAGIS_ADMIN_TOKEN or "").strip() or not VAGIS_ADMIN_TOKEN:
        return HTMLResponse(_admin_page(token, '<div class="err">Admin token did not match.</div>'))
    if kind not in PROVIDER_PREFIX:
        return HTMLResponse(_admin_page(token, '<div class="err">Pick a valid system.</div>'))
    if not email.strip():
        return HTMLResponse(_admin_page(token, '<div class="err">A provider email is required.</div>'))

    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            out = _issue_provider(cur, kind, name, email.strip())
    finally:
        conn.close()

    sys_label = "research" if kind == "research" else "clinical"
    portal_word = "subjects" if kind == "research" else "patients"
    mail_body = (
        f"Hello,\n\nYou have been set up as a {sys_label} provider on Vagis.\n\n"
        f"Provider ID: {out['provider_code']}\nKey: {out['secret']}\n\n"
        f"Sign in to the portal with these to manage your {portal_word} and view shared data. "
        f"Keep the Key private.\n\nThanks."
    )
    badge = "res" if kind == "research" else "phy"
    banner = (
        '<div class="result">'
        f'<div class="row"><span class="k">Provider ID <span class="badge {badge}">{sys_label}</span></span>'
        f'<span class="v">{out["provider_code"]}</span></div>'
        f'<div class="row"><span class="k">Key</span><span class="v">{out["secret"]}</span></div>'
        f'<div class="row"><span class="k">Email</span><span class="v">{_esc(email.strip())}</span></div>'
        '<div class="warn">The key is never shown again after you leave this page.</div>'
        + _mailto(email.strip(), "Your Vagis provider access", mail_body, "Email this provider")
        + '</div>'
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
                "SELECT p.provider_code, p.kind, p.name, p.email, p.created_at, "
                "(SELECT COUNT(*) FROM persons x WHERE x.provider_code = p.provider_code) "
                "FROM providers p ORDER BY p.kind, p.seq;")
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        table = '<p class="muted">No providers yet.</p>'
    else:
        body = ""
        for code, kind, name, email, created, n in rows:
            badge = "res" if kind == "research" else "phy"
            body += (f'<tr><td class="mono">{_esc(code)}</td>'
                     f'<td><span class="badge {badge}">{_esc(kind)}</span></td>'
                     f'<td>{_esc(name or "")}</td><td>{_esc(email or "")}</td>'
                     f'<td>{n}</td><td>{created.isoformat()[:10] if created else ""}</td></tr>')
        table = ('<div class="tablewrap"><table><thead><tr><th>Provider ID</th><th>System</th>'
                 '<th>Name</th><th>Email</th><th>People</th><th>Created</th></tr></thead>'
                 f'<tbody>{body}</tbody></table></div>')
    banner = f'<div class="card"><h2>Providers ({len(rows)})</h2>{table}<a class="backbtn" href="/admin">&larr; Back</a></div>'
    return HTMLResponse(_admin_page(token, banner))


# ---- Portal --------------------------------------------------------------
def _portal_login(banner: str = "") -> str:
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vagis Provider Portal</title>{_style()}</head><body>
  <h1>Vagis Provider Portal</h1>
  <p class="sub">Sign in to manage your people and view shared data.</p>
  {banner}
  <div class="card">
    <h2>Sign in</h2>
    <form method="post" action="/portal/ui/dashboard">
      <label>Provider ID</label>
      <input name="provider_code" type="text" placeholder="e.g. RES001 or PHY001" autocomplete="off">
      <label>Key</label>
      <input name="key" type="password" placeholder="Your key" autocomplete="off">
      <button type="submit">Sign in</button>
    </form>
  </div>
</body></html>"""


@app.get("/portal", response_class=HTMLResponse)
def portal_login_page() -> HTMLResponse:
    return HTMLResponse(_portal_login())



def _research_style() -> str:
    return """
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 0; background: #f4f6f8; color: #1a2b34; }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 16px; height: 100vh; display:flex; flex-direction:column; }
  .topbar { display:flex; align-items:center; justify-content:space-between;
            background:#fff; border:0.5px solid #e2e6ea; border-radius:10px; padding:11px 16px; margin-bottom:12px; }
  .brand { display:flex; align-items:center; gap:9px; font-size:16px; font-weight:600; }
  .brand .logo { width:26px;height:26px;border-radius:6px;background:#1d6fa5;color:#fff;
                 display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:700; }
  .tag { font-size:11px;color:#0c447c;background:#e6f1fb;border-radius:5px;padding:2px 8px;font-weight:500; }
  .who { font-size:12px;color:#5f6b72; }
  .banner { background:#e1f5ee;border:1px solid #9fe1cb;border-radius:9px;padding:12px 14px;margin-bottom:12px;
            font-size:14px;color:#085041; }
  .banner .v { font-family:ui-monospace,Menlo,monospace;font-weight:600; }
  .grid { display:grid; grid-template-columns:130px 160px 1fr; gap:10px; align-items:stretch;
          flex:1; min-height:0; }
  .card { background:#fff; border:0.5px solid #e2e6ea; border-radius:10px; padding:11px; }
  .lbl { font-size:10px;font-weight:600;color:#5f6b72;text-transform:uppercase;letter-spacing:.3px;margin-bottom:8px; }
  .leftcol { display:flex; flex-direction:column; gap:10px; min-height:0; height:100%; }
  .leftcol .subjcard { flex:1; display:flex; flex-direction:column; min-height:0; }
  .subjlist { display:flex; flex-direction:column; gap:2px; flex:1; min-height:60px; overflow-y:auto; }
  .subj { font-family:ui-monospace,Menlo,monospace; font-size:11px; color:#3a4750;
          padding:4px 6px; border-radius:4px; cursor:pointer; user-select:none; }
  .subj:hover { background:#f0f4f7; }
  .subj.sel { background:#eef6fc; color:#12456e; font-weight:600; }
  .midcol { display:flex; flex-direction:column; min-height:0; height:100%; }
  .box { margin-bottom:9px; }
  .box.grow { flex:1; display:flex; flex-direction:column; margin-bottom:9px; min-height:0; }
  .box.grow:last-child { margin-bottom:0; }
  .box .hd { display:flex;align-items:center;justify-content:space-between;margin-bottom:6px; }
  .box .hd .name { font-size:11px;font-weight:600; }
  .box .hd .btns { display:flex;gap:4px; }
  .minibtn { font-size:10px; border:0.5px solid #ccd4da; background:#fff; border-radius:5px;
             padding:2px 6px; cursor:pointer; color:#3a4750; }
  .minibtn:hover { background:#f0f4f7; }
  .g1 { border:0.5px solid #cfe0ee; }
  .g2 { border:0.5px solid #d8e6d4; }
  .chip { font-family:ui-monospace,Menlo,monospace; font-size:10.5px; border-radius:4px;
          padding:3px 6px; margin-bottom:3px; display:flex; align-items:center; justify-content:space-between; }
  .g1 .chip { background:#f4f9fd; color:#12456e; }
  .g2 .chip { background:#f4faef; color:#2f5410; }
  .ind .chip { background:#f2f4f6; color:#2c3940; }
  .chip .rm { cursor:pointer; color:#c0392b; font-weight:700; margin-left:6px; }
  .groupbody { flex:1; min-height:90px; overflow-y:auto; border:1px dashed #dfe4e9;
               border-radius:6px; padding:6px; outline:none; }
  .groupbody:focus { border-color:#9fbdd8; }
  .groupbody:empty::before { content:"paste or add subjects"; font-size:10px; color:#b7c0c7; }
  .drop-sm { min-height:24px; border:1px dashed #dfe4e9; border-radius:6px; padding:5px; outline:none; }
  .drop-sm:focus { border-color:#9fbdd8; }
  .rosterbtn { width:100%; background:#fff; border:0.5px solid #cfe0ee; color:#1d6fa5;
               border-radius:6px; padding:7px; font-size:12px; cursor:pointer; margin-bottom:8px; }
  .rosterbtn:hover { background:#f4f9fd; }
  .agent { display:flex; flex-direction:column; min-height:0; height:100%; }
  .msgs { flex:1; overflow-y:auto; display:flex; flex-direction:column; gap:9px; padding:2px; min-height:0; }
  .msg { border-radius:9px; padding:10px 13px; font-size:13.5px; line-height:1.5; max-width:82%; white-space:pre-wrap; }
  .msg.user { background:#eef6fc; color:#12456e; align-self:flex-end; }
  .msg.bot { background:#f6f7f8; color:#2c3940; align-self:flex-start; }
  .msg.think { color:#93a0a8; font-style:italic; }
  .figwrap { padding:6px !important; background:#fff !important; border:0.5px solid #e2e6ea; max-width:92% !important; }
  .figimg { max-width:100%; border-radius:6px; display:block; }
  .dlrow { background:none !important; padding:2px !important; }
  .dlbtn { background:#1d6fa5; color:#fff; border:none; border-radius:8px; padding:9px 14px;
           font-size:12.5px; font-weight:500; cursor:pointer; }
  .dlbtn:hover { background:#185f90; }
  .dlbtn:disabled { opacity:.6; cursor:default; }
  .composer { display:flex; gap:8px; margin-top:11px; }
  .composer textarea { flex:1; border:0.5px solid #e2e6ea; border-radius:9px; padding:10px 12px;
          font-size:13.5px; font-family:inherit; resize:none; height:42px; }
  .send { width:42px; height:42px; background:#1d6fa5; border:none; border-radius:9px; color:#fff;
          font-size:18px; cursor:pointer; }
  .send:disabled { opacity:.5; cursor:default; }
  .issue { display:flex; flex-direction:column; gap:6px; }
  .issue input { border:0.5px solid #e2e6ea; border-radius:6px; padding:6px 8px; font-size:12px; }
  .issue button { background:#1d6fa5;color:#fff;border:none;border-radius:6px;padding:7px 11px;font-size:12px;cursor:pointer; }
  .hint { font-size:10px;color:#93a0a8;margin-top:6px;text-align:center; }
  .modebox { margin-top:0; }
  .modebox select { width:100%; border:0.5px solid #e2e6ea; border-radius:6px; padding:7px 8px;
                    font-size:12px; background:#fff; color:#2c3940; }
  .overlay { position:fixed; inset:0; background:rgba(20,30,40,.28); display:none;
             align-items:flex-start; justify-content:center; padding-top:40px; z-index:50; }
  .overlay.show { display:flex; }
  .rosterpanel { background:#fff; border:0.5px solid #d8dee3; border-radius:12px; width:86%;
                 max-width:640px; max-height:82vh; overflow:hidden; display:flex; flex-direction:column;
                 box-shadow:0 8px 26px rgba(0,0,0,.16); }
  .rosterhd { display:flex; align-items:center; justify-content:space-between; padding:13px 18px; border-bottom:0.5px solid #eceef1; }
  .rosterhd .title { font-size:15px; font-weight:600; color:#1a2b34; }
  .rosterhd .acts { display:flex; gap:8px; align-items:center; }
  .rosterhd .prbtn { font-size:12px; color:#1d6fa5; border:0.5px solid #cfe0ee; background:#fff;
                     border-radius:6px; padding:5px 11px; cursor:pointer; }
  .rosterhd .clbtn { font-size:18px; color:#93a0a8; cursor:pointer; background:none; border:none; }
  .rostermeta { font-size:11px; color:#5f6b72; padding:8px 18px 0; }
  .rosterbody { overflow-y:auto; padding:6px 18px 16px; }
  .rostertbl { width:100%; border-collapse:collapse; font-size:12.5px; }
  .rostertbl th { text-align:left; font-size:10px; color:#5f6b72; text-transform:uppercase; letter-spacing:.3px;
                  padding:9px 8px; border-bottom:1.5px solid #d5dbe0; position:sticky; top:0; background:#fff; }
  .rostertbl td { padding:8px; border-top:0.5px solid #eceef1; color:#2c3940; }
  .rostertbl td.mono { font-family:ui-monospace,Menlo,monospace; }
  @media print {
    body > .wrap { display:none !important; }
    .overlay { position:static; background:none; display:block; padding:0; }
    .rosterpanel { box-shadow:none; border:none; width:100%; max-width:100%; max-height:none; }
    .rosterhd .acts { display:none; }
  }
</style>
"""

_RESEARCH_BODY = r"""
<div class="wrap">
  <div class="topbar">
    <div class="brand"><span class="logo">V</span> Vagis Research Portal <span class="tag">research</span></div>
    <div class="who" id="who"></div>
  </div>
  <div id="banner"></div>
  <div class="grid">

    <div class="leftcol">
      <div class="card subjcard">
        <div class="lbl">Subjects</div>
        <div class="subjlist" id="subjlist"></div>
      </div>
      <div class="card addbox">
        <div class="lbl">Add subjects</div>
        <button type="button" class="rosterbtn" onclick="openRoster()">View subject roster</button>
        <form method="post" action="/portal/ui/issue" class="issue" id="issueForm">
          <input type="hidden" name="provider_code" id="pcField">
          <input type="hidden" name="key" id="keyField">
          <input type="text" name="email" placeholder="new subject email" style="width:100%">
          <input type="text" name="label" placeholder="label (optional)" style="width:100%">
          <button type="submit">Generate &amp; email code</button>
        </form>
      </div>
    </div>

    <div class="midcol">
      <div class="card box ind">
        <div class="hd"><span class="name">Individual</span>
          <span class="btns"><button class="minibtn" onclick="addSel('individual')">&rarr;</button>
          <button class="minibtn" onclick="clearBox('individual')">clear</button></span></div>
        <div id="individual"></div>
        <div class="drop drop-sm" data-box="individual" tabindex="0"></div>
      </div>
      <div class="card box g1 grow">
        <div class="hd"><span class="name" style="color:#185fa5">Group 1</span>
          <span class="btns"><button class="minibtn" onclick="addSel('group1')">&rarr;</button>
          <button class="minibtn" onclick="clearBox('group1')">clear</button></span></div>
        <div class="groupbody" id="group1" data-box="group1" tabindex="0"></div>
      </div>
      <div class="card box g2 grow">
        <div class="hd"><span class="name" style="color:#3b6d11">Group 2</span>
          <span class="btns"><button class="minibtn" onclick="addSel('group2')">&rarr;</button>
          <button class="minibtn" onclick="clearBox('group2')">clear</button></span></div>
        <div class="groupbody" id="group2" data-box="group2" tabindex="0"></div>
      </div>
      <div class="card modebox">
        <div class="lbl">Group comparison mode</div>
        <select id="modeSelect">
          <option value="sleep">Sleep</option>
          <option value="rest">Rest</option>
          <option value="stand">Stand</option>
          <option value="breathwork">Breathwork</option>
        </select>
      </div>
    </div>

    <div class="card agent">
      <div class="lbl">Analysis agent</div>
      <div class="msgs" id="msgs"></div>
      <div class="composer">
        <textarea id="input" placeholder="Ask the agent to analyze the individual or compare groups..."></textarea>
        <button class="send" id="sendBtn" onclick="send()">&uarr;</button>
      </div>
      <div class="hint">Preview: the agent is connected. Live statistics and figures are being added next.</div>
    </div>

  </div>
</div>

<div class="overlay" id="rosterOverlay" onclick="if(event.target===this)closeRoster()">
  <div class="rosterpanel">
    <div class="rosterhd">
      <span class="title" id="rosterTitle">Subject roster</span>
      <span class="acts">
        <button class="prbtn" onclick="window.print()">Print</button>
        <button class="clbtn" onclick="closeRoster()">&times;</button>
      </span>
    </div>
    <div class="rostermeta" id="rosterMeta"></div>
    <div class="rosterbody">
      <table class="rostertbl">
        <thead><tr><th>SE code</th><th>Email</th><th>Label</th><th>Enrolled</th></tr></thead>
        <tbody id="rosterRows"></tbody>
      </table>
    </div>
  </div>
</div>

<script>
const boxes = { individual: [], group1: [], group2: [] };
let selected = null;
const conversation = [];

document.getElementById('who').textContent = PROVIDER + (PROVNAME ? '  \u00b7  ' + PROVNAME : '');
document.getElementById('pcField').value = PROVIDER;
document.getElementById('keyField').value = KEY;

function knownCode(code) { return SUBJECTS.some(function(s){ return s.code === code; }); }

function renderList() {
  const el = document.getElementById('subjlist');
  el.innerHTML = '';
  SUBJECTS.forEach(function(s) {
    const d = document.createElement('div');
    d.className = 'subj' + (selected === s.code ? ' sel' : '');
    d.textContent = s.code;
    d.title = s.label || '';
    d.onclick = function(){ selected = (selected === s.code ? null : s.code); renderList(); };
    el.appendChild(d);
  });
}

function inOtherBox(code, box) {
  return Object.keys(boxes).some(function(b){ return b !== box && boxes[b].indexOf(code) !== -1; });
}

function addCode(box, code) {
  code = (code || '').trim().toUpperCase();
  if (!code || !knownCode(code)) return false;
  Object.keys(boxes).forEach(function(b){
    const i = boxes[b].indexOf(code);
    if (i !== -1) boxes[b].splice(i, 1);
  });
  if (box === 'individual') boxes.individual = [code];
  else if (boxes[box].indexOf(code) === -1) boxes[box].push(code);
  return true;
}

function addSel(box) {
  if (!selected) return;
  addCode(box, selected);
  selected = null;
  renderAll();
}

function removeCode(box, code) {
  const i = boxes[box].indexOf(code);
  if (i !== -1) boxes[box].splice(i, 1);
  renderAll();
}

function clearBox(box) { boxes[box] = []; renderAll(); }

function parsePaste(text) {
  return (text || '').split(/[\s,;]+/).map(function(x){ return x.trim().toUpperCase(); }).filter(Boolean);
}

function renderBox(box) {
  const el = document.getElementById(box);
  el.innerHTML = '';
  boxes[box].forEach(function(code) {
    const c = document.createElement('div');
    c.className = 'chip';
    const span = document.createElement('span');
    span.textContent = code;
    const rm = document.createElement('span');
    rm.className = 'rm';
    rm.textContent = '\u00d7';
    rm.onclick = function(){ removeCode(box, code); };
    c.appendChild(span); c.appendChild(rm);
    el.appendChild(c);
  });
}

function renderAll() { renderList(); ['individual','group1','group2'].forEach(renderBox); }

document.querySelectorAll('[data-box]').forEach(function(d) {
  d.addEventListener('paste', function(e) {
    e.preventDefault();
    const box = d.getAttribute('data-box');
    const text = (e.clipboardData || window.clipboardData).getData('text');
    parsePaste(text).forEach(function(code){ addCode(box, code); });
    renderAll();
  });
});

function groupsPayload() {
  return { individual: boxes.individual.slice(), group1: boxes.group1.slice(), group2: boxes.group2.slice() };
}

function esc(s) { return String(s == null ? '' : s).replace(/[&<>]/g, function(c){ return {'&':'&amp;','<':'&lt;','>':'&gt;'}[c]; }); }

function openRoster() {
  const rows = document.getElementById('rosterRows');
  rows.innerHTML = '';
  SUBJECTS.forEach(function(s) {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td class="mono">' + esc(s.code) + '</td><td>' + esc(s.email || '\u2014') +
                   '</td><td>' + esc(s.label || '\u2014') + '</td><td>' + esc(s.enrolled || '\u2014') + '</td>';
    rows.appendChild(tr);
  });
  const today = new Date().toISOString().slice(0, 10);
  document.getElementById('rosterTitle').textContent = 'Subject roster \u00b7 ' + SUBJECTS.length + ' subjects';
  document.getElementById('rosterMeta').textContent =
    PROVIDER + (PROVNAME ? ' \u00b7 ' + PROVNAME : '') + ' \u00b7 generated ' + today;
  document.getElementById('rosterOverlay').classList.add('show');
}

function closeRoster() { document.getElementById('rosterOverlay').classList.remove('show'); }

function addMsg(role, text, cls) {
  const m = document.createElement('div');
  m.className = 'msg ' + (cls || role);
  m.textContent = text;
  document.getElementById('msgs').appendChild(m);
  const box = document.getElementById('msgs');
  box.scrollTop = box.scrollHeight;
  return m;
}

function addFigure(fileId) {
  const wrap = document.createElement('div');
  wrap.className = 'msg bot figwrap';
  const img = document.createElement('img');
  img.className = 'figimg';
  img.alt = 'analysis figure';
  // Fetch the figure with auth, show as blob.
  fetch('/portal/agent/figure', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider_code: PROVIDER, key: KEY, file_id: fileId })
  }).then(function(r){ return r.ok ? r.blob() : null; })
    .then(function(b){ if (b) img.src = URL.createObjectURL(b); else wrap.remove(); })
    .catch(function(){ wrap.remove(); });
  wrap.appendChild(img);
  document.getElementById('msgs').appendChild(wrap);
  const box = document.getElementById('msgs');
  box.scrollTop = box.scrollHeight;
}

function addSummaryButton(replyText, figureIds) {
  const row = document.createElement('div');
  row.className = 'msg bot dlrow';
  const btn = document.createElement('button');
  btn.className = 'dlbtn';
  btn.textContent = 'Download summary (PDF)';
  btn.onclick = function() {
    btn.disabled = true; btn.textContent = 'Preparing...';
    fetch('/portal/agent/summary', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider_code: PROVIDER, key: KEY,
                             title: 'Vagis analysis summary', text: replyText, figure_ids: figureIds })
    }).then(function(r){ return r.ok ? r.blob() : null; })
      .then(function(b){
        btn.disabled = false; btn.textContent = 'Download summary (PDF)';
        if (!b) return;
        const url = URL.createObjectURL(b);
        const a = document.createElement('a');
        a.href = url; a.download = 'vagis_analysis_summary.pdf'; a.click();
        URL.revokeObjectURL(url);
      }).catch(function(){ btn.disabled = false; btn.textContent = 'Download summary (PDF)'; });
  };
  row.appendChild(btn);
  document.getElementById('msgs').appendChild(row);
  const box = document.getElementById('msgs');
  box.scrollTop = box.scrollHeight;
}

async function send() {
  const inp = document.getElementById('input');
  const msg = inp.value.trim();
  if (!msg) return;
  inp.value = '';
  const btn = document.getElementById('sendBtn');
  btn.disabled = true;
  addMsg('user', msg);
  conversation.push({ role: 'user', content: msg });
  const thinking = addMsg('bot', 'Analyzing...', 'bot think');
  let secs = 0;
  const ticker = setInterval(function(){
    secs += 5;
    thinking.textContent = 'Analyzing... (' + secs + 's — larger analyses can take a minute or two)';
  }, 5000);
  try {
    const res = await fetch('/portal/agent/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider_code: PROVIDER, key: KEY, message: msg,
                             history: conversation, groups: groupsPayload(),
                             mode: document.getElementById('modeSelect').value })
    });
    const data = await res.json();
    clearInterval(ticker); thinking.remove();
    const reply = (data && data.reply) ? data.reply : (data && data.detail ? data.detail : 'No response.');
    addMsg('bot', reply);
    conversation.push({ role: 'assistant', content: reply });
    const figs = (data && data.figures) ? data.figures : [];
    figs.forEach(addFigure);
    if (figs.length) addSummaryButton(reply, figs);
  } catch (e) {
    clearInterval(ticker); thinking.remove();
    addMsg('bot', 'Could not reach the agent: ' + e.message);
  } finally {
    btn.disabled = false;
  }
}

document.getElementById('input').addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

renderAll();
</script>
</body></html>
"""

def _research_dashboard(prov: dict, key: str, cur, banner: str = "") -> str:
    """Rich research analysis workspace: thin subject list, Individual/Group1/Group2
    boxes with click-to-add + paste, and a live agent chat panel."""
    import json as _json
    cur.execute("SELECT person_code, label, email, created_at FROM persons "
                "WHERE provider_code = %s AND kind='research' ORDER BY person_seq;",
                (prov["provider_code"],))
    rows = cur.fetchall()
    subjects = [{"code": r[0], "label": r[1] or "", "email": r[2] or "",
                 "enrolled": r[3].isoformat()[:10] if r[3] else ""} for r in rows]
    subjects_json = _json.dumps(subjects)
    prov_code = prov["provider_code"]
    name = prov.get("name") or ""

    head = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vagis Research Portal</title>{_research_style()}</head><body>
<script>
const SUBJECTS = {subjects_json};
const PROVIDER = {_json.dumps(prov_code)};
const KEY = {_json.dumps(key)};
const PROVNAME = {_json.dumps(name)};
</script>
"""
    return head + _RESEARCH_BODY


def _portal_dashboard(prov: dict, key: str, cur, banner: str = "") -> str:
    """Dispatch: research providers get the rich analysis workspace;
    clinical providers keep the plain dashboard for now."""
    if prov["kind"] == "research":
        return _research_dashboard(prov, key, cur, banner)
    return _plain_dashboard(prov, key, cur, banner)


def _plain_dashboard(prov: dict, key: str, cur, banner: str = "") -> str:
    kind = prov["kind"]
    is_research = kind == "research"
    word = "subject" if is_research else "patient"
    words = "subjects" if is_research else "patients"

    cur.execute("SELECT person_code, person_seq, label, email, created_at FROM persons "
                "WHERE provider_code = %s ORDER BY person_seq;", (prov["provider_code"],))
    people = cur.fetchall()

    # Which modes each person has, and (clinical) whether data is currently held.
    if is_research:
        cur.execute("SELECT person_code, mode FROM research_uploads WHERE person_code IN "
                    "(SELECT person_code FROM persons WHERE provider_code = %s);", (prov["provider_code"],))
    else:
        purge_expired(cur)
        cur.execute("SELECT person_code, mode FROM clinical_holds WHERE person_code IN "
                    "(SELECT person_code FROM persons WHERE provider_code = %s);", (prov["provider_code"],))
    modes_by: dict[str, list[str]] = {}
    for pc, m in cur.fetchall():
        modes_by.setdefault(pc, []).append(m)

    if people:
        rows = ""
        for pcode, seq, label, email, created in people:
            has = sorted(modes_by.get(pcode, []))
            if has:
                if is_research:
                    marks = "".join(f'<span class="pill">{_esc(_mode_label(m))}</span>' for m in has)
                else:
                    marks = ('<span class="flag">NEW DATA</span>'
                             + "".join(f'<span class="pill">{_esc(_mode_label(m))}</span>' for m in has))
            else:
                marks = '<span class="muted" style="margin-left:6px">no data</span>'
            meta = " &middot; ".join(x for x in [_esc(label) if label else "", _esc(email) if email else ""] if x)
            rows += (
                f'<div class="subrow"><div><span class="mono">{_esc(pcode)}</span>'
                f'{(" &middot; " + meta) if meta else ""}{marks}'
                f'<div class="muted">issued {created.isoformat()[:10] if created else ""}</div></div>'
                f'<form class="inline" method="post" action="/portal/ui/person">'
                f'{_hidden(prov["provider_code"], key)}'
                f'<input type="hidden" name="person_code" value="{_esc(pcode)}">'
                f'<button class="small" type="submit">View</button></form></div>'
            )
        people_block = rows
    else:
        people_block = f'<p class="muted">No {words} yet. Issue a code below to add your first.</p>'

    badge = "res" if is_research else "phy"
    retention_note = ("Study data is stored persistently for your protocol."
                      if is_research else
                      f"Patient data is held only briefly and auto-deletes {CLINICAL_HOLD_HOURS}h after the patient sends it.")

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vagis Portal</title>{_style()}</head><body>
  <h1>Vagis Portal <span class="badge {badge}">{_esc(kind)}</span></h1>
  <p class="sub">Signed in as <span class="mono">{_esc(prov["provider_code"])}</span>{(" &middot; " + _esc(prov["name"])) if prov.get("name") else ""} &middot; {retention_note}</p>
  {banner}
  <div class="card">
    <h2>Issue a new {word} code</h2>
    <p class="muted">Creates the next code under your account. Email it to the {word} to enrol them.</p>
    <form method="post" action="/portal/ui/issue">
      {_hidden(prov["provider_code"], key)}
      <label>{word.capitalize()} email (required)</label>
      <input name="email" type="text" placeholder="{word}@example.com">
      <label>Label (optional, private to you)</label>
      <input name="label" type="text" placeholder="e.g. pilot {word} 1">
      <button type="submit">Issue {word} code</button>
    </form>
  </div>
  <div class="card">
    <h2>Your {words} ({len(people)})</h2>
    {people_block}
  </div>
</body></html>"""


def _auth_provider(cur, provider_code: str, key: str):
    p = authenticate_provider(cur, provider_code, key)
    if p:
        # fetch name
        cur.execute("SELECT name FROM providers WHERE provider_code = %s;", (p["provider_code"],))
        row = cur.fetchone()
        p["name"] = row[0] if row else None
    return p


@app.post("/portal/ui/dashboard", response_class=HTMLResponse)
def portal_dashboard(provider_code: str = Form(""), key: str = Form("")) -> HTMLResponse:
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            p = _auth_provider(cur, provider_code, key)
            if not p:
                return HTMLResponse(_portal_login('<div class="err">Wrong Provider ID or Key.</div>'))
            page = _portal_dashboard(p, key, cur)
    finally:
        conn.close()
    return HTMLResponse(page)


@app.post("/portal/ui/issue", response_class=HTMLResponse)
def portal_issue(provider_code: str = Form(""), key: str = Form(""),
                 label: str = Form(""), email: str = Form("")) -> HTMLResponse:
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            p = _auth_provider(cur, provider_code, key)
            if not p:
                return HTMLResponse(_portal_login('<div class="err">Wrong Provider ID or Key.</div>'))
            word = "subject" if p["kind"] == "research" else "patient"
            if not email.strip():
                return HTMLResponse(_portal_dashboard(p, key, cur,
                    f'<div class="err">A {word} email is required.</div>'))
            try:
                out = _issue_person(cur, p, label, email.strip())
            except HTTPException as e:
                return HTMLResponse(_portal_dashboard(p, key, cur, f'<div class="err">{_esc(e.detail)}</div>'))
            mail_body = (
                f"Hello,\n\nYou've been invited to share your Vagis data. Your enrollment code is:\n\n"
                f"{out['person_code']}\n\n"
                f"Open the Vagis app, go to Data Share, enter this code, and follow the prompts. "
                f"Only your metric summaries are shared \u2014 your raw recordings stay on your phone.\n\nThanks."
            )
            banner = (
                '<div class="result">'
                f'<div class="row"><span class="k">New {word} code</span><span class="v">{_esc(out["person_code"])}</span></div>'
                f'<div class="row"><span class="k">Email</span><span class="v">{_esc(email.strip())}</span></div>'
                f'<div class="warn">Send this code to the {word}. They enter it in the app to share their data with you.</div>'
                + _mailto(email.strip(), "Your Vagis enrollment code", mail_body, f"Email this {word}")
                + '</div>'
            )
            page = _portal_dashboard(p, key, cur, banner)
    finally:
        conn.close()
    return HTMLResponse(page)


@app.post("/portal/ui/person", response_class=HTMLResponse)
def portal_person(provider_code: str = Form(""), key: str = Form(""),
                  person_code: str = Form("")) -> HTMLResponse:
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            p = _auth_provider(cur, provider_code, key)
            if not p:
                return HTMLResponse(_portal_login('<div class="err">Wrong Provider ID or Key.</div>'))
            pc = (person_code or "").strip().upper()
            cur.execute("SELECT provider_code, label FROM persons WHERE person_code = %s;", (pc,))
            row = cur.fetchone()
            if not row or row[0] != p["provider_code"]:
                word = "subject" if p["kind"] == "research" else "patient"
                return HTMLResponse(_portal_login(f'<div class="err">{word.capitalize()} not found under your account.</div>'))
            label = row[1]
            if p["kind"] == "research":
                cur.execute("SELECT mode, row_count, uploaded_at FROM research_uploads "
                            "WHERE person_code = %s ORDER BY mode;", (pc,))
                extra = ""
            else:
                purge_expired(cur)
                cur.execute("SELECT mode, row_count, uploaded_at, expires_at FROM clinical_holds "
                            "WHERE person_code = %s ORDER BY mode;", (pc,))
            uploads = cur.fetchall()

            if uploads:
                items = ""
                for u in uploads:
                    mode, rc, up = u[0], u[1], u[2]
                    up_s = up.isoformat()[:16].replace("T", " ") if up else ""
                    exp_note = ""
                    if p["kind"] == "clinical":
                        exp = u[3]
                        exp_s = exp.isoformat()[:16].replace("T", " ") if exp else ""
                        exp_note = f' &middot; auto-deletes {exp_s}'
                    items += (
                        f'<div class="subrow"><div><b>{_esc(_mode_label(mode))}</b>'
                        f'<div class="muted">{rc if rc is not None else "?"} sessions &middot; updated {up_s}{exp_note}</div></div>'
                        f'<form class="inline" method="post" action="/portal/ui/view">'
                        f'{_hidden(p["provider_code"], key)}'
                        f'<input type="hidden" name="person_code" value="{_esc(pc)}">'
                        f'<input type="hidden" name="mode" value="{_esc(mode)}">'
                        f'<button class="small" type="submit">Open</button></form></div>'
                    )
                block = items
            else:
                block = '<p class="muted">No data shared yet' + ('' if p["kind"] == "research" else ' (or it has expired)') + '.</p>'

            back = (f'<form class="inline" method="post" action="/portal/ui/dashboard">'
                    f'{_hidden(p["provider_code"], key)}'
                    f'<button class="backbtn" type="submit">&larr; Back</button></form>')
            word = "Subject" if p["kind"] == "research" else "Patient"
            page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{word} {_esc(pc)}</title>{_style()}</head><body>
  <h1>{word} <span class="mono">{_esc(pc)}</span></h1>
  <p class="sub">{_esc(label) if label else "No label"} &middot; {_esc(p["provider_code"])}</p>
  {back}
  <div class="card"><h2>Shared data</h2>{block}</div>
</body></html>"""
    finally:
        conn.close()
    return HTMLResponse(page)


@app.post("/portal/ui/view", response_class=HTMLResponse)
def portal_view(provider_code: str = Form(""), key: str = Form(""),
                person_code: str = Form(""), mode: str = Form("")) -> HTMLResponse:
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            p = _auth_provider(cur, provider_code, key)
            if not p:
                return HTMLResponse(_portal_login('<div class="err">Wrong Provider ID or Key.</div>'))
            pc = (person_code or "").strip().upper()
            md = (mode or "").strip().lower()
            cur.execute("SELECT provider_code FROM persons WHERE person_code = %s;", (pc,))
            own = cur.fetchone()
            if not own or own[0] != p["provider_code"]:
                return HTMLResponse(_portal_login('<div class="err">Not found under your account.</div>'))
            if p["kind"] == "research":
                cur.execute("SELECT csv_text FROM research_uploads WHERE person_code=%s AND mode=%s;", (pc, md))
            else:
                purge_expired(cur)
                cur.execute("SELECT csv_text FROM clinical_holds WHERE person_code=%s AND mode=%s;", (pc, md))
            row = cur.fetchone()
    finally:
        conn.close()

    back = (f'<form class="inline" method="post" action="/portal/ui/person">'
            f'{_hidden(provider_code, key)}'
            f'<input type="hidden" name="person_code" value="{_esc(person_code)}">'
            f'<button class="backbtn" type="submit">&larr; Back</button></form>')

    if not row:
        table = '<p class="muted">No data for this mode (it may have expired).</p>'
    else:
        reader = csv.reader(io.StringIO(row[0]))
        all_rows = list(reader)
        if not all_rows:
            table = '<p class="muted">File is empty.</p>'
        else:
            header, body_rows = all_rows[0], all_rows[1:]
            thead = "<tr>" + "".join(f"<th>{_esc(h)}</th>" for h in header) + "</tr>"
            tbody = "".join("<tr>" + "".join(f'<td class="mono">{_esc(c)}</td>' for c in r) + "</tr>" for r in body_rows)
            table = (f'<p class="muted">{len(body_rows)} sessions &middot; {len(header)} metrics &middot; view only</p>'
                     f'<div class="tablewrap"><table><thead>{thead}</thead><tbody>{tbody}</tbody></table></div>')

    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(_mode_label(md))}</title>{_style()}</head><body>
  <h1>{_esc(_mode_label(md))} data</h1>
  <p class="sub"><span class="mono">{_esc(person_code)}</span></p>
  {back}
  <div class="card">{table}</div>
</body></html>"""
    return HTMLResponse(page)


# --------------------------------------------------------------------------
# Research analysis agent  (Stage 1: live agent pipe, no code execution yet)
# --------------------------------------------------------------------------
class AgentChatRequest(BaseModel):
    provider_code: str
    key: str
    message: str
    history: list[dict[str, Any]] = Field(default_factory=list)
    groups: dict[str, list[str]] = Field(default_factory=dict)
    mode: str = ""            # group-comparison mode (sleep/rest/stand/breathwork)


# Rough budget for how much subject data to hand the agent as raw rows before
# falling back to computed summaries. ~4 chars/token; keep well under context.
DATA_CHAR_BUDGET = 220_000
RESEARCH_MODES = ["sleep", "rest", "stand", "breathwork"]


def _fetch_csv(cur, person_code: str, mode: str) -> Optional[str]:
    cur.execute("SELECT csv_text FROM research_uploads WHERE person_code=%s AND mode=%s;",
                (person_code, mode))
    row = cur.fetchone()
    return row[0] if row else None


def _summarize_csv(csv_text: str) -> str:
    """Compact per-column summary (n, mean, sd, min, max) for numeric columns —
    the graceful fallback when raw rows exceed the budget."""
    import csv as _csv, io as _io, math
    reader = list(_csv.reader(_io.StringIO(csv_text)))
    if len(reader) < 2:
        return "(no data rows)"
    header, rows = reader[0], reader[1:]
    lines = [f"n_sessions={len(rows)}"]
    for ci, col in enumerate(header):
        vals = []
        for r in rows:
            if ci < len(r):
                try:
                    vals.append(float(r[ci]))
                except (ValueError, TypeError):
                    pass
        if len(vals) >= 2:
            mean = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
            lines.append(f"{col}: n={len(vals)} mean={mean:.3g} sd={sd:.3g} "
                         f"min={min(vals):.3g} max={max(vals):.3g}")
    return "\n".join(lines)


def _build_data_context(cur, groups: dict[str, list[str]], mode: str) -> tuple[str, bool]:
    """Assemble the data block for the agent. Individual -> all 4 modes raw.
    Groups -> the chosen mode raw for each subject. Falls back to summaries if
    the raw payload would exceed DATA_CHAR_BUDGET. Returns (text, used_summaries)."""
    ind = groups.get("individual", []) or []
    g1 = groups.get("group1", []) or []
    g2 = groups.get("group2", []) or []
    mode = (mode or "").strip().lower()

    # Gather raw pieces first, measure, then decide raw vs summary.
    def subject_block(code: str, modes: list[str]) -> list[tuple[str, str, str]]:
        out = []
        for md in modes:
            csv_text = _fetch_csv(cur, code, md)
            if csv_text:
                out.append((code, md, csv_text))
        return out

    pieces: list[tuple[str, str, str]] = []  # (code, mode, csv_text)
    if ind:
        for code in ind:
            pieces += subject_block(code, RESEARCH_MODES)
    if (g1 or g2):
        cmp_modes = [mode] if mode in RESEARCH_MODES else []
        if cmp_modes:
            for code in g1 + g2:
                pieces += subject_block(code, cmp_modes)

    if not pieces:
        return ("", False)

    total_chars = sum(len(c) for _, _, c in pieces)
    use_summary = total_chars > DATA_CHAR_BUDGET

    sections = []
    def label(code: str) -> str:
        where = []
        if code in ind: where.append("Individual")
        if code in g1:  where.append("Group 1")
        if code in g2:  where.append("Group 2")
        return f"{code} [{', '.join(where)}]" if where else code

    for code, md, csv_text in pieces:
        body = _summarize_csv(csv_text) if use_summary else csv_text.strip()
        kind = "summary" if use_summary else "raw CSV"
        sections.append(f"--- {label(code)} · {md} · {kind} ---\n{body}")

    header = ("SUBJECT DATA (summaries — full raw data exceeded the size budget)"
              if use_summary else "SUBJECT DATA (raw session rows)")
    return (header + "\n\n" + "\n\n".join(sections), use_summary)


def _research_agent_system(groups: dict[str, list[str]], mode: str,
                           data_block: str, used_summary: bool) -> str:
    ind = groups.get("individual", []) or []
    g1 = groups.get("group1", []) or []
    g2 = groups.get("group2", []) or []
    sel = []
    if ind: sel.append(f"Individual: {', '.join(ind)}")
    if g1:  sel.append(f"Group 1: {', '.join(g1)}")
    if g2:  sel.append(f"Group 2: {', '.join(g2)}")
    if (g1 or g2) and mode:
        sel.append(f"Group comparison mode: {mode}")
    sel_block = "\n".join(sel) if sel else "No subjects are selected yet."

    base = (
        "You are the Vagis research analysis assistant, helping a researcher analyze "
        "autonomic-metric data collected from study subjects via a smart ring. You speak "
        "to a professional researcher, so be technical and precise.\n\n"
        "The researcher has currently selected:\n" + sel_block + "\n\n"
        "The researcher is responsible for which subject codes belong to which group.\n\n"
    )

    if data_block:
        cap = (
            "The selected subjects' data is provided below. For an Individual, all four "
            "recording modes are included; for a group comparison, the chosen mode is "
            "included for every subject.\n\n"
            "YOU HAVE A PYTHON CODE-EXECUTION TOOL. When the researcher asks for a "
            "statistical test, comparison, or figure, USE IT to compute real results on the "
            "data above (pandas, numpy, scipy, statsmodels, matplotlib, seaborn are "
            "available). Run exactly the analysis the researcher prescribes — they own the "
            "statistical choice; execute it faithfully rather than substituting your own.\n\n"
            "SPEED IS CRITICAL — WORK IN ONE PASS:\n"
            "- Do the ENTIRE analysis in a SINGLE code execution: parse the data, do any "
            "splitting/grouping, run the test, and generate the figure(s) all in one script. "
            "Do NOT run code to 'first look at' or 'examine' the data and then run more code — "
            "that wastes minutes. One script that does everything, then report.\n"
            "- Do NOT narrate what you are about to do (no 'let me first examine...', no 'now "
            "I'll run the stats'). Just run the one script, then give the results.\n"
            "- Make ONE focused figure unless the researcher explicitly asks for more. Keep it "
            "simple and fast to render.\n"
            "- If any assumption is needed (e.g. how to split), make the reasonable choice "
            "inside the single script and state it in your summary — don't stop to ask.\n\n"
            "HOW TO REPORT:\n"
            "- The researcher is a scientist who wants RESULTS, not code. NEVER show, print, "
            "or describe the code you ran. Give only a clear, plain-language summary.\n"
            "- Report the real computed numbers: test used, n per group, group means ± SD, "
            "the statistic, p-value, and an effect size where appropriate.\n"
            "- Save each figure as a PNG. Put the KEY STATISTICS ON THE FIGURE ITSELF where "
            "sensible (p-value, group means, error bars, n per group), with clear axis labels "
            "and a short title, so it is publication-usable and self-contained.\n"
            "- Do NOT fabricate. Only report what the computation produced. If the data can't "
            "support the requested test, say so plainly rather than forcing a result.\n"
        )
        if used_summary:
            cap += ("\nNote: the data was large, so per-column SUMMARIES (n, mean, sd, min, "
                    "max) are provided instead of raw rows. You can reason over these but "
                    "cannot run per-session tests on summarized data — say so if a test needs "
                    "raw rows.\n")
        return base + cap + "\n" + data_block
    else:
        return base + (
            "No subject data is loaded for this request (nothing selected, or the selected "
            "subjects have no uploaded data for the chosen mode). You can discuss study design "
            "and which tests would fit — but do NOT run code or fabricate numbers."
        )


CODE_EXEC_TOOL = {"type": "code_execution_20250825", "name": "code_execution"}
AGENT_MAX_TOKENS = int(os.environ.get("VAGIS_AGENT_MAX_TOKENS", "16000"))


def _extract_text(content) -> str:
    return "".join(b.text for b in content if getattr(b, "type", None) == "text").strip()


def _extract_figure_ids(content) -> list[str]:
    """Pull file_ids for any files the code execution created (e.g. saved PNGs)."""
    ids = []
    for b in content:
        if getattr(b, "type", None) == "bash_code_execution_tool_result":
            inner = getattr(b, "content", None)
            files = getattr(inner, "content", None) if inner is not None else None
            if files:
                for fb in files:
                    fid = getattr(fb, "file_id", None)
                    if fid:
                        ids.append(fid)
    return ids


@app.post("/portal/agent/chat")
def research_agent_chat(req: AgentChatRequest) -> dict[str, Any]:
    """Research analysis agent. Authenticated by provider_code+key. Stage 3: the
    agent runs real Python (code execution) on the selected subjects' data to
    compute prescribed statistics and generate figures. It reports plain-language
    results only (never code); figures are returned as downloadable images."""
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="Anthropic key not configured.")
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message.")

    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            prov = authenticate_provider(cur, req.provider_code, req.key)
            if not prov or prov["kind"] != "research":
                raise HTTPException(status_code=401, detail="Not authorized for the research agent.")
            owned = _owned_selection(cur, prov["provider_code"], req.groups)
            data_block, used_summary = _build_data_context(cur, owned, req.mode)
    finally:
        conn.close()

    system = _research_agent_system(owned, req.mode, data_block, used_summary)

    messages = []
    for turn in req.history[-20:]:
        role = turn.get("role")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    if not messages:
        messages = [{"role": "user", "content": req.message}]

    figure_ids: list[str] = []
    container_id = None
    text_parts: list[str] = []
    CONTINUE_REASONS = {"pause_turn", "tool_use"}

    # Tool-use loop: code execution runs server-side on Anthropic's side. A turn
    # that is still working reports stop_reason "pause_turn" or "tool_use"; we feed
    # the assistant turn back and continue until it reaches a final stop reason and
    # produces its closing text. Each call carries a timeout so nothing hangs.
    hit_token_limit = False
    try:
        for _i in range(10):  # safety bound on continuations
            kwargs = dict(model=MODEL, max_tokens=AGENT_MAX_TOKENS,
                          system=system, messages=messages, tools=[CODE_EXEC_TOOL])
            if container_id:
                kwargs["container"] = container_id
            m = client.with_options(timeout=170.0).messages.create(**kwargs)

            if getattr(m, "container", None):
                container_id = m.container.id
            figure_ids += _extract_figure_ids(m.content)
            t = _extract_text(m.content)
            if t:
                text_parts.append(t)

            if getattr(m, "stop_reason", None) == "max_tokens":
                hit_token_limit = True
                break
            if getattr(m, "stop_reason", None) in CONTINUE_REASONS:
                # Feed the assistant turn back verbatim so it can continue.
                messages.append({"role": "assistant", "content": m.content})
                continue
            break
    except anthropic.APITimeoutError:
        partial = "\n\n".join(p for p in text_parts if p).strip()
        note = ("The analysis is taking longer than expected and timed out. This can "
                "happen with larger multi-step analyses. Please try again, or ask for a "
                "more specific single test (e.g. name the exact metric and test).")
        reply = (partial + "\n\n" + note) if partial else note
        return {"reply": reply, "figures": figure_ids,
                "container": container_id, "has_analysis": bool(figure_ids), "timed_out": True}
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"Anthropic error: {e.status_code}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {type(e).__name__}")

    reply = "\n\n".join(p for p in text_parts if p).strip()
    if not reply:
        if hit_token_limit:
            reply = ("The analysis was too large to complete in one response. Please ask for "
                     "a more focused analysis — for example, one specific metric and one test "
                     "at a time.")
        elif figure_ids:
            reply = ("The analysis ran but didn't return a written summary. "
                     "Please try again, or rephrase the request more specifically.")
        else:
            reply = "(no reply)"
    return {"reply": reply, "figures": figure_ids,
            "container": container_id, "has_analysis": bool(figure_ids)}


def _owned_selection(cur, provider_code: str, groups: dict[str, list[str]]) -> dict[str, list[str]]:
    """Filter the selected codes down to subjects that truly belong to this provider,
    so a tampered request can't pull another researcher's data."""
    cur.execute("SELECT person_code FROM persons WHERE provider_code=%s AND kind='research';",
                (provider_code,))
    mine = {r[0] for r in cur.fetchall()}
    out = {}
    for box in ("individual", "group1", "group2"):
        codes = [c.strip().upper() for c in (groups.get(box) or [])]
        out[box] = [c for c in codes if c in mine]
    return out


def _provider_ok(provider_code: str, key: str) -> bool:
    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_tables(cur)
            prov = authenticate_provider(cur, provider_code, key)
            return bool(prov and prov["kind"] == "research")
    finally:
        conn.close()


class FigureRequest(BaseModel):
    provider_code: str
    key: str
    file_id: str


@app.post("/portal/agent/figure")
def research_agent_figure(req: FigureRequest):
    """Stream a figure the code execution produced, by its Files API id.
    Authenticated by provider key so figures aren't world-readable."""
    from fastapi.responses import Response
    if not _provider_ok(req.provider_code, req.key):
        raise HTTPException(status_code=401, detail="Not authorized.")
    try:
        data = client.beta.files.download(req.file_id)
        raw = data.read() if hasattr(data, "read") else bytes(data)
    except Exception:
        raise HTTPException(status_code=404, detail="Figure not available.")
    return Response(content=raw, media_type="image/png")


class SummaryRequest(BaseModel):
    provider_code: str
    key: str
    title: str = "Vagis analysis summary"
    text: str = ""
    figure_ids: list[str] = Field(default_factory=list)


@app.post("/portal/agent/summary")
def research_agent_summary(req: SummaryRequest):
    """Assemble a PDF summary (plain-language results + figures) for download.
    Built server-side from the agent's reply text and the figures it generated."""
    from fastapi.responses import Response
    if not _provider_ok(req.provider_code, req.key):
        raise HTTPException(status_code=401, detail="Not authorized.")

    imgs = []
    for fid in req.figure_ids[:12]:
        try:
            d = client.beta.files.download(fid)
            imgs.append(d.read() if hasattr(d, "read") else bytes(d))
        except Exception:
            pass

    try:
        pdf_bytes = _build_summary_pdf(req.title, req.text, imgs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not build summary: {type(e).__name__}")

    return Response(content=pdf_bytes, media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="vagis_analysis_summary.pdf"'})


def _build_summary_pdf(title: str, text: str, images: list[bytes]) -> bytes:
    """Compose a simple PDF: title, plain-language results, then figures."""
    import io as _io
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas as _canvas
    from datetime import datetime as _dt

    buf = _io.BytesIO()
    c = _canvas.Canvas(buf, pagesize=letter)
    W, H = letter
    margin = 0.9 * inch
    y = H - margin

    c.setFont("Helvetica-Bold", 15)
    c.drawString(margin, y, title[:90]); y -= 20
    c.setFont("Helvetica", 9)
    c.setFillGray(0.4)
    c.drawString(margin, y, "Generated by Vagis · " + _dt.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
    c.setFillGray(0); y -= 22

    c.setFont("Helvetica", 10.5)
    max_w = W - 2 * margin
    for para in (text or "").split("\n"):
        para = para.replace("**", "").rstrip()
        if not para:
            y -= 6; continue
        words = para.split(" ")
        line = ""
        for w in words:
            test = (line + " " + w).strip()
            if c.stringWidth(test, "Helvetica", 10.5) > max_w:
                c.drawString(margin, y, line); y -= 14; line = w
                if y < margin + 40:
                    c.showPage(); y = H - margin; c.setFont("Helvetica", 10.5)
            else:
                line = test
        if line:
            c.drawString(margin, y, line); y -= 14
            if y < margin + 40:
                c.showPage(); y = H - margin; c.setFont("Helvetica", 10.5)

    for img in images:
        try:
            ir = ImageReader(_io.BytesIO(img))
            iw, ih = ir.getSize()
            disp_w = max_w
            disp_h = disp_w * ih / iw
            if disp_h > H - 2 * margin:
                disp_h = H - 2 * margin
                disp_w = disp_h * iw / ih
            if y - disp_h < margin:
                c.showPage(); y = H - margin
            c.drawImage(ir, margin, y - disp_h, width=disp_w, height=disp_h,
                        preserveAspectRatio=True, mask="auto")
            y -= disp_h + 18
        except Exception:
            pass

    c.showPage(); c.save()
    return buf.getvalue()
