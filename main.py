"""
Vagis backend — agent relay server + metric ingestion.

This is the always-on server that sits between the Vagis app and Claude.
The app never calls Anthropic directly and never holds the API key. Instead:

    Vagis app  ->  POST /chat (this server, holds the key)  ->  Anthropic API
    Vagis app  <-  reply                                    <-  Anthropic API

Why this exists:
  * The API key lives ONLY here, in an environment variable. It is never shipped
    in the app, so any tester's phone can use the agent without a key of their own.
  * The system prompt is built HERE, server-side. Updating the agent's behaviour
    or knowledge is a server edit + restart -- no app release. Every tester goes
    current immediately. (This is what fixes "stuck on an old version".)
  * The server does NOT hardcode Vagis metric names. The app sends whatever metrics
    it currently computes; the server formats them generically. Add a metric or a
    mode in the app and the agent reflects it automatically, with no server change.

This same server is also the ingestion backend for the research portal. The app
uploads cumulative CSV metric files (Sleep, Stand, Rest, Breathwork, ...) tagged
with the subject's enrollment code; the server stores them in Postgres so they
survive redeploys and can later be read by the portal.

    Vagis app  ->  POST /ingest (enrollment_code, mode, CSV)  ->  Postgres

Endpoints:
    GET  /health        -- server status
    POST /chat          -- personal agent relay
    POST /ingest        -- upload a cumulative CSV for one enrollment code + mode
    GET  /ingest/list   -- list what's been uploaded (verification / portal read)
"""

from __future__ import annotations

import csv
import io
import os
from typing import Any

import anthropic
import psycopg2
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Configuration  (all via environment variables -- nothing secret in the code)
# --------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# A shared secret the app sends in the Authorization header. This stops random
# people on the internet from hitting /chat and spending your Anthropic credits.
# Generate one with:  python -c "import secrets; print(secrets.token_urlsafe(32))"
VAGIS_APP_TOKEN = os.environ.get("VAGIS_APP_TOKEN", "")

# One line to switch models. Sonnet = best speed/cost for chat; swap to
# "claude-opus-4-8" for deeper reasoning at higher cost/latency.
MODEL = os.environ.get("VAGIS_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = int(os.environ.get("VAGIS_MAX_TOKENS", "1024"))

# Postgres connection string. On Render this is the Internal Database URL of
# vagis-db, set as the DATABASE_URL environment variable on this service.
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# Reject absurdly large uploads. Cumulative metric CSVs are tiny (a few KB);
# 10 MB is a generous ceiling that still blocks abuse.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

app = FastAPI(title="Vagis Agent Server")


# --------------------------------------------------------------------------
# Database helpers
# --------------------------------------------------------------------------
# One row per (enrollment_code, mode). Uploads are cumulative masters, so a new
# upload for the same code + mode REPLACES the previous one (upsert). csv_text
# holds the whole file verbatim -- ingestion stays schema-agnostic, so each mode
# can have completely different columns and nothing here needs to change.
CREATE_TABLE_SQL = """
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

UPSERT_SQL = """
INSERT INTO metric_uploads (enrollment_code, mode, filename, csv_text, row_count, uploaded_at)
VALUES (%s, %s, %s, %s, %s, now())
ON CONFLICT (enrollment_code, mode)
DO UPDATE SET filename    = EXCLUDED.filename,
              csv_text    = EXCLUDED.csv_text,
              row_count   = EXCLUDED.row_count,
              uploaded_at = now()
RETURNING uploaded_at;
"""

LIST_SQL = """
SELECT enrollment_code, mode, filename, row_count, uploaded_at
FROM metric_uploads
{where}
ORDER BY enrollment_code, mode;
"""


def db_connect():
    """Open a Postgres connection, or fail clearly if the server isn't wired up."""
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="Database not configured.")
    try:
        return psycopg2.connect(DATABASE_URL)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Database connection failed: {type(e).__name__}")


def ensure_table(cur) -> None:
    """Create the uploads table if it doesn't exist yet. Idempotent and cheap."""
    cur.execute(CREATE_TABLE_SQL)


@app.on_event("startup")
def init_db() -> None:
    """Best-effort table creation on boot. If the DB is down the agent still runs."""
    if not DATABASE_URL:
        return
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn, conn.cursor() as cur:
            ensure_table(cur)
        conn.close()
    except Exception as e:
        print(f"[startup] could not initialise database: {type(e).__name__}: {e}")


# --------------------------------------------------------------------------
# Request / response shapes
# --------------------------------------------------------------------------
class Turn(BaseModel):
    """One message in the running conversation."""
    role: str          # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    # What the user was recording. Free-form string -- whatever the app calls
    # the mode (Sleep, Stand, Rest, Breathwork, Circadian, ...). The server does
    # not need to know the set in advance.
    mode: str = ""

    # Human-readable session date, as the app already formats it.
    date: str = ""

    # The user's own metrics for the session they're looking at.
    # Flexible by design: a dict of section name -> { metric label: value }.
    # Example the app might send:
    #   {
    #     "HRV":   {"RMSSD": "42 ms", "%VLF": "31 %", "SDNN": "55 ms"},
    #     "Sleep": {"AMMA": "0.71", "PWAD index": "12.4"}
    #   }
    # Add/rename anything app-side; it renders here untouched.
    metrics: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # Optional short summary of recent sessions for longitudinal context.
    history_summary: str = ""

    # The running chat. The app keeps this and resends it each turn, because
    # the model itself is stateless.
    conversation: list[Turn] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str


# --------------------------------------------------------------------------
# System prompt  (built here -- the single source of the agent's behaviour)
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
# Auth
# --------------------------------------------------------------------------
def check_auth(authorization: str | None) -> None:
    """Reject anyone who doesn't send the shared app token."""
    if not VAGIS_APP_TOKEN:
        # Server misconfigured -- fail closed rather than run wide open.
        raise HTTPException(status_code=500, detail="Server token not configured.")
    expected = f"Bearer {VAGIS_APP_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized.")


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, Any]:
    """Quick check that the server is up and configured. Hit this in a browser."""
    return {
        "status": "ok",
        "model": MODEL,
        "anthropic_key_set": bool(ANTHROPIC_API_KEY),
        "app_token_set": bool(VAGIS_APP_TOKEN),
        "database_url_set": bool(DATABASE_URL),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, authorization: str | None = Header(default=None)) -> ChatResponse:
    check_auth(authorization)
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
    except Exception as e:  # network etc.
        raise HTTPException(status_code=502, detail=f"Upstream error: {type(e).__name__}")

    reply = "".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    ).strip()
    if not reply:
        raise HTTPException(status_code=502, detail="Empty reply from model.")

    return ChatResponse(reply=reply)


def normalise_enrollment_code(raw: str) -> str:
    """Trim and upper-case; require the RP-/SE- shape the app already uses."""
    code = (raw or "").strip().upper()
    if not code:
        raise HTTPException(status_code=400, detail="enrollment_code is required.")
    if not (code.startswith("RP-") or code.startswith("SE-")):
        raise HTTPException(
            status_code=400,
            detail="enrollment_code must start with 'RP-' or 'SE-'.",
        )
    return code


def count_csv_rows(text: str) -> int:
    """Data rows only (header excluded). Uses the csv module so quoted newlines
    inside a field don't inflate the count."""
    reader = csv.reader(io.StringIO(text))
    n = sum(1 for _ in reader)
    return max(0, n - 1)


@app.post("/ingest")
async def ingest(
    enrollment_code: str = Form(...),
    mode: str = Form(...),
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Receive one cumulative CSV for a given enrollment code + mode and store it.

    Multipart form fields:
        enrollment_code : RP-xxxx or SE-xxxx  (the subject's stored code)
        mode            : sleep | stand | rest | breathwork | ...  (free-form)
        file            : the cumulative CSV

    Cumulative semantics: re-uploading the same code + mode replaces the prior
    file for that pair. Any mode's columns are accepted as-is.
    """
    check_auth(authorization)

    code = normalise_enrollment_code(enrollment_code)
    mode_clean = (mode or "").strip().lower()
    if not mode_clean:
        raise HTTPException(status_code=400, detail="mode is required.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")

    try:
        csv_text = raw.decode("utf-8-sig")  # -sig strips a BOM if present
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 text CSV.")

    row_count = count_csv_rows(csv_text)

    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_table(cur)
            cur.execute(
                UPSERT_SQL,
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
    """List what's been uploaded (metadata only, no CSV bodies).

    Optional ?enrollment_code=SE-xxxx filters to one subject. This is the read
    the portal will build on, and it lets you verify uploads landed.
    """
    check_auth(authorization)

    params: tuple[Any, ...] = ()
    where = ""
    if enrollment_code:
        where = "WHERE enrollment_code = %s"
        params = (enrollment_code.strip().upper(),)

    conn = db_connect()
    try:
        with conn, conn.cursor() as cur:
            ensure_table(cur)
            cur.execute(LIST_SQL.format(where=where), params)
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
