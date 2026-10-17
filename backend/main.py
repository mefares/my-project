"""FastAPI routes. Glue only: check the input, call llm.py, save with db.py, return JSON."""

import json
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend import db
from backend.llm import PROVIDER_INFO, ChatError, call_llm, stream_llm

app = FastAPI(title="LLM playground")
db.init_db()


class RunRequest(BaseModel):
    provider: Literal["mistral", "gemini"]
    system_prompt: Optional[str] = None
    question: str = Field(min_length=1)
    # None means "do not send this setting", so the provider uses its own default.
    temperature: Optional[float] = Field(default=None, ge=0)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)
    max_tokens: Optional[int] = Field(default=None, ge=1)


def check_settings(request):
    """Reject combinations the provider would refuse, before spending a call."""
    limit = PROVIDER_INFO[request.provider]["temperature_max"]
    if request.temperature is not None and request.temperature > limit:
        raise HTTPException(422, f"{request.provider} temperature must be between 0 and {limit}.")
    # Found by testing: Mistral answers 400 "top_p must be 1 when using greedy sampling".
    if request.provider == "mistral" and request.temperature == 0 and request.top_p not in (None, 1):
        raise HTTPException(422, "Mistral: with temperature 0, top_p must be empty or 1.")


def asked_fields(request):
    """The part of a run that is known before the model answers."""
    return dict(
        provider=request.provider,
        model=PROVIDER_INFO[request.provider]["model"],
        system_prompt=request.system_prompt,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        question=request.question,
    )


def answer_fields(result):
    """The part of a run that comes from the model's answer (an LLMResult)."""
    return dict(
        model=result.model,
        answer=result.answer,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        finish_reason=result.finish_reason,
        latency_ms=result.latency_ms,
        ttft_ms=result.ttft_ms,
    )


@app.get("/api/providers")
def get_providers():
    return PROVIDER_INFO


@app.post("/api/run")
def post_run(request: RunRequest):
    check_settings(request)
    # Everything that was asked is saved, whether the call works or not.
    asked = asked_fields(request)
    try:
        result = call_llm(
            request.provider, request.system_prompt, request.question,
            request.temperature, request.top_p, request.max_tokens,
        )
    except ChatError as error:
        row = db.insert_run(**asked, error=str(error))
        return JSONResponse(status_code=502, content={"detail": str(error), "run": row})
    return db.insert_run(**{**asked, **answer_fields(result)})


def json_line(event):
    """One event as one line of JSON (the browser reads the stream line by line)."""
    return json.dumps(event) + "\n"


UNFINISHED = "Not finished: the connection was closed before the end of the answer."


def stream_events(request):
    """Send the answer piece by piece. The run is saved first, marked unfinished, and updated when the
    answer ends. If the browser leaves midway, the marker stays: the send is still recorded."""
    run = db.insert_run(**asked_fields(request), error=UNFINISHED)
    try:
        for event in stream_llm(
            request.provider, request.system_prompt, request.question,
            request.temperature, request.top_p, request.max_tokens,
        ):
            if event["type"] == "delta":
                yield json_line(event)
            else:
                run = db.update_run(run["id"], **answer_fields(event["result"]), error=None)
                yield json_line({"type": "done", "run": run})
    except ChatError as error:
        run = db.update_run(run["id"], error=str(error))
        yield json_line({"type": "error", "detail": str(error), "run": run})


@app.post("/api/run/stream")
def post_run_stream(request: RunRequest):
    check_settings(request)
    return StreamingResponse(stream_events(request), media_type="application/x-ndjson")


@app.get("/api/runs")
def get_runs(limit: int = Query(50, ge=1, le=500)):
    return db.list_runs(limit)
