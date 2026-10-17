"""FastAPI routes. Glue only: check the input, call llm.py, save with db.py, return JSON."""

from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend import db
from backend.llm import PROVIDER_INFO, ChatError, call_llm

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


@app.get("/api/providers")
def get_providers():
    return PROVIDER_INFO


@app.post("/api/run")
def post_run(request: RunRequest):
    check_settings(request)
    # Everything that was asked is saved, whether the call works or not.
    saved = dict(
        provider=request.provider,
        model=PROVIDER_INFO[request.provider]["model"],
        system_prompt=request.system_prompt,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        question=request.question,
    )
    try:
        result = call_llm(
            request.provider, request.system_prompt, request.question,
            request.temperature, request.top_p, request.max_tokens,
        )
    except ChatError as error:
        row = db.insert_run(**saved, error=str(error))
        return JSONResponse(status_code=502, content={"detail": str(error), "run": row})

    saved.update(
        model=result.model,
        answer=result.answer,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        finish_reason=result.finish_reason,
        latency_ms=result.latency_ms,
    )
    return db.insert_run(**saved)


@app.get("/api/runs")
def get_runs(limit: int = Query(50, ge=1, le=500)):
    return db.list_runs(limit)
