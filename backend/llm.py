"""The only module that talks to the model providers (Mistral and Gemini).

The rest of the app calls call_llm() (whole answer at once) or stream_llm() (answer piece by piece)
and gets the same LLMResult back, whichever provider answers.
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

TIMEOUT = 60  # seconds to wait for a reply before giving up

# Pinned on purpose: "gemini-flash-latest" is an alias that moves to newer models over time.
PROVIDERS = {
    "mistral": {"env": "MISTRAL_API_KEY", "model": "ministral-8b-latest"},
    "gemini": {"env": "GEMINI_API_KEY", "model": "gemini-3.8-flash"},
}

# What the UI needs to know about each provider (sent by GET /api/providers).
# Ranges are from the official docs: Mistral temperature 0 to 1.5, Gemini 0.0 to 2.0, top_p 0 to 1.
# "ignored" lists settings the provider accepts but does not use (Gemini docs: temperature and topP
# are silently ignored on Gemini 3.6 Flash and newer models; gemini-3.8-flash is one of them).
PROVIDER_INFO = {
    "mistral": {"model": PROVIDERS["mistral"]["model"], "temperature_max": 1.5, "ignored": []},
    "gemini": {"model": PROVIDERS["gemini"]["model"], "temperature_max": 2.0, "ignored": ["temperature", "top_p"]},
}

MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:{method}"


class ChatError(Exception):
    """A readable error that is safe to print and store (it never contains the API key)."""


@dataclass
class LLMResult:
    answer: str
    input_tokens: int
    output_tokens: int
    finish_reason: str  # 'stop', 'length' or 'other'
    latency_ms: int
    model: str
    ttft_ms: Optional[int] = None  # time to first token; only measured when streaming


def redact(text, key):
    # Even if a server echoed the key back, it never gets printed or stored.
    return text.replace(key, "***")


def load_key(provider):
    """Read the API key from the .env file at the repo root."""
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    name = PROVIDERS[provider]["env"]
    key = os.environ.get(name, "").strip()
    if not key:
        raise ChatError(f"{name} is not set. Add it to your .env file.")
    return key


def read_json(response, key):
    """Return the parsed body, or raise ChatError with the server's message on failure."""
    try:
        data = response.json()
    except ValueError:
        data = {}
    if response.status_code != 200:
        error = data.get("error") if isinstance(data, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        if not message and isinstance(data, dict):
            message = data.get("message") or data.get("detail")
        message = redact(str(message or "no details"), key)[:300]
        raise ChatError(f"Request failed with status {response.status_code}: {message}")
    return data


def post_json(url, headers, body, key):
    """POST and return (parsed reply, latency in ms). Latency is the whole request, network included."""
    start = time.perf_counter()
    response = requests.post(url, headers=headers, json=body, timeout=TIMEOUT)
    latency_ms = round((time.perf_counter() - start) * 1000)
    return read_json(response, key), latency_ms


def milliseconds_since(start):
    return round((time.perf_counter() - start) * 1000)


# ---------- request bodies (shared by the normal and the streaming call) ----------

def mistral_body(system_prompt, question, temperature, top_p, max_tokens, stream=False):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    body = {"model": PROVIDERS["mistral"]["model"], "messages": messages}
    # A setting left as None is not sent at all, so the provider uses its own default.
    if temperature is not None:
        body["temperature"] = temperature
    if top_p is not None:
        body["top_p"] = top_p
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if stream:
        body["stream"] = True
    return body


def gemini_body(system_prompt, question, temperature, top_p, max_tokens):
    body = {"contents": [{"role": "user", "parts": [{"text": question}]}]}
    if system_prompt:
        body["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    config = {}
    if temperature is not None:
        config["temperature"] = temperature
    if top_p is not None:
        config["topP"] = top_p
    if max_tokens is not None:
        config["maxOutputTokens"] = max_tokens
    if config:
        body["generationConfig"] = config
    return body


def gemini_url(method):
    return GEMINI_URL.format(model=PROVIDERS["gemini"]["model"], method=method)


# ---------- reading the replies (shared by the normal and the streaming call) ----------

def mistral_finish(reason):
    return {"stop": "stop", "length": "length"}.get(reason, "other")


def gemini_finish(reason):
    return {"STOP": "stop", "MAX_TOKENS": "length"}.get(reason, "other")


def gemini_output_tokens(usage):
    # Thinking tokens are reported separately but billed as output, so they are counted as output.
    return usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)


def gemini_text(candidate):
    # With a very small max_tokens the model can use it all for thinking and send no text at all.
    parts = candidate.get("content", {}).get("parts", [])
    return "".join(part.get("text", "") for part in parts)


# ---------- whole answer at once ----------

def call_mistral(key, system_prompt, question, temperature, top_p, max_tokens):
    body = mistral_body(system_prompt, question, temperature, top_p, max_tokens)
    data, latency_ms = post_json(MISTRAL_URL, {"Authorization": f"Bearer {key}"}, body, key)
    choice = data["choices"][0]
    usage = data["usage"]
    return LLMResult(
        answer=choice["message"]["content"],
        input_tokens=usage["prompt_tokens"],
        output_tokens=usage["completion_tokens"],
        finish_reason=mistral_finish(choice["finish_reason"]),
        latency_ms=latency_ms,
        model=data.get("model", body["model"]),
    )


def call_gemini(key, system_prompt, question, temperature, top_p, max_tokens):
    body = gemini_body(system_prompt, question, temperature, top_p, max_tokens)
    data, latency_ms = post_json(gemini_url("generateContent"), {"x-goog-api-key": key}, body, key)
    if not data.get("candidates"):
        raise ChatError("Gemini returned no answer (the prompt may have been blocked).")
    candidate = data["candidates"][0]
    usage = data["usageMetadata"]
    return LLMResult(
        answer=gemini_text(candidate),
        input_tokens=usage["promptTokenCount"],
        output_tokens=gemini_output_tokens(usage),
        finish_reason=gemini_finish(candidate.get("finishReason")),
        latency_ms=latency_ms,
        model=data.get("modelVersion", PROVIDERS["gemini"]["model"]),
    )


def call_llm(provider, system_prompt, question, temperature=None, top_p=None, max_tokens=None):
    """Ask one question and return an LLMResult. Raises ChatError with a safe message on any failure."""
    if provider not in PROVIDERS:
        raise ChatError(f"Unknown provider: {provider}")
    key = load_key(provider)
    call = call_mistral if provider == "mistral" else call_gemini
    try:
        return call(key, (system_prompt or "").strip(), question, temperature, top_p, max_tokens)
    except requests.RequestException:
        raise ChatError("Network problem: could not reach the provider.") from None
    except (KeyError, IndexError, TypeError):
        raise ChatError("The reply was not in the expected format.") from None


# ---------- answer piece by piece (streaming) ----------

def open_stream(url, headers, body, key):
    """Send the request and return the response, ready to be read line by line."""
    response = requests.post(url, headers=headers, json=body, timeout=TIMEOUT, stream=True)
    if response.status_code != 200:
        read_json(response, key)  # raises ChatError with the provider's message
    return response


def read_events(response):
    """Yield the JSON of each 'data: {...}' line of a server-sent-events stream, until it ends."""
    for raw_line in response.iter_lines():
        line = raw_line.decode("utf-8")  # decoded by hand: the stream may not say it is UTF-8
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            return
        yield json.loads(payload)


def stream_mistral(key, system_prompt, question, temperature, top_p, max_tokens):
    body = mistral_body(system_prompt, question, temperature, top_p, max_tokens, stream=True)
    start = time.perf_counter()
    response = open_stream(MISTRAL_URL, {"Authorization": f"Bearer {key}"}, body, key)

    pieces, usage, finish, ttft_ms, model = [], None, None, None, body["model"]
    with response:
        for chunk in read_events(response):
            model = chunk.get("model", model)
            if chunk.get("usage"):  # Mistral sends the token counts in the last chunk
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                text = choice.get("delta", {}).get("content")
                if isinstance(text, str) and text:
                    if ttft_ms is None:
                        ttft_ms = milliseconds_since(start)
                    pieces.append(text)
                    yield {"type": "delta", "text": text}
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    latency_ms = milliseconds_since(start)

    if usage is None:
        raise ChatError("The stream ended without token counts.")
    yield {"type": "done", "result": LLMResult(
        answer="".join(pieces),
        input_tokens=usage["prompt_tokens"],
        output_tokens=usage["completion_tokens"],
        finish_reason=mistral_finish(finish),
        latency_ms=latency_ms,
        model=model,
        ttft_ms=ttft_ms,
    )}


def stream_gemini(key, system_prompt, question, temperature, top_p, max_tokens):
    body = gemini_body(system_prompt, question, temperature, top_p, max_tokens)
    start = time.perf_counter()
    # alt=sse makes Gemini send each piece as a "data: {...}" line, like Mistral does.
    response = open_stream(gemini_url("streamGenerateContent") + "?alt=sse", {"x-goog-api-key": key}, body, key)

    pieces, usage, finish, ttft_ms, model = [], None, None, None, PROVIDERS["gemini"]["model"]
    with response:
        for chunk in read_events(response):  # every chunk is a normal Gemini reply holding a bit more text
            model = chunk.get("modelVersion", model)
            if chunk.get("usageMetadata"):  # the counts so far; the last chunk has the final ones
                usage = chunk["usageMetadata"]
            for candidate in chunk.get("candidates", []):
                text = gemini_text(candidate)
                if text:
                    if ttft_ms is None:
                        ttft_ms = milliseconds_since(start)  # first visible text, after any thinking
                    pieces.append(text)
                    yield {"type": "delta", "text": text}
                if candidate.get("finishReason"):
                    finish = candidate["finishReason"]
    latency_ms = milliseconds_since(start)

    if usage is None:
        raise ChatError("Gemini returned no answer (the prompt may have been blocked).")
    yield {"type": "done", "result": LLMResult(
        answer="".join(pieces),
        input_tokens=usage["promptTokenCount"],
        output_tokens=gemini_output_tokens(usage),
        finish_reason=gemini_finish(finish),
        latency_ms=latency_ms,
        model=model,
        ttft_ms=ttft_ms,
    )}


def stream_llm(provider, system_prompt, question, temperature=None, top_p=None, max_tokens=None):
    """Like call_llm, but a generator: yields {"type": "delta", "text": ...} for each piece of the
    answer, then one {"type": "done", "result": LLMResult}. Raises ChatError on any failure."""
    if provider not in PROVIDERS:
        raise ChatError(f"Unknown provider: {provider}")
    key = load_key(provider)
    stream = stream_mistral if provider == "mistral" else stream_gemini
    try:
        yield from stream(key, (system_prompt or "").strip(), question, temperature, top_p, max_tokens)
    except requests.RequestException:
        raise ChatError("Network problem: the connection to the provider was lost.") from None
    except (KeyError, IndexError, TypeError, ValueError):
        raise ChatError("The reply was not in the expected format.") from None


if __name__ == "__main__":
    # Quick manual check: python -m backend.llm   (makes a few short real calls on both providers)
    question = "Name one planet. Reply with the name only."
    cases = [
        ("default", {}),
        ("system prompt", {"system_prompt": "Always answer in capital letters."}),
        ("temperature+top_p", {"temperature": 0.7, "top_p": 0.5}),
        ("max_tokens=5", {"max_tokens": 5}),
    ]
    for provider in PROVIDERS:
        for label, extra in cases:
            try:
                result = call_llm(provider, extra.get("system_prompt"), question,
                                  **{k: v for k, v in extra.items() if k != "system_prompt"})
                print(f"{provider:8} {label:18} {result}")
            except ChatError as error:
                print(f"{provider:8} {label:18} ERROR: {error}")
    print("streaming:")
    for provider in PROVIDERS:
        try:
            for event in stream_llm(provider, None, "Count from 1 to 5, separated by commas."):
                print(f"{provider:8} {event}")
        except ChatError as error:
            print(f"{provider:8} ERROR: {error}")
