"""The only module that talks to the model providers (Mistral and Gemini).

The rest of the app calls call_llm() and gets the same LLMResult back, whichever provider answers.
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path

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


def call_mistral(key, system_prompt, question, temperature, top_p, max_tokens):
    model = PROVIDERS["mistral"]["model"]
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    body = {"model": model, "messages": messages}
    # A setting left as None is not sent at all, so the provider uses its own default.
    if temperature is not None:
        body["temperature"] = temperature
    if top_p is not None:
        body["top_p"] = top_p
    if max_tokens is not None:
        body["max_tokens"] = max_tokens

    data, latency_ms = post_json(
        "https://api.mistral.ai/v1/chat/completions", {"Authorization": f"Bearer {key}"}, body, key
    )
    choice = data["choices"][0]
    usage = data["usage"]
    finish = {"stop": "stop", "length": "length"}.get(choice["finish_reason"], "other")
    return LLMResult(
        answer=choice["message"]["content"],
        input_tokens=usage["prompt_tokens"],
        output_tokens=usage["completion_tokens"],
        finish_reason=finish,
        latency_ms=latency_ms,
        model=data.get("model", model),
    )


def call_gemini(key, system_prompt, question, temperature, top_p, max_tokens):
    model = PROVIDERS["gemini"]["model"]
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

    data, latency_ms = post_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        {"x-goog-api-key": key},
        body,
        key,
    )
    if not data.get("candidates"):
        raise ChatError("Gemini returned no answer (the prompt may have been blocked).")
    candidate = data["candidates"][0]
    # With a very small max_tokens the model can use it all for thinking and send no text at all.
    parts = candidate.get("content", {}).get("parts", [])
    answer = "".join(part.get("text", "") for part in parts)

    usage = data["usageMetadata"]
    # Thinking tokens are reported separately but billed as output, so they are counted as output.
    output_tokens = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
    finish = {"STOP": "stop", "MAX_TOKENS": "length"}.get(candidate.get("finishReason"), "other")
    return LLMResult(
        answer=answer,
        input_tokens=usage["promptTokenCount"],
        output_tokens=output_tokens,
        finish_reason=finish,
        latency_ms=latency_ms,
        model=data.get("modelVersion", model),
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


if __name__ == "__main__":
    # Quick manual check: python -m backend.llm   (makes a few short real calls on both providers)
    question = "Name one planet. Reply with the name only."
    cases = [
        ("default", {}),
        ("system prompt", {"system_prompt": "Always answer in capital letters."}),
        ("temperature+top_p", {"temperature": 0.0, "top_p": 0.5}),
        ("max_tokens=5", {"max_tokens": 5}),
    ]
    for provider in PROVIDERS:
        for label, extra in cases:
            try:
                result = call_llm(provider, extra.pop("system_prompt", None), question, **extra)
                print(f"{provider:8} {label:18} {result}")
            except ChatError as error:
                print(f"{provider:8} {label:18} ERROR: {error}")
