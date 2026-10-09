import argparse
import os
import sys

import requests
from dotenv import load_dotenv

TIMEOUT = 60  # seconds to wait for a reply before giving up

# Temperature controls how random the model's next-word choice is. Near 0 it
# picks the most likely words, so answers are stable and repeatable; higher
# values give more varied and more creative answers, but also more mistakes.
DEFAULT_TEMPERATURE = 0.7


class ChatError(Exception):
    """A readable error that is safe to print (it never contains the API key)."""


def redact(text, key):
    # Defense in depth: even if a server echoed the key back, it never gets printed.
    return text.replace(key, "***")


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


def ask_mistral(key, model, history, temperature):
    """Send the whole history to Mistral. Returns (reply, input_tokens, output_tokens)."""
    response = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "messages": history, "temperature": temperature},
        timeout=TIMEOUT,
    )
    data = read_json(response, key)
    # A token is a chunk of text the model reads or writes, usually a word piece of
    # about 3-4 English characters. Providers bill and limit by tokens, not characters.
    # "Input" tokens are everything we sent; "output" tokens are what the model wrote.
    usage = data["usage"]
    reply = data["choices"][0]["message"]["content"]
    return reply, usage["prompt_tokens"], usage["completion_tokens"]


def ask_gemini(key, model, history, temperature):
    """Send the whole history to Gemini. Returns (reply, input_tokens, output_tokens)."""
    # Gemini calls the assistant "model" and wraps each text in a list of "parts".
    contents = [
        {
            "role": "model" if message["role"] == "assistant" else "user",
            "parts": [{"text": message["content"]}],
        }
        for message in history
    ]
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": key},
        json={"contents": contents, "generationConfig": {"temperature": temperature}},
        timeout=TIMEOUT,
    )
    data = read_json(response, key)
    parts = data["candidates"][0]["content"]["parts"]
    reply = "".join(part.get("text", "") for part in parts)
    usage = data["usageMetadata"]
    # Models that "think" before answering report that reasoning separately, but it is
    # billed as output, so it is counted as output here.
    output_tokens = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
    return reply, usage["promptTokenCount"], output_tokens


PROVIDERS = {
    "mistral": {"env": "MISTRAL_API_KEY", "model": "ministral-8b-latest", "ask": ask_mistral},
    "gemini": {"env": "GEMINI_API_KEY", "model": "gemini-flash-latest", "ask": ask_gemini},
}


def load_key(provider):
    """Read the API key from the environment. Called once, at startup."""
    load_dotenv()  # copies the lines of .env into the environment
    name = PROVIDERS[provider]["env"]
    key = os.environ.get(name, "").strip()
    if not key:
        raise SystemExit(f"Error: {name} is not set. Add it to your .env file.")
    return key


def main():
    parser = argparse.ArgumentParser(description="Chat with an AI model in the terminal.")
    parser.add_argument("--provider", choices=PROVIDERS, default="mistral")
    parser.add_argument("--model", help="override the provider's default model")
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help=f"randomness from 0 (steady) to 2 (varied); default {DEFAULT_TEMPERATURE}",
    )
    args = parser.parse_args()
    if not 0 <= args.temperature <= 2:
        parser.error("--temperature must be between 0 and 2")

    provider = PROVIDERS[args.provider]
    model = args.model or provider["model"]
    key = load_key(args.provider)

    # Keep going if the terminal cannot display a character in a reply.
    sys.stdout.reconfigure(errors="replace")

    # The model has no memory: every call must carry the whole conversation. It can only
    # read that much text at once, up to its context window, the maximum number of tokens
    # it accepts in one request. So the input token count grows every turn, and a long
    # enough chat would eventually fail or need its oldest messages trimmed.
    history = []
    total_in = total_out = 0

    print(f"Chatting with {model} ({args.provider}), temperature {args.temperature}.")
    print("Commands: /reset clears the history, /quit exits.\n")

    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/reset":
            history.clear()
            print("History cleared.\n")
            continue

        history.append({"role": "user", "content": text})
        try:
            reply, tokens_in, tokens_out = provider["ask"](key, model, history, args.temperature)
        except ChatError as error:
            history.pop()  # keep the history valid: this question got no answer
            print(f"Error: {error}\n")
            continue
        except requests.RequestException as error:
            history.pop()
            print(f"Error: network problem ({type(error).__name__}).\n")
            continue
        except (KeyError, IndexError, TypeError):
            history.pop()
            print("Error: the reply was not in the expected format (it may have been blocked).\n")
            continue

        history.append({"role": "assistant", "content": reply})
        total_in += tokens_in
        total_out += tokens_out
        print(f"\nai> {reply}\n")
        print(f"[tokens] in: {tokens_in}  out: {tokens_out}  |  session in: {total_in}  out: {total_out}\n")


if __name__ == "__main__":
    main()
