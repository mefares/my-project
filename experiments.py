import argparse
import json
import os

from chat import DEFAULT_TEMPERATURE, PROVIDERS, ChatError, load_key, redact

RUNS = 3  # how many times each combination is repeated
# None means "the default temperature of chat.py"; it is recorded as null in the results.
TEMPERATURES = [None, 0, 1]


def read_list(path, what):
    """Load a JSON file that holds a list, or exit with a readable message."""
    try:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, ValueError):
        raise SystemExit(f"Error: cannot read {what} from {path} (is it valid JSON?).")
    if not isinstance(data, list):
        raise SystemExit(f"Error: {path} must contain a JSON list.")
    return data


def save(path, results):
    # Write to a temporary file first, so an interruption never leaves a half-written file.
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def run_once(provider, key, messages, temperature):
    """One API call. Returns the answer and token counts, or the error that happened."""
    sent_temperature = DEFAULT_TEMPERATURE if temperature is None else temperature
    model = PROVIDERS[provider]["model"]
    try:
        answer, tokens_in, tokens_out = PROVIDERS[provider]["ask"](key, model, messages, sent_temperature)
        return {"answer": answer, "input_tokens": tokens_in, "output_tokens": tokens_out, "error": None}
    except Exception as error:  # record it and keep the experiment going
        if isinstance(error, ChatError):
            message = str(error)  # already safe to show: it never contains the key
        elif isinstance(error, (KeyError, IndexError, TypeError)):
            message = "the reply was not in the expected format"
        else:
            message = "request failed"  # network details are left out, as in chat.py
        error_info = {"type": type(error).__name__, "message": redact(message, key)[:300]}
        return {"answer": None, "input_tokens": None, "output_tokens": None, "error": error_info}


def main():
    parser = argparse.ArgumentParser(description="Run every question several times on every provider.")
    parser.add_argument("--questions", default="questions.json")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--ids", help="only run these question ids, e.g. 1,6 (handy for a quick test)")
    parser.add_argument(
        "--providers", nargs="+", choices=PROVIDERS, default=list(PROVIDERS),
        help="providers to run (default: all), e.g. --providers mistral",
    )
    args = parser.parse_args()

    questions = read_list(args.questions, "the questions")
    if args.ids:
        wanted = {int(number) for number in args.ids.split(",")}
        questions = [entry for entry in questions if entry["id"] in wanted]

    # Append to the results of earlier runs, if there are any.
    results = read_list(args.output, "the results") if os.path.exists(args.output) else []

    # Each key is read once, here. It only ever travels to the provider function.
    keys = {provider: load_key(provider) for provider in args.providers}

    total = len(questions) * len(args.providers) * len(TEMPERATURES) * RUNS
    done = 0
    for entry in questions:
        for provider in args.providers:
            for temperature in TEMPERATURES:
                for run in range(1, RUNS + 1):
                    # A fresh list every time: no history is shared between runs.
                    if entry.get("messages"):
                        messages = [dict(message) for message in entry["messages"]]
                    else:
                        messages = [{"role": "user", "content": entry["question"]}]

                    outcome = run_once(provider, keys[provider], messages, temperature)
                    results.append({
                        "id": entry["id"],
                        "provider": provider,
                        "temperature": temperature,
                        "run": run,
                        **outcome,
                    })
                    save(args.output, results)

                    done += 1
                    label = "default" if temperature is None else temperature
                    status = "ok" if outcome["error"] is None else outcome["error"]["type"]
                    print(f"[{done}/{total}] question {entry['id']} {provider} T={label} run {run}: {status}")

    print(f"Done. {len(results)} results in {args.output}.")


if __name__ == "__main__":
    main()
