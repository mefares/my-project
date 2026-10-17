"""Run the same question N times per setting and save every run in the `runs` table.

Examples:
    python scripts/replay.py --experiment temperature --provider mistral --n 5
    python scripts/replay.py --experiment all --fresh

The report is built from these rows, so it rests on measurements, not on impressions.
"""

import argparse
import os
import sys
import time

# Let this script import the backend package when it is started as "python scripts/replay.py".
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import db  # noqa: E402
from backend.llm import PROVIDER_INFO, ChatError, call_llm  # noqa: E402

QUESTION_EXPLAIN = "Explain what a database index is."
QUESTION_NAME = "Suggest a name for a coffee shop. Reply with the name only."

# Each experiment changes ONE thing between its settings; everything else stays the same.
# A value of None means "not sent" (provider default).
EXPERIMENTS = {
    # 1. System prompt: a format constraint, an imposed role, and a strict output format.
    "system_prompt": {
        "question": QUESTION_EXPLAIN,
        "settings": [
            {"label": "no system prompt", "system_prompt": None, "temperature": 0.7},
            {"label": "exactly two sentences", "system_prompt": "Answer in exactly two sentences.", "temperature": 0.7},
            {"label": "pirate role", "system_prompt": "You are a pirate. Always talk like a pirate.", "temperature": 0.7},
            {
                "label": "JSON only",
                "system_prompt": 'Reply only with valid JSON of the form {"answer": "..."} and no other text.',
                "temperature": 0.7,
            },
        ],
    },
    # 2. Temperature: low, medium and high on a question that has many good answers.
    "temperature": {
        "question": QUESTION_NAME,
        "settings": [
            {"label": "temp=0.0", "temperature": 0.0},
            {"label": "temp=1.0", "temperature": 1.0},
            {"label": "temp=1.5", "temperature": 1.5},
        ],
    },
    # 3. max_tokens: very short, medium, and generous. This model's unconstrained answer to this
    #    question is around 800 tokens, so only the generous setting lets it finish by itself.
    "max_tokens": {
        "question": QUESTION_EXPLAIN,
        "settings": [
            {"label": "max_tokens=20", "max_tokens": 20, "temperature": 0.7},
            {"label": "max_tokens=300", "max_tokens": 300, "temperature": 0.7},
            {"label": "max_tokens=2000", "max_tokens": 2000, "temperature": 0.7},
        ],
    },
}

MAX_ERRORS_IN_A_ROW = 5  # stop early instead of wasting calls when the provider is failing


def run_once(provider, experiment, question, setting):
    """Make one call and save it (also when it fails). Returns True if it worked."""
    saved = dict(
        provider=provider,
        model=PROVIDER_INFO[provider]["model"],
        system_prompt=setting.get("system_prompt"),
        temperature=setting.get("temperature"),
        top_p=setting.get("top_p"),
        max_tokens=setting.get("max_tokens"),
        question=question,
        experiment=experiment,
        setting_label=setting["label"],
    )
    try:
        result = call_llm(
            provider, saved["system_prompt"], question, saved["temperature"], saved["top_p"], saved["max_tokens"]
        )
    except ChatError as error:
        db.insert_run(**saved, error=str(error))
        print(f"    ERROR: {str(error)[:100]}")
        return False

    saved.update(
        model=result.model,
        answer=result.answer,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        finish_reason=result.finish_reason,
        latency_ms=result.latency_ms,
    )
    db.insert_run(**saved)
    print(f"    ok: {result.output_tokens} tokens out, {result.latency_ms} ms, finish={result.finish_reason}")
    return True


def run_experiment(name, provider, n, fresh, pause):
    experiment = EXPERIMENTS[name]
    if fresh:
        print(f"--fresh: deleted {db.delete_experiment(name, provider)} old rows of '{name}' on {provider}")

    total = len(experiment["settings"]) * n
    done = failed = errors_in_a_row = 0
    for setting in experiment["settings"]:
        for run_number in range(1, n + 1):
            done += 1
            print(f"[{name}] {done}/{total} {setting['label']} (run {run_number})")
            if run_once(provider, name, experiment["question"], setting):
                errors_in_a_row = 0
            else:
                failed += 1
                errors_in_a_row += 1
                if errors_in_a_row >= MAX_ERRORS_IN_A_ROW:
                    print(f"Stopping: {MAX_ERRORS_IN_A_ROW} errors in a row.")
                    return done, failed
            time.sleep(pause)  # small pause so free-tier rate limits are not hit
    return done, failed


def main():
    parser = argparse.ArgumentParser(description="Replay an experiment and store the runs.")
    parser.add_argument("--experiment", choices=list(EXPERIMENTS) + ["all"], required=True)
    parser.add_argument("--provider", choices=list(PROVIDER_INFO), default="mistral")
    parser.add_argument("--n", type=int, default=5, help="runs per setting (default 5)")
    parser.add_argument("--fresh", action="store_true", help="delete earlier rows of this experiment first")
    parser.add_argument("--pause", type=float, default=1.0, help="seconds between calls (default 1)")
    args = parser.parse_args()

    names = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    ignored = PROVIDER_INFO[args.provider]["ignored"]
    if "temperature" in ignored and "temperature" in names:
        print(f"Note: {PROVIDER_INFO[args.provider]['model']} ignores temperature, so that experiment will show no effect.")

    db.init_db()
    for name in names:
        done, failed = run_experiment(name, args.provider, args.n, args.fresh, args.pause)
        print(f"Finished '{name}': {done - failed} ok, {failed} failed.\n")


if __name__ == "__main__":
    main()
