"""Build the mini-report from the `runs` table.

    python scripts/report.py

Writes report/report.md (tables and quoted answers, all computed from the database) and
report/evidence.csv (the exact rows behind every number). The short interpretation under each
table is written by hand in report/commentary.md and copied in, so re-running this script after
new measurements never loses it.
"""

import csv
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend import db  # noqa: E402

REPORT_DIR = os.path.join(ROOT, "report")
EXPERIMENTS = ["system_prompt", "temperature", "max_tokens"]


# ---------- small helpers ----------

def clean(text, limit=None):
    """Make text safe for one markdown table cell."""
    text = (text or "").replace("\n", " ").replace("|", "/").strip()
    return text[:limit] + "..." if limit and len(text) > limit else text


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def average(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 1) if values else "n/a"


def id_range(rows):
    ids = [row["id"] for row in rows]
    return f"#{min(ids)}-{max(ids)}"


def latency_text(rows):
    values = [row["latency_ms"] for row in rows if row["latency_ms"] is not None]
    if not values:
        return "n/a"
    return f"{round(sum(values) / len(values))} ({min(values)}-{max(values)})"


# ---------- checks used in the system prompt section ----------

def count_sentences(text):
    text = re.sub(r"[*`_]", "", text.strip())
    return len([part for part in re.split(r"(?<=[.!?])\s+", text) if part])


def is_two_sentences(answer):
    return count_sentences(answer) == 2


def sounds_like_pirate(answer):
    return re.search(r"\b(arr+|matey|ahoy|aye|ye|yer|cap'?n)\b", answer.lower()) is not None


def is_json_answer(text, strict=True):
    """True if text parses as a JSON object with an "answer" key.

    strict=False also accepts raw line breaks inside strings, which the JSON standard forbids.
    """
    try:
        data = json.loads(text, strict=strict)
    except ValueError:
        return False
    return isinstance(data, dict) and "answer" in data


def remove_code_fence(answer):
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", answer.strip())


CHECKS = [
    ("exactly 2 sentences", is_two_sentences),
    ("pirate words", sounds_like_pirate),
    ("JSON exactly as sent", lambda answer: is_json_answer(answer.strip())),
    ("JSON after removing a code fence", lambda answer: is_json_answer(remove_code_fence(answer))),
    ("same, lenient parser", lambda answer: is_json_answer(remove_code_fence(answer), strict=False)),
]


# ---------- reading the database ----------

def load_rows(experiment, provider):
    """Rows of one experiment on one provider, grouped by setting in the order they were run."""
    with db.connect() as connection:
        rows = connection.execute(
            "SELECT * FROM runs WHERE experiment = ? AND provider = ? ORDER BY id", (experiment, provider)
        ).fetchall()
    groups = {}
    for row in rows:
        groups.setdefault(row["setting_label"], []).append(dict(row))
    return groups


def providers_of(experiment):
    with db.connect() as connection:
        found = connection.execute(
            "SELECT DISTINCT provider FROM runs WHERE experiment = ? ORDER BY provider", (experiment,)
        ).fetchall()
    return [row["provider"] for row in found]


def successful(rows):
    return [row for row in rows if row["error"] is None]


# ---------- one section per experiment ----------

def section_system_prompt(groups):
    summary, matrix, samples = [], [], []
    for label, rows in groups.items():
        ok = successful(rows)
        summary.append([label, id_range(rows), f"{len(ok)}/{len(rows)}", average([r["output_tokens"] for r in ok]),
                        latency_text(ok)])
        matrix.append([label] + [f"{sum(check(r['answer']) for r in ok)}/{len(ok)}" for _, check in CHECKS])
        if ok:
            samples.append(f"- **{label}** (run #{ok[0]['id']}): {clean(ok[0]['answer'], 220)}")
    text = table(["Setting", "Run ids", "Successful", "Avg output tokens", "Avg latency ms (min-max)"], summary)
    text += "\n\nHow many answers pass each check (every check is applied to every setting):\n\n"
    text += table(["Setting"] + [name for name, _ in CHECKS], matrix)
    text += "\n\nFirst answer of each setting:\n\n" + "\n".join(samples)
    return text


def section_temperature(groups):
    summary, answers = [], []
    for label, rows in groups.items():
        ok = successful(rows)
        distinct = len(set(r["answer"].strip() for r in ok))
        summary.append([label, id_range(rows), f"{len(ok)}/{len(rows)}", f"{distinct} of {len(ok)}",
                        average([r["output_tokens"] for r in ok]), latency_text(ok)])
        counts = Counter(clean(r["answer"], 60) for r in ok)
        listed = "; ".join(f"{answer} x{n}" if n > 1 else answer for answer, n in counts.items())
        answers.append(f"- **{label}**: {listed}")
    text = table(["Setting", "Run ids", "Successful", "Distinct answers", "Avg output tokens",
                  "Avg latency ms (min-max)"], summary)
    text += "\n\nEvery answer, with a count when it repeats:\n\n" + "\n".join(answers)
    return text


def section_max_tokens(groups):
    summary, endings = [], []
    for label, rows in groups.items():
        ok = successful(rows)
        cut = sum(r["finish_reason"] == "length" for r in ok)
        summary.append([label, id_range(rows), f"{len(ok)}/{len(rows)}", average([r["output_tokens"] for r in ok]),
                        f"{cut}/{len(ok)}", average([len(r["answer"]) for r in ok]), latency_text(ok)])
        if ok:
            endings.append(f"- **{label}** (run #{ok[0]['id']}, finish={ok[0]['finish_reason']}): "
                           f"...{clean(ok[0]['answer'][-80:])}")
    text = table(["Setting", "Run ids", "Successful", "Avg output tokens", "Cut off (finish=length)",
                  "Avg answer characters", "Avg latency ms (min-max)"], summary)
    text += "\n\nHow the first answer of each setting ends:\n\n" + "\n".join(endings)
    return text


SECTIONS = {
    "system_prompt": ("1. System prompt change (format constraint and imposed role)",
                      "Question: *Explain what a database index is.* Temperature fixed at 0.7. Only the system prompt changes.",
                      section_system_prompt),
    "temperature": ("2. Low versus high temperature (same question replayed)",
                    "Question: *Suggest a name for a coffee shop. Reply with the name only.* No system prompt.",
                    section_temperature),
    "max_tokens": ("3. Third parameter: max_tokens",
                   "Question: *Explain what a database index is.* Temperature fixed at 0.7. Only max_tokens changes.",
                   section_max_tokens),
}


def read_commentary():
    """Read report/commentary.md: a '## name' heading per section, free text below it."""
    path = os.path.join(REPORT_DIR, "commentary.md")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as file:
        parts = re.split(r"^## (\w+)\s*$", file.read(), flags=re.MULTILINE)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def write_evidence(path):
    """Every row used in the report, in one CSV file."""
    marks = ", ".join("?" for _ in EXPERIMENTS)
    with db.connect() as connection:
        rows = connection.execute(
            f"SELECT * FROM runs WHERE experiment IN ({marks}) ORDER BY id", EXPERIMENTS
        ).fetchall()
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        if rows:
            writer.writerow(rows[0].keys())
        for row in rows:
            writer.writerow(list(row))
    return len(rows)


def main():
    db.init_db()
    os.makedirs(REPORT_DIR, exist_ok=True)
    commentary = read_commentary()

    models = {}
    with db.connect() as connection:
        for row in connection.execute(
            "SELECT DISTINCT provider, model FROM runs WHERE experiment IN (?, ?, ?)", EXPERIMENTS
        ):
            models.setdefault(row["provider"], []).append(row["model"])
    model_text = "; ".join(f"{provider}: {', '.join(names)}" for provider, names in models.items()) or "no runs yet"

    out = [
        "# Mini-report: what happens when one model setting changes",
        "",
        f"Generated by `scripts/report.py` on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC "
        "from the `runs` table. Every number below is computed from stored rows; the run ids point to them, "
        "and `report/evidence.csv` holds all of those rows.",
        "",
        f"Models: {model_text}. Settings not listed in a section were not sent (provider default).",
        "Latency is the full request time measured around the HTTP call (network included, no streaming).",
    ]

    for name in EXPERIMENTS:
        title, setup, build = SECTIONS[name]
        out += ["", f"## {title}", "", setup, ""]
        providers = providers_of(name)
        if not providers:
            out.append("_No runs stored for this experiment. Run `python scripts/replay.py`._")
        for provider in providers:
            out += [f"### {provider}", "", build(load_rows(name, provider)), ""]
        out += ["**What this shows.** " + commentary.get(name, "_(no commentary written yet)_")]

    out += ["", "## Limits of this report", "", commentary.get("limits", "_(no commentary written yet)_"), ""]

    with open(os.path.join(REPORT_DIR, "report.md"), "w", encoding="utf-8") as file:
        file.write("\n".join(out))
    count = write_evidence(os.path.join(REPORT_DIR, "evidence.csv"))
    print(f"Wrote report/report.md and report/evidence.csv ({count} rows).")


if __name__ == "__main__":
    main()
