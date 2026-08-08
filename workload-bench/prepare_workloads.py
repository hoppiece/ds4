#!/usr/bin/env python3
"""Download public LLM workloads and render deterministic DS4 benchmark cases.

Generated third-party data is gitignored.  The script uses only the Python
standard library and records SHA-256 hashes for reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import random
import shutil
import sys
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


SOURCES = {
    "mt_bench_questions": {
        "url": "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
        "file": "mt_bench_question.jsonl",
        "license": "Apache-2.0",
    },
    "mt_bench_answers": {
        "url": "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/reference_answer/gpt-4.jsonl",
        "file": "mt_bench_gpt4_reference.jsonl",
        "license": "Apache-2.0",
    },
    "elyza_tasks_100": {
        "url": "https://huggingface.co/datasets/elyza/ELYZA-tasks-100/resolve/main/test.csv",
        "file": "elyza_tasks_100.csv",
        "license": "CC-BY-SA-4.0",
    },
    "oasst1": {
        "url": "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_ready.messages.jsonl.gz",
        "file": "oasst1_ready.messages.jsonl.gz",
        "license": "Apache-2.0",
    },
    "longbench": {
        "url": "https://huggingface.co/datasets/zai-org/LongBench/resolve/main/data.zip",
        "file": "longbench_data.zip",
        "license": "MIT",
    },
    "longbench_prompts": {
        "url": "https://raw.githubusercontent.com/THUDM/LongBench/main/LongBench/config/dataset2prompt.json",
        "file": "longbench_dataset2prompt.json",
        "license": "MIT",
    },
}

LICENSES = {
    "FastChat-Apache-2.0.txt":
        "https://raw.githubusercontent.com/lm-sys/FastChat/main/LICENSE",
    "ELYZA-CC-BY-SA-4.0.txt":
        "https://huggingface.co/datasets/elyza/ELYZA-tasks-100/resolve/main/LICENSE",
    "OASST1-Apache-2.0.txt":
        "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/LICENSE",
    "LongBench-MIT.txt":
        "https://raw.githubusercontent.com/THUDM/LongBench/main/LICENSE",
}

DEFAULT_SYSTEM = "You are a helpful assistant."
LONG_TASKS = ("qasper", "hotpotqa", "gov_report", "qmsum", "lcc", "repobench-p")


@dataclass
class Case:
    cid: str
    category: str
    source: str
    messages: list[dict]
    rendered: str = ""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, path: Path, force: bool) -> None:
    if path.exists() and not force:
        print(f"reuse {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "ds4-workload-bench/1"})
    print(f"download {url}")
    with urllib.request.urlopen(req, timeout=120) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    tmp.replace(path)


def load_renderer(root: Path):
    path = root / "gguf-tools/imatrix/dataset/build_ds4_imatrix_dataset.py"
    spec = importlib.util.spec_from_file_location("ds4_imatrix_dataset", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import DS4 renderer from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.render


def chat(user: str, *, history: list[dict] | None = None) -> list[dict]:
    messages = [{"role": "system", "content": DEFAULT_SYSTEM}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user})
    return messages


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def mt_bench_cases(raw: Path) -> list[Case]:
    refs = {}
    for row in load_jsonl(raw / SOURCES["mt_bench_answers"]["file"]):
        refs[row.get("question_id")] = row.get("choices", [{}])[0].get("turns", [])
    cases = []
    for row in load_jsonl(raw / SOURCES["mt_bench_questions"]["file"]):
        qid = int(row["question_id"])
        category = str(row.get("category", "unknown"))
        turns = row.get("turns", [])
        if not turns:
            continue
        cases.append(Case(f"mt-{qid}-t1", f"mt_{category}", "mt-bench", chat(turns[0])))
        answers = refs.get(qid, [])
        if len(turns) > 1 and answers:
            history = [
                {"role": "user", "content": turns[0]},
                {"role": "assistant", "content": answers[0]},
            ]
            cases.append(Case(f"mt-{qid}-t2", f"mt_{category}", "mt-bench",
                              chat(turns[1], history=history)))
    return cases


def elyza_cases(raw: Path) -> list[Case]:
    path = raw / SOURCES["elyza_tasks_100"]["file"]
    cases = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=1):
            prompt = row.get("input", "").strip()
            if prompt:
                cases.append(Case(f"elyza-{i:03d}", "japanese_instruction",
                                  "elyza-tasks-100", chat(prompt)))
    return cases


def oasst_cases(raw: Path, limit: int) -> list[Case]:
    path = raw / SOURCES["oasst1"]["file"]
    messages = {}
    children = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("deleted") or row.get("review_result") is False:
                continue
            mid = row.get("message_id")
            if not mid:
                continue
            messages[mid] = row
            if row.get("parent_id"):
                children[row["parent_id"]].append(row)

    def best_child(parent: str, role: str):
        rows = [r for r in children.get(parent, []) if r.get("role") == role]
        rows.sort(key=lambda r: (r.get("rank") is None,
                                 r.get("rank") if r.get("rank") is not None else 999,
                                 r.get("message_id", "")))
        return rows[0] if rows else None

    preferred = {"en": 0, "ja": 1, "zh": 2, "es": 3, "de": 4, "fr": 5, "ru": 6}
    roots_by_lang = defaultdict(list)
    for row in messages.values():
        if not row.get("parent_id") and row.get("role") == "prompter":
            roots_by_lang[row.get("lang") or "unknown"].append(row)
    for rows in roots_by_lang.values():
        rows.sort(key=lambda r: r.get("message_id", ""))
    languages = sorted(roots_by_lang, key=lambda lang: (preferred.get(lang, 99), lang))
    roots = []
    while True:
        added = False
        for lang in languages:
            if roots_by_lang[lang]:
                roots.append(roots_by_lang[lang].pop(0))
                added = True
        if not added:
            break
    cases = []
    for root in roots:
        lang = root.get("lang") or "unknown"
        text = (root.get("text") or "").strip()
        if not text:
            continue
        rid = root["message_id"][:12]
        cases.append(Case(f"oasst-{rid}-t1", f"oasst_{lang}", "oasst1", chat(text)))
        answer = best_child(root["message_id"], "assistant")
        followup = best_child(answer["message_id"], "prompter") if answer else None
        if answer and followup:
            history = [
                {"role": "user", "content": text},
                {"role": "assistant", "content": answer.get("text", "")},
            ]
            cases.append(Case(f"oasst-{rid}-t2", f"oasst_{lang}", "oasst1",
                              chat(followup.get("text", ""), history=history)))
        if len(cases) >= limit:
            break
    return cases[:limit]


def zip_jsonl(zf: zipfile.ZipFile, task: str) -> list[dict]:
    suffix = f"/{task}.jsonl"
    names = [n for n in zf.namelist() if n.endswith(suffix) or n == f"{task}.jsonl"]
    names = [n for n in names if "/data_e/" not in f"/{n}"] or names
    if not names:
        raise RuntimeError(f"LongBench archive does not contain {task}.jsonl")
    with zf.open(sorted(names, key=len)[0]) as raw:
        return [json.loads(line) for line in raw if line.strip()]


def evenly_spaced(rows: list[dict], count: int) -> list[dict]:
    if len(rows) <= count:
        return rows
    if count == 1:
        return [rows[len(rows) // 2]]
    return [rows[round(i * (len(rows) - 1) / (count - 1))] for i in range(count)]


def longbench_cases(raw: Path, per_task: int, max_chars: int) -> list[Case]:
    prompts = json.loads((raw / SOURCES["longbench_prompts"]["file"]).read_text())
    cases = []
    with zipfile.ZipFile(raw / SOURCES["longbench"]["file"]) as zf:
        for task in LONG_TASKS:
            rows = zip_jsonl(zf, task)
            rows = [r for r in rows if len(r.get("context", "")) <= max_chars]
            rows.sort(key=lambda r: (len(r.get("context", "")),
                                     json.dumps(r, sort_keys=True)[:128]))
            for i, row in enumerate(evenly_spaced(rows, per_task), start=1):
                cases.append(Case(f"long-{task}-{i:02d}", f"long_{task}",
                                  "longbench", chat(prompts[task].format(**row))))
    return cases


def safe_field(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)


def render_cases(cases: list[Case], render, mode: str) -> None:
    for case in cases:
        case.cid = safe_field(case.cid)
        case.category = safe_field(case.category)
        case.source = safe_field(case.source)
        case.rendered = render(case.messages, mode)
        if "\n===== DS4_WORKLOAD " in case.rendered:
            raise RuntimeError(f"workload marker collision in {case.cid}")


def write_workload(path: Path, cases: list[Case]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(f"===== DS4_WORKLOAD {case.cid} {case.category} {case.source} =====\n")
            f.write(case.rendered.rstrip())
            f.write("\n")


def quick_cases(cases: list[Case]) -> list[Case]:
    chosen = []
    for source in ("mt-bench", "elyza-tasks-100", "oasst1", "longbench"):
        rows = [c for c in cases if c.source == source]
        if rows:
            chosen.append(min(rows, key=lambda c: len(c.rendered)))
    return chosen


def balanced_cases(cases: list[Case]) -> list[Case]:
    """Interleave sources so small benchmark prefixes cover every workload."""
    source_order = ("mt-bench", "elyza-tasks-100", "oasst1", "longbench")
    grouped = {source: [case for case in cases if case.source == source]
               for source in source_order}
    balanced = []
    index = 0
    while True:
        added = False
        for source in source_order:
            rows = grouped[source]
            if index < len(rows):
                balanced.append(rows[index])
                added = True
        if not added:
            break
        index += 1
    return balanced


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="Data directory (default: workload-bench/data)")
    parser.add_argument("--force", action="store_true", help="Redownload existing raw files")
    parser.add_argument("--mode", choices=("nothink", "think"), default="nothink")
    parser.add_argument("--oasst-limit", type=int, default=128)
    parser.add_argument("--longbench-per-task", type=int, default=3)
    parser.add_argument("--longbench-max-chars", type=int, default=60000)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    out = Path(args.out).resolve() if args.out else script_dir / "data"
    raw = out / "raw"
    licenses = out / "licenses"
    out.mkdir(parents=True, exist_ok=True)

    for meta in SOURCES.values():
        download(meta["url"], raw / meta["file"], args.force)
    for name, url in LICENSES.items():
        download(url, licenses / name, args.force)

    render = load_renderer(root)
    cases = []
    cases.extend(mt_bench_cases(raw))
    cases.extend(elyza_cases(raw))
    cases.extend(oasst_cases(raw, args.oasst_limit))
    cases.extend(longbench_cases(raw, args.longbench_per_task, args.longbench_max_chars))
    render_cases(cases, render, args.mode)
    random.Random(20260808).shuffle(cases)

    write_workload(out / "workload.txt", cases)
    write_workload(out / "workload-balanced.txt", balanced_cases(cases))
    write_workload(out / "workload-quick.txt", quick_cases(cases))
    with (out / "records.jsonl").open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps({"id": case.cid, "category": case.category,
                                "source": case.source, "messages": case.messages},
                               ensure_ascii=False, separators=(",", ":")) + "\n")

    counts = defaultdict(int)
    for case in cases:
        counts[case.source] += 1
    manifest = {
        "format": "DS4_WORKLOAD_v1",
        "mode": args.mode,
        "record_count": len(cases),
        "counts": dict(sorted(counts.items())),
        "selection": {
            "oasst_limit": args.oasst_limit,
            "longbench_tasks": list(LONG_TASKS),
            "longbench_per_task": args.longbench_per_task,
            "longbench_max_chars": args.longbench_max_chars,
            "shuffle_seed": 20260808,
            "balanced_source_order": [
                "mt-bench", "elyza-tasks-100", "oasst1", "longbench"
            ],
        },
        "sources": {
            name: {"url": meta["url"], "license": meta["license"],
                   "file": meta["file"], "sha256": sha256(raw / meta["file"])}
            for name, meta in SOURCES.items()
        },
        "outputs": {"workload": "workload.txt",
                    "balanced": "workload-balanced.txt",
                    "quick": "workload-quick.txt", "records": "records.jsonl"},
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases to {out / 'workload.txt'}")
    for source, count in sorted(counts.items()):
        print(f"  {source}: {count}")


if __name__ == "__main__":
    main()
