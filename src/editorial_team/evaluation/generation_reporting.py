"""Generation evaluation JSON and Markdown reporting."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any


def write_generation_results(result: dict[str, Any], json_path: Path, report_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_generation_report(result), encoding="utf-8")


def render_generation_report(result: dict[str, Any]) -> str:
    on = result["conditions"]["on"]
    off = result["conditions"]["off"]
    off_ids = set(off["case_ids"])
    on_subset = _averages([row for row in on["cases"] if row["case_id"] in off_ids])
    categories = Counter(row["failure_category"] for row in on["cases"])
    lines = [
        "# Judged generation evaluation",
        "",
        f"Cases: {result['case_count']} rerank-on; {len(off['cases'])} stratified rerank-off. ",
        f"Generator: `{result['generator_model']}`. Judge: `{result['judge_model']}`.",
        "",
        "## Failure-category distribution",
        "",
        *[f"- `{name}`: {count}" for name, count in sorted(categories.items())],
        "",
        "## Aggregate metrics",
        "",
        "| Condition | Faithfulness | Answer relevance | Context precision | Context recall |",
        "|---|---:|---:|---:|---:|",
        _row("rerank on", on["overall"]),
        _row("rerank on comparison subset", on_subset),
        _row("rerank off subset", off["overall"]),
        "",
        "## Category-level metrics (rerank on)",
        "",
        "| Category | Faithfulness | Answer relevance | Context precision | Context recall |",
        "|---|---:|---:|---:|---:|",
        *[_row(name, values) for name, values in on["categories"].items()],
        "",
        "## Reranking comparison",
        "",
        f"Improved: {', '.join(result['comparison']['improved']) or 'none'}.  ",
        f"Unchanged: {', '.join(result['comparison']['unchanged']) or 'none'}.  ",
        f"Worsened: {', '.join(result['comparison']['worsened']) or 'none'}.",
        "",
        "## Out-of-corpus behavior and retrieval/generation disagreements",
        "",
        _out_of_corpus(on["cases"]),
        "Retrieval rank quality and answer quality are separate: correct context can still ",
        "yield an ",
        "unsupported or incomplete answer, while a rank change may leave the generated answer ",
        "materially unchanged. Per-case answers, reasons, and chunk orders are retained in JSON.",
        "",
        "## Judge bias and limitations",
        "",
        "LLM scores are not deterministic. Risks include position bias, verbosity preference, ",
        "same-family self-preference, judge-model mismatch, and sensitivity to golden wording. ",
        "Mitigations are fixed metric-specific rubrics, stable context order, hidden reranking ",
        "condition, structured scores, prompt/model version recording, persistent caching, and ",
        "category-level reporting. A fixed manual sample should still be inspected. This harness ",
        "does not evaluate agents, routing, tools, or generation outside this standalone RAG path.",
        "",
        f"Cache: {result['cache']['hits']} hits / {result['cache']['misses']} misses.",
        "",
    ]
    return "\n".join(lines)


def _row(label: str, values: dict[str, float]) -> str:
    return (
        f"| {label} | {values['faithfulness']:.4f} | {values['answer_relevance']:.4f} | "
        f"{values['context_precision']:.4f} | {values['context_recall']:.4f} |"
    )


def _out_of_corpus(rows: list[dict[str, Any]]) -> str:
    values = [row for row in rows if row["expected_out_of_corpus"]]
    return " ".join(f"{row['case_id']}: {row['candidate_answer']}" for row in values)


def _averages(rows: list[dict[str, Any]]) -> dict[str, float]:
    metrics = ("faithfulness", "answer_relevance", "context_precision", "context_recall")
    return {metric: fmean(row["metrics"][metric]["score"] for row in rows) for metric in metrics}
