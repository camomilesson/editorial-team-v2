"""JSON-compatible result rendering for retrieval evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_results(result: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(result), encoding="utf-8")


def render_markdown(result: dict[str, Any]) -> str:
    corpus = result["corpus"]
    cases = result["case_set"]
    lines = [
        "# Retrieval evaluation report",
        "",
        "## Design and dataset",
        "",
        f"Fixed corpus: {corpus['artifact_count']} artifacts / {corpus['chunk_count']} chunks. ",
        f"Cases: {cases['case_count']}. Corpus SHA-256: `{corpus['sha256']}`. ",
        f"Case SHA-256: `{cases['sha256']}`.",
        "",
        "Metrics preserve the exact final SearchResult order. Precision uses requested k as the ",
        "denominator. MRR is truncated at k. Empty-golden cases are N/A and excluded.",
        "",
        "## Aggregate metrics",
        "",
        "| Rerank | k | Hit rate | Precision | Recall | MRR@k | nDCG@k |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, condition in result["conditions"].items():
        for k, row in condition["aggregate_metrics"].items():
            lines.append(
                f"| {mode} | {k} | {row['hit_rate']:.4f} | {row['precision']:.4f} | "
                f"{row['recall']:.4f} | {row['mrr']:.4f} | {row['ndcg']:.4f} |"
            )
    lines.extend(["", "## Per-case qualitative stage analysis", ""])
    for mode, condition in result["conditions"].items():
        lines.extend(
            [
                f"### Reranking {mode}",
                "",
                "| Case | Dense relevant ranks | BM25 relevant ranks | RRF relevant ranks | "
                "Final relevant ranks |",
                "|---|---|---|---|---|",
            ]
        )
        for case in condition["cases"]:
            positions = case["analysis"]["relevant_positions"]
            lines.append(
                f"| {case['case_id']} | {_positions(positions['dense'])} | "
                f"{_positions(positions['bm25'])} | {_positions(positions['rrf'])} | "
                f"{_positions(positions['final'])}; "
                f"{' '.join(case['analysis']['findings'])} |"
            )
        lines.append("")
    if result["reranking_deltas"]:
        lines.extend(
            [
                "## Reranking outcome by case",
                "",
                "| Case | Outcome at maximum k |",
                "|---|---|",
            ]
        )
        for case_id, delta in result["reranking_deltas"]["cases"].items():
            lines.append(f"| {case_id} | {delta['status']} |")
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "Compare dense and BM25 ranks to identify semantic and exact-term recovery; compare ",
            "RRF and final ranks to identify fusion and reranking changes. Neutral and negative ",
            "reranking ",
            "deltas are retained in the JSON rather than hidden.",
            "",
            "As k increases, recall and hit rate can rise while requested-k precision generally ",
            "falls because every additional slot remains in the denominator.",
            "",
            "Empty-golden cases: "
            f"{next(iter(result['conditions'].values()))['empty_golden_count']}.",
            "They are qualitative out-of-corpus probes only; no abstention threshold is ",
            "introduced.",
            "",
            "## Limitations",
            "",
            "Binary relevance and a fixed corpus cannot measure generation quality, agent ",
            "behavior, ",
            "or production-distribution drift. This milestone makes no such claims.",
            "",
        ]
    )
    return "\n".join(lines)


def _positions(values: dict[str, int | None]) -> str:
    if not values:
        return "N/A"
    return ", ".join(f"`{key}`: {value or '—'}" for key, value in values.items())
