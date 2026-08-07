"""CLI for Stage 5 aggregation, stored-trace rescoring, and MLflow feedback."""

from __future__ import annotations

import argparse
from pathlib import Path

from editorial_team.evaluation.agent_cases import load_agent_evaluation_cases
from editorial_team.evaluation.agent_reporting import (
    aggregate_campaign,
    load_campaign_manifest,
    load_run_results,
    log_campaign_feedback,
    log_campaign_safety_feedback,
    rescore_stored_traces,
    write_campaign_summary,
)
from editorial_team.evaluation.generation_judges import StructuredGenerationJudge
from editorial_team.gemini import create_gemini_client_from_env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate and log an HW3 agent campaign")
    parser.add_argument("--results", type=Path, required=True, help="Raw Stage 4 result JSON")
    parser.add_argument("--summary", type=Path, required=True, help="Derived summary JSON")
    parser.add_argument("--manifest", type=Path, help="Tracking-store campaign manifest")
    parser.add_argument("--case", action="append", dest="case_ids", help="Include this case")
    parser.add_argument("--log-feedback", action="store_true")
    parser.add_argument("--log-safety-feedback", action="store_true")
    parser.add_argument("--rescore-stored", action="store_true")
    parser.add_argument(
        "--rescore-generation",
        action="store_true",
        help="Make judge calls over stored outputs; never reruns the agent",
    )
    args = parser.parse_args(argv)

    results = load_run_results(args.results)
    selected = frozenset(args.case_ids) if args.case_ids else None
    if selected is not None:
        results = tuple(result for result in results if result.case_id in selected)
        if {result.case_id for result in results} != set(selected):
            parser.error("--case contains an unknown or absent case ID")
    manifest = load_campaign_manifest(args.manifest) if args.manifest else None
    if (
        args.log_feedback
        or args.log_safety_feedback
        or args.rescore_stored
        or args.rescore_generation
    ) and manifest is None:
        parser.error("--manifest is required for feedback or stored-trace rescoring")
    if args.rescore_generation and not args.rescore_stored:
        parser.error("--rescore-generation requires --rescore-stored")

    if args.rescore_stored:
        judge = (
            StructuredGenerationJudge(create_gemini_client_from_env())
            if args.rescore_generation
            else None
        )
        cases = load_agent_evaluation_cases()
        results = rescore_stored_traces(
            results,
            cases,
            manifest,
            generation_judge=judge,
            case_ids=selected,
        )
    summary = aggregate_campaign(results)
    write_campaign_summary(args.summary, summary)
    if args.log_feedback:
        logged = log_campaign_feedback(results, manifest)
        print(f"Logged {logged} feedback assessments")
    if args.log_safety_feedback:
        safety = log_campaign_safety_feedback(results, manifest)
        print(
            f"Safety traces: {safety.evaluated} evaluated, "
            f"{safety.unevaluable} unevaluable, {safety.flagged} flagged"
        )
    print(
        f"Aggregated {summary.suite.total_runs} runs across {summary.suite.total_scenarios} cases"
    )
    print(f"Summary: {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
