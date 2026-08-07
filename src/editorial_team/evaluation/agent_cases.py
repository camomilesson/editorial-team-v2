"""Declarative, fixed Editorial Team end-to-end evaluation cases."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from editorial_team.artifacts import ArtifactProducer, artifact_id_for


@dataclass(frozen=True)
class ArtifactFixture:
    fixture_id: str
    task_id: str
    created_at: datetime
    user_request: str
    content: str


@dataclass(frozen=True)
class ActiveTaskFixture:
    task_id: str
    original_request: str
    working_draft: str


@dataclass(frozen=True)
class CaseSetup:
    artifacts: tuple[ArtifactFixture, ...] = ()
    active_task: ActiveTaskFixture | None = None


@dataclass(frozen=True)
class ParameterExpectation:
    call_index: int
    field: str
    mode: Literal["equals", "timestamp_equals", "contains_terms", "non_empty"]
    expected: object


@dataclass(frozen=True)
class OutcomeExpectation:
    description: str
    require_response: bool = True
    required_response_terms: tuple[str, ...] = ()
    forbidden_response_terms: tuple[str, ...] = ()
    required_facts: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentEvaluationCase:
    case_id: str
    input_message: str
    setup: CaseSetup
    expected_trajectory: tuple[str, ...]
    acceptable_alternatives: tuple[tuple[str, ...], ...]
    outcome: OutcomeExpectation
    parameter_expectations: tuple[ParameterExpectation, ...] = ()
    score_retrieval: bool = False
    score_generation: bool = False
    golden_fixture_ids: tuple[str, ...] = ()
    generation_golden_answer: str | None = None

    @property
    def accepted_trajectories(self) -> tuple[tuple[str, ...], ...]:
        return (self.expected_trajectory, *self.acceptable_alternatives)


FIXED_NOW = datetime(2026, 8, 7, 12, tzinfo=UTC)
LAST_WEEK_FROM = "2026-07-27T00:00:00+00:00"
LAST_WEEK_TO = "2026-08-02T23:59:59+00:00"


def _artifact(
    fixture_id: str, task_id: str, created_at: datetime, request: str, content: str
) -> ArtifactFixture:
    return ArtifactFixture(fixture_id, task_id, created_at, request, content)


def load_agent_evaluation_cases() -> tuple[AgentEvaluationCase, ...]:
    """Return the immutable 12-case plan, defined before any model run occurs."""

    memory = _artifact(
        "aurora-memory",
        "eval-aurora-memory",
        datetime(2026, 7, 15, 9, tzinfo=UTC),
        "Prepare Aurora launch notes",
        "Aurora launches on September 14. The approved theme is Quiet Momentum.",
    )
    exact = _artifact(
        "cedar-draft",
        "eval-cedar-draft",
        datetime(2026, 6, 4, 10, tzinfo=UTC),
        "Draft the Cedar manifesto",
        "Cedar manifesto full draft. Build patiently, publish clearly, revise with evidence.",
    )
    latest_old = _artifact(
        "solstice-old",
        "eval-solstice-old",
        datetime(2026, 5, 1, 10, tzinfo=UTC),
        "Draft a Solstice update",
        "Old Solstice draft: the beta opens in May.",
    )
    latest_new = _artifact(
        "solstice-latest",
        "eval-solstice-latest",
        datetime(2026, 7, 30, 10, tzinfo=UTC),
        "Draft a Solstice update",
        "Latest Solstice draft: public preview opens in August.",
    )
    recent = _artifact(
        "recent-week",
        "eval-recent-week",
        datetime(2026, 7, 30, 14, tzinfo=UTC),
        "Write last week's Northstar note",
        "Northstar reached the design-partner milestone last week.",
    )
    historical = _artifact(
        "historical-orbit",
        "eval-historical-orbit",
        datetime(2026, 4, 3, 8, tzinfo=UTC),
        "Write the Orbit launch draft",
        "Orbit launch draft: a calm introduction to the research workspace.",
    )
    ember_active = ActiveTaskFixture(
        "eval-active-ember",
        "Write a factual two-paragraph Ember update for product leaders",
        "Ember helps product leaders compare editorial revisions. Its workspace keeps factual "
        "review notes beside each draft. The opening currently contains too much framing.",
    )
    rerank_a = _artifact(
        "atlas-map",
        "eval-atlas-map",
        datetime(2026, 7, 1, 8, tzinfo=UTC),
        "Atlas geography copy",
        "Atlas map map map: cartography notes for a printed classroom atlas.",
    )
    rerank_b = _artifact(
        "atlas-product",
        "eval-atlas-product",
        datetime(2026, 7, 2, 8, tzinfo=UTC),
        "Atlas product launch",
        "Atlas workspace launch positioning for collaborative editorial research teams.",
    )
    ambiguous_a = _artifact(
        "meridian-a",
        "eval-meridian-a",
        datetime(2026, 7, 10, 8, tzinfo=UTC),
        "Meridian customer announcement",
        "Meridian customer announcement draft for agencies.",
    )
    ambiguous_b = _artifact(
        "meridian-b",
        "eval-meridian-b",
        datetime(2026, 7, 11, 8, tzinfo=UTC),
        "Meridian product announcement",
        "Meridian product announcement draft for independent editors.",
    )
    normal_search = (
        ParameterExpectation(0, "query", "contains_terms", ("aurora",)),
        ParameterExpectation(0, "top_k", "equals", 5),
        ParameterExpectation(0, "prefer_recent", "equals", False),
        ParameterExpectation(0, "rerank", "equals", False),
    )
    return (
        AgentEvaluationCase(
            "chat_simple",
            "Hello! In one sentence, tell me how you can help an editorial team.",
            CaseSetup(),
            (),
            (),
            OutcomeExpectation(
                "A relevant conversational response", required_response_terms=("editor",)
            ),
        ),
        AgentEvaluationCase(
            "write_from_prompt",
            "Write a two-sentence launch blurb using only these facts: Lumen launches "
            "September 2; it helps editors compare revisions.",
            CaseSetup(),
            (),
            (),
            OutcomeExpectation(
                "A two-sentence supplied-facts blurb",
                required_response_terms=("lumen", "september"),
            ),
        ),
        AgentEvaluationCase(
            "write_with_memory",
            "Using our saved Aurora notes, write one sentence stating its launch date "
            "and approved theme.",
            CaseSetup((memory,)),
            ("search_corpus", "get_draft"),
            (),
            OutcomeExpectation(
                "Uses the saved Aurora facts",
                required_response_terms=("september 14", "quiet momentum"),
            ),
            normal_search,
            True,
            True,
            ("aurora-memory",),
            "Aurora launches on September 14 under the approved theme Quiet Momentum.",
        ),
        AgentEvaluationCase(
            "retrieve_exact_draft",
            "Retrieve the Cedar manifesto draft and rewrite the complete draft in a warmer voice.",
            CaseSetup((exact,)),
            ("search_corpus", "get_draft"),
            (),
            OutcomeExpectation(
                "Transforms the complete Cedar draft",
                required_facts=("exact_draft_transformation_completed",),
            ),
            (
                ParameterExpectation(0, "query", "contains_terms", ("cedar", "manifesto")),
                ParameterExpectation(
                    1,
                    "artifact_id",
                    "equals",
                    artifact_id_for("eval-cedar-draft", ArtifactProducer.WRITER),
                ),
            ),
            True,
            True,
            ("cedar-draft",),
            "Cedar encourages teams to build patiently, publish clearly, and revise with evidence.",
        ),
        AgentEvaluationCase(
            "retrieve_latest_topic",
            "Use the latest previous Solstice draft and turn the complete draft into a "
            "concise status email.",
            CaseSetup((latest_old, latest_new)),
            ("search_corpus", "get_draft"),
            (),
            OutcomeExpectation(
                "Uses the latest Solstice artifact", required_response_terms=("august",)
            ),
            (
                ParameterExpectation(0, "query", "contains_terms", ("solstice",)),
                ParameterExpectation(0, "prefer_recent", "equals", True),
                ParameterExpectation(
                    1,
                    "artifact_id",
                    "equals",
                    artifact_id_for("eval-solstice-latest", ArtifactProducer.WRITER),
                ),
            ),
            True,
            True,
            ("solstice-latest",),
            "The latest Solstice draft says the public preview opens in August.",
        ),
        AgentEvaluationCase(
            "retrieve_recent_period",
            "Find the Northstar note from last week and rewrite the complete note as one concise "
            "status sentence.",
            CaseSetup((recent,)),
            ("search_corpus", "get_draft"),
            (),
            OutcomeExpectation(
                "Uses only the fixed last-week window", required_response_terms=("design-partner",)
            ),
            (
                ParameterExpectation(0, "created_from", "timestamp_equals", LAST_WEEK_FROM),
                ParameterExpectation(0, "created_to", "timestamp_equals", LAST_WEEK_TO),
                ParameterExpectation(
                    1,
                    "artifact_id",
                    "equals",
                    artifact_id_for("eval-recent-week", ArtifactProducer.WRITER),
                ),
            ),
            True,
            True,
            ("recent-week",),
            "Northstar reached the design-partner milestone last week.",
        ),
        AgentEvaluationCase(
            "active_revision",
            "Revise the active draft to make the opening shorter while preserving its facts.",
            CaseSetup(active_task=ember_active),
            (),
            (),
            OutcomeExpectation(
                "Revises active state without historical retrieval",
                required_facts=("active_revision_applied",),
            ),
        ),
        AgentEvaluationCase(
            "historical_revision",
            "Revise the older Orbit launch draft in full; do not alter the unrelated "
            "active Ember task.",
            CaseSetup((historical,), active_task=ember_active),
            ("search_corpus", "get_draft"),
            (),
            OutcomeExpectation(
                "Revises Orbit without using the unrelated Ember draft as its source",
                required_response_terms=("orbit",),
                required_facts=("historical_orbit_selected", "ember_not_used_as_source"),
            ),
            (
                ParameterExpectation(0, "query", "contains_terms", ("orbit",)),
                ParameterExpectation(
                    1,
                    "artifact_id",
                    "equals",
                    artifact_id_for("eval-historical-orbit", ArtifactProducer.WRITER),
                ),
            ),
            True,
            True,
            ("historical-orbit",),
            "Orbit is a calm introduction to the research workspace.",
        ),
        AgentEvaluationCase(
            "search_no_match",
            "Find our previous Zephyr submarine launch copy and summarize its approved claim.",
            CaseSetup((memory,)),
            ("search_corpus",),
            (("search_corpus", "search_corpus"),),
            OutcomeExpectation(
                "Gracefully reports no matching material",
                forbidden_response_terms=("approved claim is",),
                required_facts=("no_match_safe",),
            ),
            (ParameterExpectation(0, "query", "contains_terms", ("zephyr", "submarine")),),
            False,
            False,
        ),
        AgentEvaluationCase(
            "retrieval_rerank",
            "Using prior Atlas work, summarize the product launch positioning for editorial "
            "research teams, not the geography project.",
            CaseSetup((rerank_a, rerank_b)),
            ("search_corpus", "get_draft"),
            (),
            OutcomeExpectation("Produces a response whose grounding is scored separately"),
            (
                ParameterExpectation(
                    0, "query", "contains_terms", ("atlas", "product", "editorial")
                ),
            ),
            True,
            True,
            ("atlas-product",),
            "Atlas is positioned as a workspace for collaborative editorial research teams.",
        ),
        AgentEvaluationCase(
            "ambiguous_reference",
            "Please improve the previous Meridian announcement, but first handle the "
            "ambiguity safely.",
            CaseSetup((ambiguous_a, ambiguous_b)),
            ("search_corpus",),
            (("search_corpus", "get_draft"),),
            OutcomeExpectation(
                "Clarifies or safely selects a relevant Meridian draft",
                required_response_terms=("meridian",),
            ),
            (ParameterExpectation(0, "query", "contains_terms", ("meridian", "announcement")),),
            True,
            False,
            ("meridian-a", "meridian-b"),
        ),
        AgentEvaluationCase(
            "tool_restraint",
            "An old topic was Aurora, but do not use prior work. Rewrite only this supplied "
            "text as a headline: Quiet tools help editors focus.",
            CaseSetup((memory,)),
            (),
            (),
            OutcomeExpectation(
                "Transforms supplied text without retrieval", required_response_terms=("editor",)
            ),
        ),
    )
