"""ToolBench loading, and the construction of *ToolBench-Sessions*.

Source
------
The official ToolBench tool-retrieval benchmark (Qin et al., 2023), G1 split:

* ``retrieval/G1/corpus.tsv``     10,439 unique RapidAPI endpoint documents
* ``retrieval/G1/test.query.txt`` 100 held-out instructions
* ``retrieval/G1/qrels.test.tsv`` 451 (query, endpoint) relevance judgements
* ``retrieval/G1/qrels.train.tsv`` training judgements, used *only* to fit the
  O1 co-occurrence graph

``scripts/prepare_data.py`` downloads these and writes the compact bundle under
``data/toolbench/`` that this module reads.

Why sessionisation
------------------
ToolBench instructions are compound by construction - a single query asks for
2-10 endpoints, phrased as "<intent A>. Additionally, <intent B>." The official
task scores one ranking against the union of gold endpoints, which makes it
impossible to measure whether a retriever exploits *what the session already
did*. That is precisely the claim O1 makes.

So we derive **ToolBench-Sessions** deterministically from the released data:

1. Segment the instruction into ordered sub-goals (``common.text.split_sub_goals``).
2. Assign every gold endpoint to the sub-goal it lexically matches best,
   scored on the endpoint's rendered document text.
3. Sub-goals that attract no gold endpoint are merged into their predecessor
   (they are usually politeness or context, not intents).
4. Emit the sub-goals in the order they appear as the turns of one session.

The union of per-turn gold sets is exactly the original qrels gold set, so
session-level metrics remain comparable to the published single-shot numbers -
asserted in ``tests/test_data.py``. No labels are invented: the construction
only *partitions* released judgements.
"""

from __future__ import annotations

import gzip
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from paim.common.text import bag, cosine, split_sub_goals
from paim.common.types import Session, SubGoal, Tool, ToolParameter, UserProfile

DEFAULT_DATA_DIR = Path(
    os.environ.get("PAIM_DATA_DIR", Path(__file__).resolve().parents[3] / "data")
)


def _param(raw: dict[str, Any], required: bool) -> ToolParameter:
    return ToolParameter(
        name=str(raw.get("name", "")),
        type=str(raw.get("type", "")),
        description=str(raw.get("description", "") or ""),
        default=str(raw.get("default", "") or ""),
        required=required,
    )


def parse_tool(doc_id: str, raw: dict[str, Any]) -> Tool:
    """Build a :class:`Tool` from one ToolBench corpus document."""
    params = [_param(p, True) for p in raw.get("required_parameters") or []]
    params += [_param(p, False) for p in raw.get("optional_parameters") or []]
    return Tool(
        doc_id=str(doc_id),
        category_name=str(raw.get("category_name", "") or "Uncategorised").strip(),
        tool_name=str(raw.get("tool_name", "") or "").strip(),
        api_name=str(raw.get("api_name", "") or "").strip(),
        api_description=" ".join(str(raw.get("api_description", "") or "").split()),
        method=str(raw.get("method", "GET") or "GET").strip(),
        parameters=tuple(params),
        source="toolbench",
    )


@dataclass
class ToolBenchBundle:
    tools: dict[str, Tool] = field(default_factory=dict)
    queries: list[dict[str, Any]] = field(default_factory=list)
    train_trajectories: list[list[str]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_tools(self) -> int:
        return len(self.tools)

    def query(self, qid: str) -> dict[str, Any]:
        for q in self.queries:
            if str(q["qid"]) == str(qid):
                return q
        raise KeyError(qid)

    def categories(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for t in self.tools.values():
            out[t.category_name] += 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def load_bundle(data_dir: str | Path | None = None) -> ToolBenchBundle:
    """Read the compact bundle written by ``scripts/prepare_data.py``."""
    root = Path(data_dir or DEFAULT_DATA_DIR) / "toolbench"
    corpus_path = root / "corpus.jsonl.gz"
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"ToolBench bundle not found at {corpus_path}. "
            "Run: python research/scripts/prepare_data.py"
        )

    tools: dict[str, Tool] = {}
    with gzip.open(corpus_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            tool = parse_tool(row["doc_id"], row)
            tools[tool.doc_id] = tool

    queries: list[dict[str, Any]] = []
    with (root / "queries.test.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                queries.append(json.loads(line))

    trajectories: list[list[str]] = []
    traj_path = root / "train_trajectories.jsonl"
    if traj_path.exists():
        with traj_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    trajectories.append(json.loads(line)["doc_ids"])

    meta_path = root / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return ToolBenchBundle(tools=tools, queries=queries, train_trajectories=trajectories, meta=meta)


# ---------------------------------------------------------------------------
# Sessionisation
# ---------------------------------------------------------------------------


def assign_gold_to_sub_goals(
    spans: Sequence[str], gold_doc_ids: Sequence[str], tools: dict[str, Tool]
) -> list[list[str]]:
    """Partition ``gold_doc_ids`` across ``spans`` by best lexical match."""
    if not spans:
        return []
    span_bags = [bag(s) for s in spans]
    buckets: list[list[str]] = [[] for _ in spans]
    for doc_id in gold_doc_ids:
        tool = tools.get(doc_id)
        if tool is None:
            buckets[0].append(doc_id)
            continue
        doc_bag = bag(tool.document_text())
        sims = [cosine(doc_bag, sb) for sb in span_bags]
        best = max(range(len(spans)), key=lambda i: (sims[i], -i))
        buckets[best].append(doc_id)
    return buckets


def sessionise(
    qid: str,
    query: str,
    gold_doc_ids: Sequence[str],
    tools: dict[str, Tool],
    profile: UserProfile | None = None,
    source: str = "toolbench",
) -> Session:
    """Turn one ToolBench instruction into a multi-turn session."""
    spans = split_sub_goals(query) or [query]
    buckets = assign_gold_to_sub_goals(spans, list(gold_doc_ids), tools)

    # Merge empty spans into their predecessor (or successor for span 0).
    merged_spans: list[str] = []
    merged_gold: list[list[str]] = []
    for i, span in enumerate(spans):
        gold = buckets[i]
        if not gold and merged_spans:
            merged_spans[-1] = f"{merged_spans[-1]} {span}".strip()
            continue
        if not gold and not merged_spans and i + 1 < len(spans):
            spans[i + 1] = f"{span} {spans[i + 1]}".strip()
            continue
        merged_spans.append(span)
        merged_gold.append(gold)

    if not merged_spans:
        merged_spans = [query]
        merged_gold = [list(gold_doc_ids)]

    sub_goals = [
        SubGoal(
            sub_goal_id=f"{qid}-g{i}",
            text=text,
            turn=i,
            gold_doc_ids=tuple(gold),
        )
        for i, (text, gold) in enumerate(zip(merged_spans, merged_gold))
    ]

    return Session(
        session_id=f"tb-{qid}",
        profile=profile or UserProfile(),
        query=query,
        sub_goals=sub_goals,
        source=source,
    )


def build_sessions(
    bundle: ToolBenchBundle,
    profile: UserProfile | None = None,
    min_turns: int = 1,
) -> list[Session]:
    sessions = []
    for q in bundle.queries:
        s = sessionise(
            str(q["qid"]), q["query"], q.get("gold_doc_ids", []), bundle.tools, profile
        )
        if len(s.sub_goals) >= min_turns:
            sessions.append(s)
    return sessions


def sessionisation_report(bundle: ToolBenchBundle) -> dict[str, Any]:
    """Diagnostics proving the construction is label-preserving."""
    sessions = build_sessions(bundle)
    total_gold = 0
    preserved = 0
    turn_hist: dict[int, int] = defaultdict(int)
    for s, q in zip(sessions, bundle.queries):
        gold = set(q.get("gold_doc_ids", []))
        total_gold += len(gold)
        preserved += len(set(s.gold_doc_ids) & gold)
        turn_hist[len(s.sub_goals)] += 1
    multi = sum(c for t, c in turn_hist.items() if t >= 2)
    return {
        "sessions": len(sessions),
        "multiTurnSessions": multi,
        "multiTurnFraction": round(multi / len(sessions), 4) if sessions else 0.0,
        "goldJudgements": total_gold,
        "goldPreserved": preserved,
        "labelPreserving": total_gold == preserved,
        "turnHistogram": {str(k): v for k, v in sorted(turn_hist.items())},
        "avgTurns": round(
            sum(k * v for k, v in turn_hist.items()) / max(1, sum(turn_hist.values())), 3
        ),
    }


def trajectories_from_qrels(qrels: Iterable[tuple[str, str]]) -> list[list[str]]:
    """Group (qid, doc_id) judgements into per-query co-called endpoint sets."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for qid, doc_id in qrels:
        grouped[qid].append(doc_id)
    return [sorted(set(v)) for v in grouped.values()]


__all__ = [
    "DEFAULT_DATA_DIR",
    "ToolBenchBundle",
    "assign_gold_to_sub_goals",
    "build_sessions",
    "load_bundle",
    "parse_tool",
    "sessionisation_report",
    "sessionise",
    "trajectories_from_qrels",
]
