"""Live Agriculture scenarios for the end-to-end walkthrough.

These are the sessions used to demonstrate the integrated pipeline on *real*
data: the requester is ``joyjeni@gmail.com`` with ``sector="Agriculture"``, and
the tool corpus is the union of the ToolBench endpoints and the data.gov.in
Agriculture resources. Gold sets are the data.gov.in resources that genuinely
answer each sub-goal, so O1 is measurable here too.

Each scenario is a compound request of the same shape as a ToolBench G1
instruction, which keeps the two evaluation legs directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from paim.common.types import Session, SubGoal, Tool, UserProfile
from paim.data.datagov import DEFAULT_SECTOR, agriculture_tools

RESEARCHER_PROFILE = UserProfile(
    email="joyjeni@gmail.com",
    sector=DEFAULT_SECTOR,
    interests=(
        "commodity price",
        "mandi market",
        "crop production",
        "weather",
        "open government data",
    ),
    risk_tolerance=0.35,
    token_budget=1600,
)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    title: str
    query: str
    sub_goals: tuple[tuple[str, tuple[str, ...]], ...]
    filters: dict[str, dict[str, str]]
    notes: str = ""

    def to_session(self, profile: UserProfile | None = None) -> Session:
        return Session(
            session_id=self.scenario_id,
            profile=profile or RESEARCHER_PROFILE,
            query=self.query,
            sub_goals=[
                SubGoal(
                    sub_goal_id=f"{self.scenario_id}-g{i}",
                    text=text,
                    turn=i,
                    gold_doc_ids=gold,
                )
                for i, (text, gold) in enumerate(self.sub_goals)
            ],
            source="datagov",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenarioId": self.scenario_id,
            "title": self.title,
            "query": self.query,
            "subGoals": [{"text": t, "goldDocIds": list(g)} for t, g in self.sub_goals],
            "filters": self.filters,
            "notes": self.notes,
        }


MANDI = "dg-mandi-daily-price"

SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        scenario_id="agri-onion-price-watch",
        title="Onion price watch across mandis",
        query=(
            "I track onion procurement for a farmer producer organisation. Can you get me "
            "today's wholesale onion prices from the Indian mandis? Additionally, I need the "
            "same day's tomato prices so I can compare which crop is moving, and I would like "
            "the spread between the minimum and maximum modal price for each."
        ),
        sub_goals=(
            ("Can you get me today's wholesale onion prices from the Indian mandis?", (MANDI,)),
            (
                "I need the same day's tomato prices so I can compare which crop is moving",
                (MANDI,),
            ),
            (
                "I would like the spread between the minimum and maximum modal price for each",
                (MANDI,),
            ),
        ),
        filters={
            "agri-onion-price-watch-g0": {"commodity": "Onion"},
            "agri-onion-price-watch-g1": {"commodity": "Tomato"},
            "agri-onion-price-watch-g2": {"commodity": "Onion"},
        },
        notes=(
            "Exercises the redundancy penalty in O1: all three sub-goals are served by the same "
            "resource, so a stateless ranker would re-rank the identical endpoint three times "
            "with no adaptation, while SASR records it as served and shifts weight to the "
            "unserved comparison intent."
        ),
    ),
    Scenario(
        scenario_id="agri-paddy-procurement",
        title="Paddy procurement planning in Andhra Pradesh",
        query=(
            "I am planning paddy procurement in Andhra Pradesh for this week. Please give me the "
            "current paddy modal prices in Andhra Pradesh mandis. Also tell me which district "
            "reports the highest modal price today, and flag any market where the maximum price "
            "exceeds the state median by a wide margin."
        ),
        sub_goals=(
            ("Please give me the current paddy modal prices in Andhra Pradesh mandis", (MANDI,)),
            ("Also tell me which district reports the highest modal price today", (MANDI,)),
            (
                "flag any market where the maximum price exceeds the state median by a wide margin",
                (MANDI,),
            ),
        ),
        filters={
            "agri-paddy-procurement-g0": {"commodity": "Paddy(Common)", "state": "Andhra Pradesh"},
            "agri-paddy-procurement-g1": {"commodity": "Paddy(Common)", "state": "Andhra Pradesh"},
            "agri-paddy-procurement-g2": {"commodity": "Paddy(Common)", "state": "Andhra Pradesh"},
        },
        notes=(
            "The word 'price' makes every sub-goal high-stakes under O2's admission policy, so "
            "the light tier is excluded and a verifier is recruited into the O3 quorum. Good "
            "demonstration of the escalation path on live data."
        ),
    ),
    Scenario(
        scenario_id="agri-pulses-basket",
        title="Pulses basket price snapshot",
        query=(
            "Build me a price snapshot for the pulses basket. I need today's modal prices for "
            "Bengal Gram and for Arhar, and I also want to know how many distinct markets "
            "reported each of them so I can judge how reliable the average is."
        ),
        sub_goals=(
            ("I need today's modal prices for Bengal Gram", (MANDI,)),
            ("and for Arhar", (MANDI,)),
            (
                "I also want to know how many distinct markets reported each of them so I can "
                "judge how reliable the average is",
                (MANDI,),
            ),
        ),
        filters={
            "agri-pulses-basket-g0": {"commodity": "Bengal Gram(Gram)(Whole)"},
            "agri-pulses-basket-g1": {"commodity": "Arhar (Tur/Red Gram)(Whole)"},
            "agri-pulses-basket-g2": {"commodity": "Bengal Gram(Gram)(Whole)"},
        },
        notes=(
            "Short second sub-goal ('and for Arhar') tests the sub-goal segmenter's merge rule "
            "and the cold-start profile prior, since the fragment alone carries almost no "
            "retrievable signal."
        ),
    ),
)

SCENARIOS_BY_ID = {s.scenario_id: s for s in SCENARIOS}


def scenario_tools() -> list[Tool]:
    """The data.gov.in tools these scenarios can call."""
    return agriculture_tools()


def scenario_arguments(scenario: Scenario, sub_goal_id: str) -> dict[str, dict[str, Any]]:
    """Concrete call arguments for one sub-goal, keyed by tool doc id."""
    filters = scenario.filters.get(sub_goal_id, {})
    args: dict[str, Any] = {"format": "json", "limit": 8}
    for k, v in filters.items():
        args[f"filters[{k}]"] = v
    return {MANDI: args}


__all__ = [
    "RESEARCHER_PROFILE",
    "SCENARIOS",
    "SCENARIOS_BY_ID",
    "Scenario",
    "scenario_arguments",
    "scenario_tools",
]
