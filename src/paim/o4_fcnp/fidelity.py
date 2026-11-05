"""Fidelity estimation: how much task-relevant information survives pruning.

The estimator is deliberately *not* a learned scorer at this stage. A learned
predictor would need supervision that only exists after downstream execution,
and would make the controller's guarantee empirical rather than structural. So
fidelity is a composite of three quantities computable from the retained blocks
alone:

``coverage``          fraction of the open sub-goal's content terms still present
``criticality_mass``  share of total block criticality retained
``schema_integrity``  **hard gate** in [0,1]: for every tool the session has
                      committed to, is its signature retained *and* all of its
                      required parameters?

.. math::

    \\Phi = \\big(w_c\\,\\mathrm{coverage} + w_m\\,\\mathrm{criticality}\\big)
            \\cdot \\mathrm{schema\\_integrity}

The multiplicative gate is the important design choice. Prompt-compression
baselines score tokens by an information criterion, so they will happily delete
a required parameter name that appears once - the resulting call then fails at
execution time, which no perplexity-based fidelity measure can see. Making
schema integrity a *factor* rather than a summand means any pruning that breaks
a callable signature scores near zero, and the controller is therefore forced
to buy the tokens back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from paim.common.text import tokenize
from paim.common.types import ContextBlock, Tool


@dataclass(frozen=True)
class FidelityWeights:
    coverage: float = 0.55
    criticality: float = 0.45

    def to_dict(self) -> dict[str, float]:
        return {"coverage": self.coverage, "criticality": self.criticality}


@dataclass
class FidelityReport:
    fidelity: float
    coverage: float
    criticality_mass: float
    schema_integrity: float
    broken_tools: tuple[str, ...] = ()
    missing_terms: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "fidelity": round(self.fidelity, 5),
            "coverage": round(self.coverage, 5),
            "criticalityMass": round(self.criticality_mass, 5),
            "schemaIntegrity": round(self.schema_integrity, 5),
            "brokenTools": list(self.broken_tools),
            "missingTerms": list(self.missing_terms)[:12],
        }


class FidelityEstimator:
    def __init__(self, weights: FidelityWeights | None = None) -> None:
        self.weights = weights or FidelityWeights()

    def estimate(
        self,
        kept: Sequence[ContextBlock],
        original: Sequence[ContextBlock],
        goal_text: str,
        committed_tools: Iterable[Tool] = (),
    ) -> FidelityReport:
        goal_terms = set(tokenize(goal_text))
        kept_terms: set[str] = set()
        for b in kept:
            kept_terms |= set(tokenize(b.text))

        if goal_terms:
            covered = goal_terms & kept_terms
            coverage = len(covered) / len(goal_terms)
            missing = tuple(sorted(goal_terms - covered))
        else:
            coverage = 1.0
            missing = ()

        total_crit = sum(b.criticality for b in original) or 1.0
        criticality_mass = sum(b.criticality for b in kept) / total_crit

        integrity, broken = self.schema_integrity(kept, committed_tools)
        w = self.weights
        raw = w.coverage * coverage + w.criticality * criticality_mass
        return FidelityReport(
            fidelity=max(0.0, min(1.0, raw * integrity)),
            coverage=coverage,
            criticality_mass=min(1.0, criticality_mass),
            schema_integrity=integrity,
            broken_tools=broken,
            missing_terms=missing,
        )

    @staticmethod
    def schema_integrity(
        kept: Sequence[ContextBlock], committed_tools: Iterable[Tool]
    ) -> tuple[float, tuple[str, ...]]:
        """Fraction of committed tools that remain callable after pruning."""
        tools = list(committed_tools)
        if not tools:
            return 1.0, ()
        kept_ids = {b.block_id for b in kept}
        intact = 0
        broken: list[str] = []
        for tool in tools:
            needs = {f"{tool.doc_id}:head"}
            for i, p in enumerate(tool.parameters):
                if p.required:
                    needs.add(f"{tool.doc_id}:param{i}")
            if needs <= kept_ids:
                intact += 1
            else:
                broken.append(tool.key)
        return intact / len(tools), tuple(broken)


__all__ = ["FidelityEstimator", "FidelityReport", "FidelityWeights"]
