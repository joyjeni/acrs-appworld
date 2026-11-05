"""The closed loop that makes O4 "fidelity-controlled" rather than "fixed-ratio".

Problem statement
-----------------
Choose, at every turn *t*, the smallest token budget :math:`B_t` such that the
retained context still clears a fidelity floor :math:`\\Phi^\\star`:

.. math::

    \\min_{B_t} B_t \\quad \\text{s.t.}\\quad \\Phi(B_t) \\ge \\Phi^\\star

Solved online with a **PI controller on the keep-fraction**. Let
:math:`\\kappa_t = B_t / N_t` be the fraction of the raw context we are willing
to pay for, and :math:`e_t = \\Phi^\\star - \\Phi_t` the fidelity deficit:

.. math::

    \\kappa_{t+1} = \\mathrm{clip}\\Big(
        \\kappa_t + K_p e_t + K_i \\sum_{s \\le t} e_s - K_s \\,\\mathbb{1}[e_t < -m]
        ,\\ \\kappa_{\\min},\\ 1\\Big)

* the proportional and integral terms *buy tokens back* whenever fidelity is
  below the floor - this is what the schema-integrity gate triggers when pruning
  breaks a required parameter;
* :math:`K_s` **squeezes** the budget whenever fidelity has been comfortably
  above the floor by more than the margin *m*, which is where the compression
  gain comes from;
* the integral term is anti-windup clamped so a permanently unachievable floor
  cannot drive :math:`\\kappa` to 1 and then stick there.

Because :math:`\\Phi` is non-decreasing in :math:`\\kappa` (a larger budget can
only admit more blocks, and the knapsack is greedy in a fixed value order), the
fixed point of this recursion is the smallest :math:`\\kappa` satisfying the
floor, up to the controller's dead band. That is the guarantee: not "compresses
by 50%", but "compresses as far as the fidelity floor allows, and provably
backs off when it does not". ``history`` records every step so the convergence
curve can be plotted, and ``verify_monotone_fidelity`` in the test suite checks
the monotonicity premise empirically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from paim.common.types import ContextBlock, Tool
from paim.o4_fcnp.pruner import FCNPPruner, PruneResult


@dataclass(frozen=True)
class ControllerConfig:
    fidelity_floor: float = 0.78
    margin: float = 0.06
    kp: float = 0.55
    ki: float = 0.12
    ks: float = 0.07
    keep_min: float = 0.18
    keep_init: float = 0.55
    integral_clamp: float = 1.5
    hard_ceiling_tokens: int = 8192

    def to_dict(self) -> dict[str, float]:
        return {
            "fidelityFloor": self.fidelity_floor,
            "margin": self.margin,
            "kp": self.kp,
            "ki": self.ki,
            "ks": self.ks,
            "keepMin": self.keep_min,
            "keepInit": self.keep_init,
        }


@dataclass
class ControlStep:
    turn: int
    keep_fraction_target: float
    budget: int
    original_tokens: int
    kept_tokens: int
    fidelity: float
    error: float
    integral: float
    action: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "keepTarget": round(self.keep_fraction_target, 5),
            "budget": self.budget,
            "originalTokens": self.original_tokens,
            "keptTokens": self.kept_tokens,
            "fidelity": round(self.fidelity, 5),
            "error": round(self.error, 5),
            "integral": round(self.integral, 5),
            "action": self.action,
        }


class FidelityController:
    """PI controller wrapping :class:`FCNPPruner`."""

    objective = "O4"
    name = "FCNP controller"

    def __init__(
        self,
        pruner: FCNPPruner | None = None,
        config: ControllerConfig | None = None,
    ) -> None:
        self.pruner = pruner or FCNPPruner()
        self.config = config or ControllerConfig()
        self.keep_target = self.config.keep_init
        self.integral = 0.0
        self.history: list[ControlStep] = []

    def reset(self) -> None:
        self.keep_target = self.config.keep_init
        self.integral = 0.0
        self.history = []

    def budget_for(self, original_tokens: int, ceiling: int | None = None) -> int:
        cap = ceiling or self.config.hard_ceiling_tokens
        return max(32, min(cap, int(round(original_tokens * self.keep_target))))

    def step(
        self,
        blocks: Sequence[ContextBlock],
        goal_text: str,
        committed_tools: Iterable[Tool] = (),
        turn: int = 0,
        ceiling: int | None = None,
    ) -> tuple[PruneResult, ControlStep]:
        cfg = self.config
        original_tokens = sum(b.tokens for b in blocks)
        budget = self.budget_for(original_tokens, ceiling)

        result = self.pruner.prune(
            blocks, budget, goal_text, committed_tools, current_turn=turn
        )
        phi = result.fidelity.fidelity
        error = cfg.fidelity_floor - phi

        prev_target = self.keep_target
        self.integral = max(
            -cfg.integral_clamp, min(cfg.integral_clamp, self.integral + error)
        )

        delta = cfg.kp * error + cfg.ki * self.integral
        if error < -cfg.margin:
            delta -= cfg.ks
            action = "squeeze"
        elif error > 0:
            action = "relax"
        else:
            action = "hold"

        self.keep_target = max(cfg.keep_min, min(1.0, self.keep_target + delta))

        step = ControlStep(
            turn=turn,
            keep_fraction_target=prev_target,
            budget=budget,
            original_tokens=original_tokens,
            kept_tokens=result.kept_tokens,
            fidelity=phi,
            error=error,
            integral=self.integral,
            action=action,
        )
        self.history.append(step)
        return result, step

    # -- diagnostics -------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        if not self.history:
            return {"steps": 0}
        fids = [h.fidelity for h in self.history]
        kept = sum(h.kept_tokens for h in self.history)
        orig = sum(h.original_tokens for h in self.history)
        floor = self.config.fidelity_floor
        return {
            "steps": len(self.history),
            "finalKeepTarget": round(self.keep_target, 5),
            "meanFidelity": round(sum(fids) / len(fids), 5),
            "minFidelity": round(min(fids), 5),
            "fidelityFloor": floor,
            "floorViolations": sum(1 for f in fids if f < floor - 1e-9),
            "floorSatisfactionRate": round(sum(1 for f in fids if f >= floor) / len(fids), 5),
            "overallCompression": round(1 - kept / orig, 5) if orig else 0.0,
            "tokensSaved": orig - kept,
            "trajectory": [h.to_dict() for h in self.history],
        }


@dataclass
class FixedBudgetController:
    """Baseline: constant keep-fraction, no feedback (the LLMLingua setting)."""

    pruner: FCNPPruner = field(default_factory=FCNPPruner)
    keep_fraction: float = 0.5
    fidelity_floor: float = 0.78
    history: list[ControlStep] = field(default_factory=list)
    name: str = "Fixed budget (no feedback)"

    def reset(self) -> None:
        self.history = []

    def step(self, blocks, goal_text, committed_tools=(), turn=0, ceiling=None):
        original = sum(b.tokens for b in blocks)
        budget = max(32, int(original * self.keep_fraction))
        result = self.pruner.prune(blocks, budget, goal_text, committed_tools, current_turn=turn)
        phi = result.fidelity.fidelity
        step = ControlStep(
            turn=turn,
            keep_fraction_target=self.keep_fraction,
            budget=budget,
            original_tokens=original,
            kept_tokens=result.kept_tokens,
            fidelity=phi,
            error=self.fidelity_floor - phi,
            integral=0.0,
            action="fixed",
        )
        self.history.append(step)
        return result, step

    def summary(self) -> dict[str, Any]:
        if not self.history:
            return {"steps": 0}
        fids = [h.fidelity for h in self.history]
        kept = sum(h.kept_tokens for h in self.history)
        orig = sum(h.original_tokens for h in self.history)
        return {
            "steps": len(self.history),
            "meanFidelity": round(sum(fids) / len(fids), 5),
            "minFidelity": round(min(fids), 5),
            "fidelityFloor": self.fidelity_floor,
            "floorViolations": sum(1 for f in fids if f < self.fidelity_floor - 1e-9),
            "floorSatisfactionRate": round(
                sum(1 for f in fids if f >= self.fidelity_floor) / len(fids), 5
            ),
            "overallCompression": round(1 - kept / orig, 5) if orig else 0.0,
            "tokensSaved": orig - kept,
            "trajectory": [h.to_dict() for h in self.history],
        }


def verify_monotone_fidelity(
    pruner: FCNPPruner,
    blocks: Sequence[ContextBlock],
    goal_text: str,
    committed_tools: Iterable[Tool] = (),
    steps: int = 12,
) -> tuple[bool, list[tuple[int, float]]]:
    """Empirically check that fidelity is non-decreasing in the token budget.

    The controller's fixed-point argument rests on this. Returns the curve so a
    violation can be inspected rather than merely reported.
    """
    committed_tools = list(committed_tools)
    total = sum(b.tokens for b in blocks) or 1
    curve: list[tuple[int, float]] = []
    for i in range(1, steps + 1):
        budget = max(16, int(total * i / steps))
        res = pruner.prune(blocks, budget, goal_text, committed_tools)
        curve.append((budget, res.fidelity.fidelity))
    monotone = all(curve[i][1] >= curve[i - 1][1] - 1e-9 for i in range(1, len(curve)))
    return monotone, curve


__all__ = [
    "ControlStep",
    "ControllerConfig",
    "FidelityController",
    "FixedBudgetController",
    "verify_monotone_fidelity",
]
