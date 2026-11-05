"""Standard bandit baselines for O2 (panel review): LinUCB and linear Thompson sampling."""

from __future__ import annotations

from paim.o2_aprr.router import APRRRouter, RoutingConfig


class LinUCBRouter(APRRRouter):
    """Plain LinUCB (Li et al., 2010): always optimistic, no cost/latency/hallucination penalty.

    This is the classical baseline whose regret bound APRR inherits when gamma = 0.
    """

    name = "LinUCB"

    def __init__(self, pool=None, config=None) -> None:
        cfg = config or RoutingConfig()
        cfg = RoutingConfig(**{**cfg.__dict__, "lambda_cost": 0.0, "lambda_latency": 0.0, "lambda_halluc": 0.0})
        super().__init__(pool, cfg)

    def risk_posture(self, state, profile):  # type: ignore[override]
        return self.config.alpha


class LinTSRouter(APRRRouter):
    """Linear Thompson sampling (Agrawal & Goyal, 2013) with the same cost-aware objective as APRR.

    The exploration bonus is a posterior draw ``v * sigma * z`` (z ~ N(0, 1)) instead of the
    signed ``(alpha - gamma) * sigma`` term, isolating the effect of APRR's risk posture.
    """

    name = "Linear Thompson sampling"

    def __init__(self, pool=None, config=None, v: float = 0.35, seed: int = 7) -> None:
        import random as _r
        super().__init__(pool, config)
        self._v = v
        self._rng = _r.Random(seed)
        self._z = 0.0

    def risk_posture(self, state, profile):  # type: ignore[override]
        return self._v * self._z

    def score_agents(self, sub_goal_text, tools, state, profile, allowed=None):  # type: ignore[override]
        agents = list(allowed if allowed is not None else self.pool.routable())
        out, contexts = [], {}
        for agent in agents:
            self._z = self._rng.gauss(0.0, 1.0)
            sc, ctx = super().score_agents(sub_goal_text, tools, state, profile, [agent])
            out.extend(sc); contexts.update(ctx)
        out.sort(key=lambda s: (-s.score, s.agent_id))
        return out, contexts

__all__ = ["LinTSRouter", "LinUCBRouter"]
