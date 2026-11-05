"""G1 - authority-bounded execution for ACRS (policy-as-code, guard, certified threshold).

See docs/AGENTIC_DESIGN_METHOD.md for how each piece maps onto the thesis.
"""

from paim.governance.authority import (
    ACRS_STAGES,
    ComponentType,
    Disposition,
    Oversight,
    Pattern,
    StageSpec,
    end_to_end_reliability,
    registry_violations,
)
from paim.governance.conformal import (
    CertifiedThreshold,
    expected_calibration_error,
    learn_then_test,
)
from paim.governance.guard import GuardedExecutor, InjectingExecutor, InjectionGuard
from paim.governance.layer import Adjudication, GovernanceConfig, GovernanceLayer, tool_side_effect
from paim.governance.policy import DecisionLog, PolicyDecision, PolicyEngine, Rule, to_rego
from paim.governance.release import (
    KillSwitch,
    ReleaseManifest,
    distribution_report,
    evaluation_gate,
    pass_hat_k,
)

__all__ = [
    "ACRS_STAGES", "Adjudication", "CertifiedThreshold", "ComponentType", "DecisionLog",
    "Disposition", "GovernanceConfig", "GovernanceLayer", "GuardedExecutor", "InjectingExecutor",
    "InjectionGuard", "KillSwitch", "Oversight", "Pattern", "PolicyDecision", "PolicyEngine",
    "ReleaseManifest", "Rule", "StageSpec", "distribution_report", "end_to_end_reliability",
    "evaluation_gate", "expected_calibration_error", "learn_then_test", "pass_hat_k",
    "registry_violations", "to_rego", "tool_side_effect",
]
