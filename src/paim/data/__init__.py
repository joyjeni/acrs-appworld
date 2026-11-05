"""Datasets and live-data adapters.

``toolbench``   the ToolBench G1 retrieval benchmark plus the derived
                ToolBench-Sessions construction
``datagov``     live data.gov.in Agriculture resources wrapped as tools
``scenarios``   hand-authored Agriculture sessions for the live walkthrough
"""

from paim.data.datagov import (
    AGRICULTURE_RESOURCES,
    DataGovClient,
    DataGovExecutor,
    DataGovResource,
    agriculture_tools,
)
from paim.data.scenarios import (
    RESEARCHER_PROFILE,
    SCENARIOS,
    SCENARIOS_BY_ID,
    Scenario,
    scenario_arguments,
    scenario_tools,
)
from paim.data.toolbench import (
    ToolBenchBundle,
    build_sessions,
    load_bundle,
    parse_tool,
    sessionisation_report,
    sessionise,
)

__all__ = [
    "AGRICULTURE_RESOURCES",
    "RESEARCHER_PROFILE",
    "SCENARIOS",
    "SCENARIOS_BY_ID",
    "DataGovClient",
    "DataGovExecutor",
    "DataGovResource",
    "Scenario",
    "ToolBenchBundle",
    "agriculture_tools",
    "build_sessions",
    "load_bundle",
    "parse_tool",
    "scenario_arguments",
    "scenario_tools",
    "sessionisation_report",
    "sessionise",
]
