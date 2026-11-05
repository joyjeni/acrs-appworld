"""Live Indian open-government data (data.gov.in), Agriculture sector.

Why this is in a tool-use proposal
----------------------------------
ToolBench documents 10k+ RapidAPI endpoints but its keys are not
redistributable, so ToolBench trajectories are necessarily simulated. A
proposal that only ever runs a simulator cannot claim its pipeline works. The
data.gov.in Open Government Data platform gives us the missing leg: a real,
key-authenticated, rate-limited, schema-described API family in a single
sector, callable from the same orchestrator with no code changes because its
resources are wrapped as the same :class:`~paim.common.types.Tool` objects.

Resources used (sector = Agriculture, Ministry of Agriculture and Farmers
Welfare). ``9ef84268-...`` is the daily mandi (wholesale market) price feed,
which is updated every day and therefore exercises fresh data on every run.

Auth
----
``api.data.gov.in`` needs an ``api-key``. The platform publishes a visualization
key on the public resource HTML (``field_datafile_url``), which is the default
here so the demo can read live rows with no setup; set ``DATA_GOV_IN_API_KEY``
to use a personal key. **There is no snapshot or dummy fallback.** If the
network or the key is unavailable the client returns ``ok=False``. Every live
record is tagged ``_source=live``.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from paim.common.types import Tool, ToolInvocation, ToolParameter
from paim.data.toolbench import DEFAULT_DATA_DIR

API_ROOT = "https://api.data.gov.in/resource"
# Visualization key published on the public AGMARKNET resource page. Not a secret.
SAMPLE_API_KEY = "579b464db66ec23bdd000001cdc3b564546246a772a26393094f5645"
DEFAULT_SECTOR = "Agriculture"


@dataclass(frozen=True)
class DataGovResource:
    """One data.gov.in resource, described well enough to become a Tool."""

    resource_id: str
    doc_id: str
    title: str
    description: str
    sector: str
    ministry: str
    filter_fields: tuple[tuple[str, str], ...] = ()
    required_fields: tuple[str, ...] = ()

    def as_tool(self) -> Tool:
        params = [
            ToolParameter(
                name="format",
                type="STRING",
                description="Response format; the pipeline always requests json.",
                default="json",
                required=True,
            ),
            ToolParameter(
                name="limit",
                type="NUMBER",
                description="Page size (1-1000).",
                default="10",
                required=True,
            ),
        ]
        for fname, fdesc in self.filter_fields:
            params.append(
                ToolParameter(
                    name=f"filters[{fname}]",
                    type="STRING",
                    description=fdesc,
                    required=fname in self.required_fields,
                )
            )
        return Tool(
            doc_id=self.doc_id,
            category_name=self.sector,
            tool_name="data.gov.in",
            api_name=self.title,
            api_description=self.description,
            method="GET",
            parameters=tuple(params),
            source="datagov",
        )


AGRICULTURE_RESOURCES: tuple[DataGovResource, ...] = (
    DataGovResource(
        resource_id="9ef84268-d588-465a-a308-a864a43d0070",
        doc_id="dg-mandi-daily-price",
        title="Current Daily Price of Various Commodities from Various Markets (Mandi)",
        description=(
            "Daily wholesale arrival and price data (minimum, maximum and modal price per "
            "quintal) for agricultural commodities across Indian APMC mandis, published by the "
            "Department of Agriculture and Farmers Welfare. Filterable by state, district, "
            "market, commodity, variety, grade and arrival date."
        ),
        sector=DEFAULT_SECTOR,
        ministry="Ministry of Agriculture and Farmers Welfare",
        filter_fields=(
            ("state", "State name, e.g. Andhra Pradesh."),
            ("district", "District name."),
            ("market", "APMC market (mandi) name."),
            ("commodity", "Commodity name, e.g. Paddy(Common), Onion, Tomato."),
            ("variety", "Commodity variety."),
            ("grade", "Quality grade, e.g. FAQ."),
        ),
        required_fields=("commodity",),
    ),
    DataGovResource(
        resource_id="35be999b-0208-4354-b557-f6ca9a5355de",
        doc_id="dg-crop-production",
        title="District-wise, season-wise crop production statistics from 1997",
        description=(
            "District-wise, season-wise, crop-wise area and production published by the "
            "Ministry of Agriculture and Farmers Welfare on data.gov.in."
        ),
        sector=DEFAULT_SECTOR,
        ministry="Ministry of Agriculture and Farmers Welfare",
        filter_fields=(
            ("state_name", "State name, e.g. Karnataka."),
            ("crop", "Crop name, e.g. Rice."),
            ("season", "Season, e.g. Kharif or Rabi."),
            ("crop_year", "Agricultural year."),
        ),
    ),
    DataGovResource(
        resource_id="8196f6cc-83ff-4b56-8581-2630de9d4a5e",
        doc_id="dg-imd-rainfall",
        title="All India Area Weighted Monthly, Seasonal And Annual Rainfall (in mm)",
        description="IMD all-India rainfall series republished on data.gov.in, used with crop statistics in agri-climate work.",
        sector=DEFAULT_SECTOR,
        ministry="Ministry of Earth Sciences / India Meteorological Department",
        filter_fields=(("year", "Calendar year."),),
    ),
)

RESOURCES_BY_DOC_ID = {r.doc_id: r for r in AGRICULTURE_RESOURCES}


def agriculture_tools() -> list[Tool]:
    return [r.as_tool() for r in AGRICULTURE_RESOURCES]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


@dataclass
class DataGovResponse:
    ok: bool
    source: str            # "live" | "none"
    records: list[dict[str, Any]] = field(default_factory=list)
    total: int = 0
    title: str = ""
    latency_ms: float = 0.0
    error: str = ""
    url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "source": self.source,
            "records": self.records,
            "total": self.total,
            "title": self.title,
            "latencyMs": round(self.latency_ms, 2),
            "error": self.error,
            "url": self.url,
        }


class DataGovClient:
    """Minimal stdlib client. Live data.gov.in only — no snapshot/dummy rows."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 12.0,
        data_dir: str | Path | None = None,
        allow_network: bool | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("DATA_GOV_IN_API_KEY") or SAMPLE_API_KEY
        self.timeout = timeout
        self.snapshot_dir = Path(data_dir or DEFAULT_DATA_DIR) / "datagov"
        if allow_network is None:
            allow_network = os.environ.get("PAIM_OFFLINE", "").lower() not in {"1", "true", "yes"}
        self.allow_network = allow_network
        self.last_source = "unknown"

    # -- helpers -----------------------------------------------------------
    def build_url(self, resource_id: str, params: dict[str, Any]) -> str:
        query = {"api-key": self.api_key, "format": "json"}
        query.update({k: v for k, v in params.items() if v not in (None, "")})
        return f"{API_ROOT}/{resource_id}?{urllib.parse.urlencode(query)}"

    def snapshot_path(self, doc_id: str) -> Path:
        return self.snapshot_dir / f"{doc_id}.snapshot.json"

    def load_snapshot(self, doc_id: str) -> DataGovResponse:
        return DataGovResponse(
            ok=False,
            source="none",
            error="snapshot loading disabled — Agriculture lab reads live data.gov.in only",
        )

    # -- main --------------------------------------------------------------
    def fetch(
        self,
        doc_id: str,
        limit: int = 10,
        offset: int = 0,
        filters: dict[str, str] | None = None,
    ) -> DataGovResponse:
        resource = RESOURCES_BY_DOC_ID.get(doc_id)
        if resource is None:
            return DataGovResponse(ok=False, source="none", error=f"unknown resource {doc_id}")

        params: dict[str, Any] = {"limit": limit, "offset": offset}
        for k, v in (filters or {}).items():
            key = k if k.startswith("filters[") else f"filters[{k}]"
            params[key] = v
        url = self.build_url(resource.resource_id, params)

        if not self.allow_network:
            return DataGovResponse(
                ok=False,
                source="none",
                error="network disabled — Agriculture lab reads live data.gov.in only (no snapshot/dummy rows)",
                url=url,
            )

        started = time.perf_counter()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "paim-research/0.1"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            records = payload.get("records", []) or []
            if not records:
                self.last_source = "none"
                return DataGovResponse(
                    ok=False,
                    source="none",
                    error="data.gov.in returned zero records; refusing to fabricate rows",
                    latency_ms=(time.perf_counter() - started) * 1000,
                    url=url,
                )
            for r in records:
                r.setdefault("_source", "live")
            self.last_source = "live"
            return DataGovResponse(
                ok=True,
                source="live",
                records=records,
                total=int(payload.get("total", len(records)) or 0),
                title=str(payload.get("title", "")),
                latency_ms=(time.perf_counter() - started) * 1000,
                url=url,
            )
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, OSError) as exc:
            self.last_source = "none"
            return DataGovResponse(
                ok=False,
                source="none",
                error=f"live fetch failed ({type(exc).__name__}: {exc}); no dummy rows substituted",
                latency_ms=(time.perf_counter() - started) * 1000,
                url=url,
            )

    def write_snapshot(self, doc_id: str, limit: int = 200, filters: dict[str, str] | None = None) -> Path:
        """Optional local capture of a live response. Never used as an answer source."""
        resp = self.fetch(doc_id, limit=limit, filters=filters)
        if not resp.ok or resp.source != "live":
            raise RuntimeError(f"cannot refresh snapshot: {resp.error or 'no live data'}")
        path = self.snapshot_path(doc_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "docId": doc_id,
                    "resourceId": RESOURCES_BY_DOC_ID[doc_id].resource_id,
                    "title": resp.title,
                    "total": resp.total,
                    "capturedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "url": resp.url.replace(self.api_key, "<API_KEY>"),
                    "records": resp.records,
                },
                indent=1,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return path


# ---------------------------------------------------------------------------
# Executor: plugs data.gov.in into the O3 mesh
# ---------------------------------------------------------------------------


class DataGovExecutor:
    """A :class:`~paim.o3_mncd.execution.ToolExecutor` backed by real HTTP."""

    name = "data.gov.in"
    live = True

    def __init__(self, client: DataGovClient | None = None, default_limit: int = 8) -> None:
        self.client = client or DataGovClient()
        self.default_limit = default_limit
        self.calls = 0
        self.live_calls = 0
        self.snapshot_calls = 0

    def execute(self, tool: Tool, arguments: dict[str, Any], nonce: str = "") -> ToolInvocation:
        self.calls += 1
        if tool.source != "datagov":
            return ToolInvocation(
                tool_key=tool.key,
                doc_id=tool.doc_id,
                arguments=arguments,
                ok=False,
                error=f"{self.name} executor cannot call a {tool.source} tool",
            )

        filters = {
            k.replace("filters[", "").rstrip("]"): v
            for k, v in arguments.items()
            if k.startswith("filters[")
        }
        limit = int(arguments.get("limit", self.default_limit) or self.default_limit)
        resp = self.client.fetch(tool.doc_id, limit=limit, filters=filters)
        if resp.source == "live":
            self.live_calls += 1
        elif resp.source == "snapshot":
            self.snapshot_calls += 1

        if not resp.ok:
            return ToolInvocation(
                tool_key=tool.key,
                doc_id=tool.doc_id,
                arguments=arguments,
                ok=False,
                latency_ms=resp.latency_ms,
                error=resp.error or "data.gov.in request failed",
            )

        # The observation is a compact, deterministic digest of the response so
        # that two peers reading the same records form the same claim key.
        observation = json.dumps(
            {
                "endpoint": tool.key,
                "source": resp.source,
                "title": resp.title,
                "matched": resp.total,
                "returned": len(resp.records),
                "sample": summarise_mandi(resp.records),
            },
            sort_keys=True,
        )
        return ToolInvocation(
            tool_key=tool.key,
            doc_id=tool.doc_id,
            arguments=arguments,
            ok=True,
            latency_ms=resp.latency_ms,
            observation=observation,
        )

    def stats(self) -> dict[str, Any]:
        return {
            "executor": self.name,
            "calls": self.calls,
            "liveCalls": self.live_calls,
            "snapshotCalls": self.snapshot_calls,
        }


def summarise_mandi(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic numeric summary of mandi price records."""
    rows = [r for r in records if r.get("modal_price") not in (None, "")]
    if not rows:
        return {"n": 0}
    prices = sorted(float(r["modal_price"]) for r in rows)
    n = len(prices)
    commodities = sorted({str(r.get("commodity", "")) for r in rows if r.get("commodity")})
    states = sorted({str(r.get("state", "")) for r in rows if r.get("state")})
    return {
        "n": n,
        "minModal": prices[0],
        "maxModal": prices[-1],
        "medianModal": prices[n // 2],
        "meanModal": round(sum(prices) / n, 2),
        "commodities": commodities[:8],
        "states": states[:8],
        "arrivalDates": sorted({str(r.get("arrival_date", "")) for r in rows})[:3],
    }


__all__ = [
    "AGRICULTURE_RESOURCES",
    "API_ROOT",
    "DEFAULT_SECTOR",
    "RESOURCES_BY_DOC_ID",
    "SAMPLE_API_KEY",
    "DataGovClient",
    "DataGovExecutor",
    "DataGovResource",
    "DataGovResponse",
    "agriculture_tools",
    "summarise_mandi",
]
