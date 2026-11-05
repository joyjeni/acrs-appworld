"""KCC-Sessions: the integrated public-problem benchmark (Kisan Call Centre).

The public problem
------------------
India's Kisan Call Centres (toll-free 1800-180-1551, Ministry of Agriculture
and Farmers Welfare) answered 33.5 lakh farmer calls in 2022-23 (Lok Sabha
reply, 2023), and the Ministry has announced chatbot and AI/ML decision-support
upgrades for them. Kisan e-Mitra (PM-KISAN) already answers ~13,000 queries a
day and is being extended to PMFBY, Kisan Credit Card and Soil Health Card.
The buyer exists and is spending now; what it lacks is an assistant whose
answers are *grounded in the government's own data*, *auditable*, and that
*knows which answers it is not allowed to give alone* (a pesticide dose, an
entitlement decision).

The dataset
-----------
Real farmer queries with expert answers: **KCC transcripts** (ICAR-IASRI
"KCC Chakshu" repository, Jan 2006 - Jul 2022; also listed on data.gov.in and
IndiaAI AIKosh). A 1.0M-row 2023 export with the same schema is mirrored at
``huggingface.co/datasets/Omegaindebt/Kisan_Call_Centre_Transcripts`` (parquet),
which is what the Kaggle notebook downloads; answers are multilingual (Hindi,
Tamil, ...). Known label noise: ``QueryType`` sometimes disagrees with the text
(e.g. a weather question filed under Post Harvest Preservation). It is used as
recorded and its noise is reported, never silently corrected. Fields: StateName, DistrictName, BlockName, CreatedOn, Season,
Sector, Category, Crop, QueryType, QueryText, KccAns.

How it becomes an ACRS benchmark (no labels invented)
-----------------------------------------------------
* **Tools** - the Kisan tool registry below: three live data.gov.in resources
  already wired in :mod:`paim.data.datagov` plus the national services a Farm
  Tele Advisor consults. Every service is a typed :class:`Tool`; services
  without a public API are marked ``live=False`` and executed by the
  simulator, which the thesis states explicitly.
* **O1 gold** - KCC's own ``QueryType`` label, mapped once (table
  ``QUERY_TYPE_TOOLS``) to the service that answers it. The mapping is a
  documented, reviewable design artefact, not a learned label.
* **Sessions** - KCC has no caller id, so a session is the ordered queries of
  one block on one day for one crop: the queue a block-level advisor works
  through. Called *pseudo-sessions* everywhere it matters.
* **Risk class** - from ``QueryType``: agronomic-dose and entitlement queries
  are recommend-only (REVIEW by a human advisor); informational ones may be
  auto-answered when G1 allows.
* **Distractors** - the 10,439 ToolBench endpoints can be added to the corpus
  (``with_distractors``) so retrieval is not trivially easy.
* **Answer reference** - ``KccAns`` is kept on every sub-goal for the
  LLM-agent stage (answer faithfulness against the expert reply).
"""

from __future__ import annotations

import csv
import gzip
import io
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from paim.common.types import Session, SubGoal, Tool, ToolParameter, UserProfile
from paim.data.datagov import agriculture_tools


@dataclass(frozen=True)
class KisanService:
    doc_id: str
    name: str
    api: str
    description: str
    owner: str
    live: bool
    params: tuple[tuple[str, str, bool], ...] = ()

    def as_tool(self) -> Tool:
        return Tool(
            doc_id=self.doc_id,
            category_name="Agriculture",
            tool_name=self.name,
            api_name=self.api,
            api_description=f"{self.description} Owner: {self.owner}.",
            method="GET",
            parameters=tuple(
                ToolParameter(name=n, type="string", description=d, required=r)
                for n, d, r in self.params
            ),
            source="kisan-registry" + ("" if self.live else "-simulated"),
        )


KISAN_SERVICES: tuple[KisanService, ...] = (
    KisanService(
        "kcc-imd-agromet", "IMD_Agromet_Advisory", "district_weather_forecast",
        "District-level 5-day weather forecast and agromet advisory: rainfall, temperature, "
        "humidity, wind, sowing and spraying windows.",
        "India Meteorological Department (Gramin Krishi Mausam Sewa)", False,
        (("district", "District name", True), ("state", "State name", False)),
    ),
    KisanService(
        "kcc-ncipm-pest", "Pest_Disease_Advisory", "crop_protection_advisory",
        "Pest and disease identification and integrated pest management advisory, "
        "insecticide and fungicide recommendation for a crop and symptom.",
        "ICAR-NCIPM / Directorate of Plant Protection", False,
        (("crop", "Crop name", True), ("symptom", "Observed symptom or pest", False)),
    ),
    KisanService(
        "kcc-shc-nutrient", "Soil_Health_Card", "nutrient_recommendation",
        "Soil test based fertilizer and nutrient dose recommendation (N, P, K, micronutrients) "
        "for a crop on the farmer's soil health card.",
        "Department of Agriculture and Farmers Welfare (Soil Health Card scheme)", False,
        (("crop", "Crop name", True), ("card_id", "Soil health card number", False)),
    ),
    KisanService(
        "kcc-fert-availability", "Fertilizer_Monitoring", "fertilizer_stock_by_district",
        "Fertilizer availability and retail stock (urea, DAP, MOP) by district and dealer.",
        "Department of Fertilizers (iFMS)", False,
        (("district", "District name", True), ("product", "Fertilizer product", False)),
    ),
    KisanService(
        "kcc-schemes", "Scheme_Eligibility", "scheme_information_and_status",
        "Government scheme information, eligibility and application status: PM-KISAN, PMFBY "
        "crop insurance, Kisan Credit Card, subsidies.",
        "Ministry of Agriculture and Farmers Welfare", False,
        (("scheme", "Scheme name", True), ("state", "State name", False)),
    ),
    KisanService(
        "kcc-varieties", "Crop_Varieties", "recommended_varieties",
        "Recommended crop varieties and hybrids by state, season and duration, with yield "
        "potential and resistance traits.",
        "ICAR / State Agricultural Universities", False,
        (("crop", "Crop name", True), ("state", "State name", False), ("season", "Season", False)),
    ),
    KisanService(
        "kcc-seeds", "Seed_Availability", "seed_dealer_stock",
        "Certified seed availability and dealers by district and crop.",
        "SeedNet India", False,
        (("crop", "Crop name", True), ("district", "District name", False)),
    ),
    KisanService(
        "kcc-practices", "Package_of_Practices", "cultivation_practices",
        "Package of practices: sowing time, seed rate, spacing, irrigation, intercultural "
        "operations, weed management and harvesting for a crop.",
        "ICAR / State Agricultural Universities", False,
        (("crop", "Crop name", True), ("state", "State name", False)),
    ),
)

# KCC QueryType -> services that answer it (O1 gold). The mandi-price and
# crop-statistics doc ids are the live data.gov.in tools from datagov.py.
QUERY_TYPE_TOOLS: Mapping[str, tuple[str, ...]] = {
    "weather": ("kcc-imd-agromet",),
    "market information": ("dg-mandi-daily-price",),
    "plant protection": ("kcc-ncipm-pest",),
    "nutrient management": ("kcc-shc-nutrient",),
    "fertilizer use and availability": ("kcc-fert-availability", "kcc-shc-nutrient"),
    "government schemes": ("kcc-schemes",),
    "varieties": ("kcc-varieties",),
    "seeds": ("kcc-seeds", "kcc-varieties"),
    "seeds and planting material": ("kcc-seeds", "kcc-varieties"),
    "cultural practices": ("kcc-practices",),
    "weed management": ("kcc-practices",),
    "water management": ("kcc-practices", "kcc-imd-agromet"),
    "field preparation": ("kcc-practices",),
    "sowing time and weather": ("kcc-practices", "kcc-imd-agromet"),
    "post harvest preservation": ("kcc-practices",),
    "organic farming": ("kcc-practices",),
    "credit": ("kcc-schemes",),
    "crop insurance": ("kcc-schemes",),
}

# Consequence class: who may decide. Doses and entitlements are a named
# human's call; the system prepares, cites and recommends.
QUERY_TYPE_RISK: Mapping[str, str] = {
    "plant protection": "agronomic-dose",
    "nutrient management": "agronomic-dose",
    "fertilizer use and availability": "agronomic-dose",
    "weed management": "agronomic-dose",
    "government schemes": "entitlement",
    "credit": "entitlement",
    "crop insurance": "entitlement",
}

_FIELDS = {
    "statename": "StateName", "districtname": "DistrictName", "blockname": "BlockName",
    "createdon": "CreatedOn", "season": "Season", "sector": "Sector", "category": "Category",
    "crop": "Crop", "querytype": "QueryType", "querytext": "QueryText", "kccans": "KccAns",
}


def kisan_tools() -> list[Tool]:
    return [s.as_tool() for s in KISAN_SERVICES] + agriculture_tools()


def normalise_query_type(qt: str) -> str:
    return re.sub(r"\s+", " ", (qt or "").replace("\t", " ").strip().lower())


def _rows(path: Path) -> Iterator[dict[str, str]]:
    if path.suffix == ".parquet":
        import pandas as pd  # optional; present on Kaggle/Colab

        df = pd.read_parquet(path)
        for raw in df.astype(str).to_dict("records"):
            yield {_FIELDS.get(k.strip().lower().replace(" ", ""), k): ("" if v in ("nan", "None") else v.strip())
                   for k, v in raw.items()}
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as fh:  # type: ignore[operator]
        reader = csv.DictReader(fh)
        for raw in reader:
            yield {_FIELDS.get(k.strip().lower().replace(" ", ""), k): (v or "").strip()
                   for k, v in raw.items() if k}


def load_kcc(
    paths: str | Path | Sequence[str | Path],
    limit: int | None = None,
    states: Iterable[str] | None = None,
    sector: str | None = "AGRICULTURE",
) -> list[dict[str, str]]:
    """Read KCC CSV(s) (plain or .gz), keep rows whose QueryType maps to a tool."""
    if isinstance(paths, (str, Path)):
        p = Path(paths)
        files = (
            sorted(p.glob("**/*.csv")) + sorted(p.glob("**/*.csv.gz")) + sorted(p.glob("**/*.parquet"))
            if p.is_dir() else [p]
        )
    else:
        files = [Path(x) for x in paths]
    want_states = {s.lower() for s in states} if states else None
    out: list[dict[str, str]] = []
    for f in files:
        for r in _rows(f):
            qt = normalise_query_type(r.get("QueryType", ""))
            if qt not in QUERY_TYPE_TOOLS or not r.get("QueryText"):
                continue
            if sector and r.get("Sector") and r["Sector"].strip().upper() != sector:
                continue
            if want_states and r.get("StateName", "").lower() not in want_states:
                continue
            out.append(r)
            if limit and len(out) >= limit:
                return out
    return out


def _day(created_on: str) -> str:
    return (created_on or "")[:10]


def build_kcc_sessions(
    rows: Sequence[Mapping[str, str]],
    max_turns: int = 6,
    min_turns: int = 1,
    profile: UserProfile | None = None,
) -> list[Session]:
    """Pseudo-sessions: (state, district, block, day, crop) in query order."""
    groups: "OrderedDict[tuple[str, ...], list[Mapping[str, str]]]" = OrderedDict()
    for r in rows:
        key = (r.get("StateName", ""), r.get("DistrictName", ""), r.get("BlockName", ""),
               _day(r.get("CreatedOn", "")), r.get("Crop", ""))
        groups.setdefault(key, []).append(r)
    sessions: list[Session] = []
    for i, (key, items) in enumerate(groups.items()):
        items = sorted(items, key=lambda x: x.get("CreatedOn", ""))[:max_turns]
        if len(items) < min_turns:
            continue
        state, district, block, day, crop = key
        prof = profile or UserProfile(
            email=f"advisor@{(district or 'block').lower().replace(' ', '-')}.kcc",
            sector="Agriculture",
            interests=tuple(x for x in (crop, state) if x),
            risk_tolerance=0.3,
            token_budget=2048,
        )
        sub_goals = []
        for t, r in enumerate(items):
            qt = normalise_query_type(r["QueryType"])
            text = r["QueryText"]
            if crop and crop.lower() not in text.lower():
                text = f"{text} (crop: {crop})"
            if state and state.lower() not in text.lower():
                text = f"{text} [{district}, {state}]"
            sg = SubGoal(
                sub_goal_id=f"kcc{i}:{t}",
                text=text,
                turn=t,
                gold_doc_ids=QUERY_TYPE_TOOLS[qt],
                risk_class=QUERY_TYPE_RISK.get(qt, "informational"),
            )
            sg.reference_answer = r.get("KccAns", "")  # type: ignore[attr-defined]
            sg.query_type = qt  # type: ignore[attr-defined]
            sub_goals.append(sg)
        sessions.append(
            Session(
                session_id=f"kcc-{i}",
                profile=prof,
                query=" ; ".join(r["QueryText"] for r in items),
                sub_goals=sub_goals,
                source="kcc",
            )
        )
    return sessions


def kcc_report(rows: Sequence[Mapping[str, str]], sessions: Sequence[Session]) -> dict[str, Any]:
    by_type: dict[str, int] = defaultdict(int)
    by_state: dict[str, int] = defaultdict(int)
    for r in rows:
        by_type[normalise_query_type(r["QueryType"])] += 1
        by_state[r.get("StateName", "")] += 1
    turns = [len(s.sub_goals) for s in sessions]
    risk = defaultdict(int)
    for s in sessions:
        for g in s.sub_goals:
            risk[g.risk_class] += 1
    return {
        "rows": len(rows),
        "sessions": len(sessions),
        "multiTurnSessions": sum(1 for t in turns if t > 1),
        "avgTurns": round(sum(turns) / max(1, len(turns)), 3),
        "queryTypes": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
        "states": dict(sorted(by_state.items(), key=lambda kv: -kv[1])[:15]),
        "riskClasses": dict(risk),
    }


def stratified_split(
    sessions: Sequence[Session], fraction: float = 0.5, seed: int = 20260905
) -> tuple[list[Session], list[Session]]:
    """Calibration / test split by session (never by sub-goal: no leakage across a queue)."""
    import random

    idx = list(range(len(sessions)))
    random.Random(seed).shuffle(idx)
    cut = int(len(idx) * fraction)
    a = [sessions[i] for i in sorted(idx[:cut])]
    b = [sessions[i] for i in sorted(idx[cut:])]
    return a, b


# A tiny, clearly synthetic sample used ONLY by unit tests and the offline
# smoke run, so the pipeline is exercisable without the real download. It is
# never reported as a result.
SYNTHETIC_SMOKE_ROWS: tuple[dict[str, str], ...] = tuple(
    dict(zip(
        ("StateName", "DistrictName", "BlockName", "CreatedOn", "Season", "Sector", "Category",
         "Crop", "QueryType", "QueryText", "KccAns"),
        row,
    ))
    for row in (
        ("KARNATAKA", "TUMKUR", "GUBBI", "2021-07-02T09:10:00", "KHARIF", "AGRICULTURE", "Cereals",
         "Paddy", "Weather", "rain forecast for next week sowing", "[synthetic]"),
        ("KARNATAKA", "TUMKUR", "GUBBI", "2021-07-02T09:40:00", "KHARIF", "AGRICULTURE", "Cereals",
         "Paddy", "Plant Protection", "leaf folder attack control measure", "[synthetic]"),
        ("KARNATAKA", "TUMKUR", "GUBBI", "2021-07-02T10:05:00", "KHARIF", "AGRICULTURE", "Cereals",
         "Paddy", "Market Information", "today mandi price of paddy", "[synthetic]"),
        ("PUNJAB", "LUDHIANA", "KHANNA", "2021-11-10T11:00:00", "RABI", "AGRICULTURE", "Cereals",
         "Wheat", "Varieties", "best wheat variety for late sowing", "[synthetic]"),
        ("PUNJAB", "LUDHIANA", "KHANNA", "2021-11-10T11:30:00", "RABI", "AGRICULTURE", "Cereals",
         "Wheat", "Government Schemes", "PM KISAN installment status", "[synthetic]"),
        ("PUNJAB", "LUDHIANA", "KHANNA", "2021-11-10T12:00:00", "RABI", "AGRICULTURE", "Cereals",
         "Wheat", "Nutrient Management", "DAP dose for wheat at sowing", "[synthetic]"),
    )
)


__all__ = [
    "KISAN_SERVICES",
    "QUERY_TYPE_RISK",
    "QUERY_TYPE_TOOLS",
    "SYNTHETIC_SMOKE_ROWS",
    "build_kcc_sessions",
    "kcc_report",
    "kisan_tools",
    "load_kcc",
    "normalise_query_type",
    "stratified_split",
]
