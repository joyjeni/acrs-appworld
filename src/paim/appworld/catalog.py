"""AppWorld API catalog: API docs -> typed ACRS tools, and static analysis of agent code.

AppWorld API docs carry the HTTP method of every endpoint, so the side effect of
an API is known exactly (GET = read; POST/PUT/PATCH/DELETE = state change). This
is what lets O3 spend consensus only on state-changing steps and lets G1 apply
approve-before-execute to destructive calls.

Nothing here imports ``appworld``; the world and task objects are duck-typed so
the logic is unit-testable offline (tests/test_appworld.py uses a fake world).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from paim.common.types import Tool, ToolParameter

READ_METHODS = {"GET", "HEAD", "OPTIONS"}
# apis.<app>.<api>(   -- also matches apis.app.api (  with whitespace
_CALL_RX = re.compile(r"\bapis\.([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)\s*\(")


@dataclass(frozen=True)
class ApiSpec:
    app: str
    api: str
    method: str
    description: str
    parameters: tuple[Mapping[str, Any], ...] = ()

    @property
    def key(self) -> str:
        return f"{self.app}.{self.api}"

    @property
    def writes(self) -> bool:
        return self.method.upper() not in READ_METHODS

    @property
    def deletes(self) -> bool:
        return self.method.upper() == "DELETE" or self.api.startswith(("delete_", "remove_"))

    def as_tool(self) -> Tool:
        return Tool(
            doc_id=self.key,
            category_name=self.app,
            tool_name=self.app,
            api_name=self.api,
            api_description=self.description,
            method=self.method.upper(),
            parameters=tuple(
                ToolParameter(
                    name=str(p.get("name", "")),
                    type=str(p.get("type", "")),
                    description=str(p.get("description", ""))[:160],
                    required=bool(p.get("required", False)),
                )
                for p in self.parameters
            ),
            source="appworld",
        )

    def render(self, max_params: int = 8) -> str:
        """Compact doc line injected into the prompt by O1."""
        ps = []
        for p in list(self.parameters)[:max_params]:
            flag = "" if p.get("required") else "?"
            ps.append(f"{p.get('name')}{flag}: {p.get('type')}")
        more = " ..." if len(self.parameters) > max_params else ""
        return f"apis.{self.app}.{self.api}({', '.join(ps)}{more})  [{self.method}] {self.description}"


@dataclass
class ApiCatalog:
    specs: dict[str, ApiSpec] = field(default_factory=dict)

    @classmethod
    def from_api_docs(cls, api_docs: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> "ApiCatalog":
        """Accepts AppWorld ``task.api_docs`` (app -> api -> doc dict)."""
        cat = cls()
        for app, apis in api_docs.items():
            if app == "api_docs":
                continue
            for api, doc in apis.items():
                spec = ApiSpec(
                    app=str(doc.get("app_name", app)),
                    api=str(doc.get("api_name", api)),
                    method=str(doc.get("method", "GET")).upper(),
                    description=str(doc.get("description", "")),
                    parameters=tuple(doc.get("parameters", ()) or ()),
                )
                cat.specs[spec.key] = spec
        return cat

    def tools(self) -> list[Tool]:
        return [s.as_tool() for s in self.specs.values()]

    def __getitem__(self, key: str) -> ApiSpec:
        return self.specs[key]

    def get(self, key: str) -> ApiSpec | None:
        return self.specs.get(key)

    def apps(self) -> list[str]:
        return sorted({s.app for s in self.specs.values()})


def extract_calls(code: str) -> list[str]:
    """API keys called by a code block, in order (``app.api``)."""
    return [f"{a}.{b}" for a, b in _CALL_RX.findall(code or "") if a != "api_docs"]


def write_calls(code: str, catalog: ApiCatalog) -> list[str]:
    out = []
    for k in extract_calls(code):
        spec = catalog.get(k)
        # Unknown APIs are treated as writes: fail safe.
        if spec is None or spec.writes:
            out.append(k)
    return out


def call_signatures(code: str, catalog: ApiCatalog) -> tuple[str, ...]:
    """Normalised signatures of state-changing calls, used to compare candidates.

    Two candidate programs that make the same write calls with the same literal
    arguments get the same signature even if their surrounding code differs.
    """
    sigs = []
    for m in re.finditer(r"\bapis\.([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)", code or ""):
        key = f"{m.group(1)}.{m.group(2)}"
        spec = catalog.get(key)
        if spec is not None and not spec.writes:
            continue
        args = re.sub(r"\s+", "", m.group(3))
        sigs.append(f"{key}({args})")
    return tuple(sigs)


def is_error(output: str) -> bool:
    o = output or ""
    return "Traceback" in o or "Execution failed" in o or "Exception:" in o


def iter_rendered(specs: Iterable[ApiSpec]) -> str:
    return "\n".join(s.render() for s in specs)


__all__ = [
    "ApiCatalog",
    "ApiSpec",
    "call_signatures",
    "extract_calls",
    "is_error",
    "iter_rendered",
    "write_calls",
]
