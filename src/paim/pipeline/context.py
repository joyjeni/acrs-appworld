"""Assembly of the per-turn context that O4 then prunes.

The orchestrator never hands a raw string to the pruner. It assembles a typed
block list so that fidelity can be gated on schema integrity, and so that a
trace can show exactly which *kind* of information was dropped:

* the standing user query and the open sub-goal (hard-keep)
* the signature, description and each parameter of every candidate tool
* observations carried over from previous turns, oldest ones cheapest to drop
* a rendering of the O1 session memory, so history competes for budget on the
  same terms as everything else instead of being privileged
"""

from __future__ import annotations

from typing import Iterable, Sequence

from paim.common.session_state import SessionState
from paim.common.types import ContextBlock, Session, SubGoal, Tool, ToolInvocation


def memory_block(state: SessionState, turn: int, top_terms: int = 12) -> ContextBlock:
    top = sorted(state.memory.items(), key=lambda kv: -kv[1])[:top_terms]
    rendered = ", ".join(f"{k}:{v:.2f}" for k, v in top) or "(empty)"
    return ContextBlock(
        block_id=f"mem:t{turn}",
        kind="session_memory",
        text=f"session intent memory (turn {turn}): {rendered}",
        criticality=0.45,
        turn=turn,
    )


def observation_block(inv: ToolInvocation, turn: int, index: int) -> ContextBlock:
    return ContextBlock(
        block_id=f"obs:t{turn}:{index}",
        kind="observation",
        text=f"[{inv.tool_key}] {inv.observation or inv.error}",
        criticality=0.55 if inv.ok else 0.2,
        owner=inv.tool_key,
        turn=turn,
    )


def assemble(
    session: Session,
    sub_goal: SubGoal,
    candidate_tools: Sequence[Tool],
    state: SessionState,
    carried_observations: Sequence[ContextBlock] = (),
    include_memory: bool = True,
) -> list[ContextBlock]:
    """Build the full, unpruned context for one turn."""
    turn = sub_goal.turn
    blocks: list[ContextBlock] = [
        ContextBlock(
            block_id="query",
            kind="user_query",
            text=session.query,
            criticality=1.0,
            turn=0,
        ),
        ContextBlock(
            block_id=f"goal:{sub_goal.sub_goal_id}",
            kind="sub_goal",
            text=sub_goal.text,
            criticality=1.0,
            turn=turn,
        ),
    ]
    if include_memory:
        blocks.append(memory_block(state, turn))
    blocks.extend(carried_observations)
    for tool in candidate_tools:
        for b in tool.schema_blocks():
            b.turn = turn
            blocks.append(b)
    return blocks


def carry_forward(
    blocks: Iterable[ContextBlock], keep_kinds: frozenset[str] = frozenset({"observation"})
) -> list[ContextBlock]:
    """Observations that survived pruning become next turn's history."""
    return [b for b in blocks if b.kind in keep_kinds]


def render(blocks: Sequence[ContextBlock], max_chars: int = 4000) -> str:
    """Human-readable rendering of the assembled prompt, for the trace view."""
    out: list[str] = []
    total = 0
    for b in blocks:
        line = f"<{b.kind}> {b.text}"
        if total + len(line) > max_chars:
            out.append(f"... ({len(blocks) - len(out)} more blocks truncated for display)")
            break
        out.append(line)
        total += len(line)
    return "\n".join(out)


__all__ = ["assemble", "carry_forward", "memory_block", "observation_block", "render"]
