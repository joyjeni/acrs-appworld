"""Tiny dense linear algebra on plain Python lists.

The router needs only rank-1 inverse updates on a 13x13 matrix, so this avoids
a NumPy dependency in the hot path. Keeping it dependency-free is also what
makes the TypeScript port in ``lib/paim`` a line-by-line translation, which is
what the cross-language parity test checks.
"""

from __future__ import annotations

from typing import Sequence

Vector = list[float]
Matrix = list[list[float]]


def identity(d: int, scale: float = 1.0) -> Matrix:
    return [[scale if i == j else 0.0 for j in range(d)] for i in range(d)]


def matvec(m: Matrix, v: Sequence[float]) -> Vector:
    return [sum(row[j] * v[j] for j in range(len(v))) for row in m]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def sherman_morrison_update(a_inv: Matrix, x: Sequence[float]) -> Matrix:
    """Return the inverse of ``A + x x^T`` given ``A^-1``.

    (A + x x^T)^-1 = A^-1 - (A^-1 x)(x^T A^-1) / (1 + x^T A^-1 x)
    """
    ax = matvec(a_inv, x)
    denom = 1.0 + dot(x, ax)
    if abs(denom) < 1e-12:  # pragma: no cover - guarded numerically
        return [row[:] for row in a_inv]
    d = len(a_inv)
    return [[a_inv[i][j] - ax[i] * ax[j] / denom for j in range(d)] for i in range(d)]


def quad_form(a_inv: Matrix, x: Sequence[float]) -> float:
    """x^T A^-1 x, clamped at zero against floating-point drift."""
    return max(0.0, dot(x, matvec(a_inv, x)))


def l2(v: Sequence[float]) -> float:
    return sum(x * x for x in v) ** 0.5


__all__ = [
    "Matrix",
    "Vector",
    "dot",
    "identity",
    "l2",
    "matvec",
    "quad_form",
    "sherman_morrison_update",
]
