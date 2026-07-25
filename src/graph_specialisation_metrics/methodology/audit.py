"""Soft numerical audits for the canonical methodology.

Every registered tolerance, invariance, and estimability check reports through :func:`audit_check`
instead of raising.  A failed check is logged once, accumulated in a run-scoped ledger, and carried
into the run's JSON records, so a long production run reports the defect rather than aborting on it.
``set_strict(True)`` (or ``MethodologyConfig.strict_audits``) restores fail-closed behaviour for
release verification.

This module is deliberately free of GRIT, torch, and plotting imports.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

from ..carriage.env import log


class AuditError(RuntimeError):
    """Raised for a failed audit only when strict auditing is enabled."""


class AuditWarning(UserWarning):
    """Category emitted for a soft (non-fatal) audit failure."""


@dataclass
class AuditFinding:
    """One failed audit, merged over every repeat within a scope."""

    name: str
    message: str
    observed: float | None = None
    tolerance: float | None = None
    context: dict[str, Any] = field(default_factory=dict)
    scope: str = ""
    count: int = 1

    def record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "message": self.message,
            "observed": None if self.observed is None else float(self.observed),
            "tolerance": None if self.tolerance is None else float(self.tolerance),
            "context": dict(self.context),
            "scope": self.scope,
            "count": int(self.count),
        }


class AuditLedger:
    """Findings recorded while one scope is active, merged by audit name."""

    def __init__(self, label: str = ""):
        self.label = str(label)
        self._entries: dict[str, AuditFinding] = {}

    def merge(self, finding: AuditFinding) -> AuditFinding:
        existing = self._entries.get(finding.name)
        if existing is None:
            self._entries[finding.name] = AuditFinding(
                name=finding.name,
                message=finding.message,
                observed=finding.observed,
                tolerance=finding.tolerance,
                context=dict(finding.context),
                scope=self.label or finding.scope,
                count=1,
            )
            return self._entries[finding.name]
        existing.count += 1
        if finding.observed is not None and (
            existing.observed is None or abs(finding.observed) > abs(existing.observed)
        ):
            # Keep the worst offender, not the first one.
            existing.observed = finding.observed
            existing.message = finding.message
            existing.context = dict(finding.context)
        return existing

    @property
    def findings(self) -> tuple[AuditFinding, ...]:
        return tuple(self._entries[name] for name in sorted(self._entries))

    def records(self) -> list[dict[str, Any]]:
        return [finding.record() for finding in self.findings]

    def clear(self) -> None:
        self._entries.clear()

    def __bool__(self) -> bool:
        return bool(self._entries)


_ROOT = AuditLedger("run")
_ACTIVE: list[AuditLedger] = [_ROOT]
_STRICT = False


def set_strict(value: bool) -> bool:
    """Select fail-closed (``True``) or report-and-continue (``False``) auditing."""

    global _STRICT
    previous, _STRICT = _STRICT, bool(value)
    return previous


def is_strict() -> bool:
    return bool(_STRICT)


def ledger() -> AuditLedger:
    """Root ledger holding every finding recorded in this process."""

    return _ROOT


def reset() -> None:
    _ROOT.clear()


@contextmanager
def audit_scope(label: str) -> Iterator[AuditLedger]:
    """Collect the findings recorded inside one task/seed run."""

    scope = AuditLedger(label)
    _ACTIVE.append(scope)
    try:
        yield scope
    finally:
        _ACTIVE.remove(scope)


def report(
    name: str,
    message: str,
    *,
    observed: float | None = None,
    tolerance: float | None = None,
    context: Mapping[str, Any] | None = None,
    strict: bool | None = None,
) -> AuditFinding:
    """Record a failed audit; raise only when strict auditing is enabled."""

    scope = _ACTIVE[-1].label if _ACTIVE else ""
    finding = AuditFinding(
        name=str(name),
        message=str(message),
        observed=None if observed is None else float(observed),
        tolerance=None if tolerance is None else float(tolerance),
        context=dict(context or {}),
        scope=scope,
    )
    if _STRICT if strict is None else bool(strict):
        raise AuditError(f"{finding.name}: {finding.message}")
    merged = finding
    for active in _ACTIVE:
        merged = active.merge(finding)
    if merged.count == 1:
        log(f"[audit] {finding.name}: {finding.message} (soft failure; run continues)")
        warnings.warn(f"{finding.name}: {finding.message}", AuditWarning, stacklevel=3)
    return merged


def audit_check(
    passed: bool,
    name: str,
    message: str,
    *,
    observed: float | None = None,
    tolerance: float | None = None,
    context: Mapping[str, Any] | None = None,
    strict: bool | None = None,
) -> bool:
    """Return whether the audit passed, recording a soft finding when it did not."""

    if bool(passed):
        return True
    report(
        name,
        message,
        observed=observed,
        tolerance=tolerance,
        context=context,
        strict=strict,
    )
    return False


def within_tolerance(
    observed: float,
    tolerance: float,
    name: str,
    message: str,
    *,
    context: Mapping[str, Any] | None = None,
    strict: bool | None = None,
) -> bool:
    """Audit one magnitude against a registered tolerance."""

    observed = float(observed)
    tolerance = float(tolerance)
    return audit_check(
        observed <= tolerance,
        name,
        f"{message} {observed:.3e} exceeds {tolerance:.3e}",
        observed=observed,
        tolerance=tolerance,
        context=context,
        strict=strict,
    )


def summary_lines(findings: Sequence[AuditFinding]) -> list[str]:
    lines = []
    for finding in findings:
        repeats = f" x{finding.count}" if finding.count > 1 else ""
        lines.append(f"  - {finding.name}{repeats}: {finding.message}")
    return lines


def log_summary(scope: AuditLedger, *, header: str) -> list[dict[str, Any]]:
    """Print and return the findings recorded in one scope."""

    records = scope.records()
    if not records:
        log(f"[audit] {header}: all registered audits passed")
        return records
    log(f"[audit] {header}: {len(records)} audit(s) failed softly")
    for line in summary_lines(scope.findings):
        log(line)
    return records
