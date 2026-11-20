"""API-cost authorization, accounting, and periodic campaign reports.

The ledger is append-only JSONL and uses a file lock because candidate arms may finish on
different workers. Unknown price is never interpreted as free: it is a review-required estimate.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class BudgetPolicy:
    """Frozen campaign and owner limits in US dollars."""

    monthly_limit_usd: float
    reserve_usd: float
    campaign_limit_usd: float
    daily_limit_usd: float
    large_job_usd: float
    burst_fraction: float = 0.15
    report_interval_hours: int = 12

    def __post_init__(self) -> None:
        values = (
            self.monthly_limit_usd,
            self.reserve_usd,
            self.campaign_limit_usd,
            self.daily_limit_usd,
            self.large_job_usd,
        )
        if any(value < 0 for value in values):
            raise ValueError("budget values must be non-negative")
        if self.reserve_usd > self.monthly_limit_usd:
            raise ValueError("budget reserve cannot exceed the monthly limit")
        if not 0 <= self.burst_fraction <= 1:
            raise ValueError("budget burst_fraction must be between zero and one")
        if self.report_interval_hours < 1:
            raise ValueError("budget report interval must be at least one hour")


@dataclass(frozen=True)
class CostRecord:
    """One projected or completed hosted operation."""

    owner: str
    project: str
    campaign: str
    stage: str
    candidate: str
    provider: str
    model: str
    estimated_usd: float | None
    confirmed_usd: float | None = None
    requests: int = 0
    input_units: int = 0
    output_units: int = 0
    retries: int = 0
    retry_reason: str = ""
    status: Literal["projected", "completed", "failed"] = "completed"
    recorded_at: str = ""
    supersedes_sha256: str = ""

    def amount(self) -> float | None:
        """Use provider-confirmed spend when available, otherwise the projection."""
        return self.confirmed_usd if self.confirmed_usd is not None else self.estimated_usd

    def counts_as_spend(self) -> bool:
        """Whether the operation completed or reached a potentially billable provider dispatch."""
        return self.status == "completed" or (
            self.status == "failed"
            and (
                self.confirmed_usd is not None
                or self.requests > 0
                or self.input_units > 0
                or self.output_units > 0
            )
        )


@dataclass(frozen=True)
class BudgetDecision:
    """Authorization outcome before a job dispatch."""

    verdict: Literal["authorized", "review_required", "blocked"]
    reason: str
    projected_usd: float | None


class CostLedger:
    """Concurrent-safe append/read API for one store's cost records."""

    def __init__(self, path: Path):
        self.path = path

    def append(self, record: CostRecord) -> None:
        """Append one record under an exclusive lock."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(record)
        if not record.supersedes_sha256:
            payload.pop("supersedes_sha256")
        payload["recorded_at"] = record.recorded_at or datetime.now(timezone.utc).isoformat()
        with self.path.open("a+") as ledger_file:
            fcntl.flock(ledger_file.fileno(), fcntl.LOCK_EX)
            if record.supersedes_sha256:
                ledger_file.seek(0)
                prior = [json.loads(line) for line in ledger_file if line.strip()]
                matches = [
                    r
                    for r in prior
                    if hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()
                    == record.supersedes_sha256
                ]
                if len(matches) != 1 or any(
                    r.get("supersedes_sha256") == record.supersedes_sha256 for r in prior
                ):
                    raise ValueError("reconciliation must supersede one unreconciled receipt")
                if any(
                    matches[0][k] != payload[k] for k in ("owner", "campaign", "stage", "candidate")
                ):
                    raise ValueError("reconciliation cannot change receipt ownership")
                ledger_file.seek(0, os.SEEK_END)
            ledger_file.write(json.dumps(payload, sort_keys=True) + "\n")
            ledger_file.flush()
            os.fsync(ledger_file.fileno())

    def records(self) -> list[CostRecord]:
        """Read a consistent ledger snapshot under a shared lock."""
        if not self.path.is_file():
            return []
        with self.path.open() as ledger_file:
            fcntl.flock(ledger_file.fileno(), fcntl.LOCK_SH)
            rows = [json.loads(line) for line in ledger_file if line.strip()]
        superseded = {r["supersedes_sha256"] for r in rows if r.get("supersedes_sha256")}
        return [
            CostRecord(**r)
            for r in rows
            if hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest() not in superseded
        ]

    def reconcile(self, record: CostRecord, *, estimated_usd, confirmed_usd, requests) -> None:
        """Append an auditable replacement receipt; retain the original bytes and history."""
        if any(
            v is not None and (not math.isfinite(v) or v < 0)
            for v in (estimated_usd, confirmed_usd)
        ):
            raise ValueError("reconciliation prices must be finite and non-negative")
        if type(requests) is not int or requests < 0:
            raise ValueError("reconciliation request count must be a non-negative integer")
        with self.path.open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        matches = [r for r in rows if CostRecord(**r) == record]
        if len(matches) != 1:
            raise ValueError("reconciliation needs one existing receipt")
        payload = asdict(record)
        payload.update(
            estimated_usd=estimated_usd,
            confirmed_usd=confirmed_usd,
            requests=requests,
            recorded_at="",
            supersedes_sha256=hashlib.sha256(
                json.dumps(matches[0], sort_keys=True).encode()
            ).hexdigest(),
        )
        self.append(CostRecord(**payload))

    def book_and_release(
        self, records: list[CostRecord], reservation_id: str, requests_complete: bool
    ) -> bool:
        """Book durable receipts and release their envelope under one admission lock.

        Superseded receipt prices remain authoritative. Missing or unpriced liabilities
        retain the envelope. Return whether all request liabilities are settled.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".reservations.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            posted = {r.candidate for r in self.records()}
            for record in records:
                if record.candidate not in posted:
                    self.append(record)
                    posted.add(record.candidate)
            owners = {record.owner for record in records}
            settled = requests_complete and not any(
                r.owner in owners and r.counts_as_spend() and r.amount() is None
                for r in self.records()
            )
            if settled:
                _release_reservation(self, reservation_id)
            return settled


def release_reconciled_reservation(ledger: CostLedger, reservation_id: str) -> None:
    """Release an interrupted controller's reservation after all receipt liabilities are priced."""
    lock_path = ledger.path.with_suffix(".reservations.lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        _release_reservation(ledger, reservation_id, missing_ok=False)


def _release_reservation(ledger: CostLedger, reservation_id: str, missing_ok=True) -> None:
    """Append a reconciled release while the caller owns the reservations lock."""
    journal_path = ledger.path.with_suffix(".reservations.jsonl")
    latest = {}
    if not journal_path.exists():
        if missing_ok:
            return
        raise FileNotFoundError(journal_path)
    for line in journal_path.read_text().splitlines():
        row = json.loads(line)
        latest[row["id"]] = row
    if missing_ok and reservation_id not in latest:
        return
    row = latest[reservation_id]
    if any(
        r.owner == row["owner"] and r.counts_as_spend() and r.amount() is None
        for r in ledger.records()
    ):
        raise RuntimeError("unbounded costs prevent reservation reconciliation")
    if row["status"] != "reserved":
        return
    with journal_path.open("a") as journal:
        journal.write(
            json.dumps(
                {**row, "status": "released", "reason": "explicit_receipt_reconciliation"},
                sort_keys=True,
            )
            + "\n"
        )
        journal.flush()
        os.fsync(journal.fileno())


@contextmanager
def reserve_cost(
    ledger: CostLedger,
    *,
    policy: BudgetPolicy,
    owner: str,
    campaign: str,
    reservation_id: str,
    estimated_usd: float,
):
    """Reserve an opt-in controller's envelope atomically across cooperating processes.

    Reservations use two owner/store-level files, never per job. They are not spend receipts.
    A normal exit requires the caller to have written its measured ledger receipt; exceptions
    retain the reservation for explicit reconciliation, including a killed/preempted worker.
    Legacy dispatchers that do not use this protocol are not covered by its lock.
    """
    if not reservation_id or not math.isfinite(estimated_usd) or estimated_usd <= 0:
        raise ValueError("reservation requires an identity and finite positive envelope")
    ledger.path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = ledger.path.with_suffix(".reservations.lock")
    journal_path = ledger.path.with_suffix(".reservations.jsonl")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        latest = {}
        if journal_path.exists():
            for line in journal_path.read_text().splitlines():
                row = json.loads(line)
                latest[row["id"]] = row
        if latest.get(reservation_id, {}).get("status") == "reserved":
            raise RuntimeError("reservation remains active; reconcile before retrying")
        records = ledger.records()
        if any(r.owner == owner and r.counts_as_spend() and r.amount() is None for r in records):
            raise RuntimeError("unconfirmed spend must be reconciled before reservation")
        records.extend(
            CostRecord(
                owner=r["owner"],
                project="reservation",
                campaign=r["campaign"],
                stage="pending",
                candidate=r["id"],
                provider="pending",
                model="pending",
                estimated_usd=r["estimated_usd"],
                recorded_at=r["recorded_at"],
            )
            for r in latest.values()
            if r["status"] == "reserved"
        )
        decision = authorize_cost(
            estimated_usd, policy=policy, records=records, owner=owner, campaign=campaign
        )
        if decision.verdict != "authorized":
            raise RuntimeError(f"reservation {decision.verdict}: {decision.reason}")
        row = {
            "id": reservation_id,
            "owner": owner,
            "campaign": campaign,
            "estimated_usd": estimated_usd,
            "status": "reserved",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        with journal_path.open("a") as journal:
            journal.write(json.dumps(row, sort_keys=True) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
    yield
    # Deliberately not a finally: an interrupted/failed controller needs reconciliation.
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not any(
            record.owner == owner and record.counts_as_spend() and record.amount() is None
            for record in ledger.records()
        ):
            _release_reservation(ledger, reservation_id)


def authorize_cost(
    estimate_usd: float | None,
    *,
    policy: BudgetPolicy,
    records: list[CostRecord],
    owner: str,
    campaign: str,
    now: datetime | None = None,
) -> BudgetDecision:
    """Decide whether a projected job fits all frozen budget limits."""
    if estimate_usd is None:
        return BudgetDecision("review_required", "job price is unknown", None)
    if estimate_usd < 0:
        raise ValueError("estimated cost must be non-negative")
    now = now or datetime.now(timezone.utc)
    current_month = now.strftime("%Y-%m")
    current_day = now.date().isoformat()

    def spent(selected: list[CostRecord]) -> float:
        return sum(record.amount() or 0.0 for record in selected if record.counts_as_spend())

    owner_records = [record for record in records if record.owner == owner]
    month_spend = spent(
        [record for record in owner_records if record.recorded_at.startswith(current_month)]
    )
    day_spend = spent(
        [record for record in owner_records if record.recorded_at.startswith(current_day)]
    )
    campaign_spend = spent([record for record in records if record.campaign == campaign])
    spendable_month = policy.monthly_limit_usd - policy.reserve_usd
    if month_spend + estimate_usd > spendable_month:
        return BudgetDecision(
            "blocked",
            "job would consume the protected month-end reserve",
            estimate_usd,
        )
    if campaign_spend + estimate_usd > policy.campaign_limit_usd:
        return BudgetDecision("blocked", "campaign hard limit would be exceeded", estimate_usd)
    if day_spend + estimate_usd > policy.daily_limit_usd * (1 + policy.burst_fraction):
        return BudgetDecision(
            "blocked", "daily hard limit plus burst would be exceeded", estimate_usd
        )
    if estimate_usd >= policy.large_job_usd:
        return BudgetDecision(
            "review_required", "job crosses the large-job threshold", estimate_usd
        )
    if day_spend + estimate_usd > policy.daily_limit_usd:
        return BudgetDecision(
            "review_required", "job requires the daily burst allowance", estimate_usd
        )
    return BudgetDecision("authorized", "job fits the frozen budget", estimate_usd)
