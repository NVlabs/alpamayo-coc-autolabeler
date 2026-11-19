"""Request accounting with distinct provider bills and explicitly configured estimates."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import threading
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ....infra.artifacts import write_json_atomic
from ....infra.model import estimate_usage_cost


class AllocationGateError(RuntimeError):
    """Paid dispatch stopped at an accounting or allocation gate; preserve resume state."""


def attempt_budget_cost(row: dict) -> float | None:
    """Return billing or an explicitly authorized conservative request liability.

    Public registry token rates estimate valid usage without creating a confirmed bill.
    Approved estimates also cover transient HTTP errors ($1), read timeouts ($5), and opt-in
    successful native Gemini calls (twice the pinned no-cache token estimate). Estimates
    are not provider charges or guaranteed bounds. Other unpriced calls require reconciliation.
    """
    if row["cost_usd"] is not None:
        value = row["cost_usd"]
        return value if math.isfinite(value) and value >= 0 else None
    if row.get("reconciled_estimated_usd") is not None:
        value = row["reconciled_estimated_usd"]
        return value if math.isfinite(value) and value >= 0 else None
    if row.get("status_code") in {429, 500, 502, 503, 504} and not row.get("usage"):
        return 1.0
    if row.get("cost_estimate_policy") == "configured_token_rates_v1":
        if row.get("service_tier") not in (None, "default"):
            return None
        try:
            amount = estimate_usage_cost(row.get("usage"), row.get("token_pricing"))
        except ValueError:
            return None
        estimate = row.get("estimated_cost_usd")
        if (
            amount is not None
            and type(estimate) in (int, float)
            and math.isfinite(estimate)
            and math.isclose(amount, estimate, rel_tol=1e-12, abs_tol=1e-12)
        ):
            return amount
        return None
    if row.get("native_estimate_multiplier") == 2 and row.get("status_code") == 200:
        native = row.get("native_receipt", {})
        usage = native.get("usage", {})
        counts = [usage.get("input_tokens"), usage.get("output_tokens")]
        estimate = native.get("no_cache_estimated_cost_usd")
        if (
            row.get("status_code") == 200
            and native.get("cost_source") == "hub_ui_token_estimate"
            and row.get("native_estimate_policy") == "explicit_user_approved_double_no_cache_v1"
            and all(type(v) is int and v > 0 for v in counts)
            and type(estimate) in (int, float)
            and math.isfinite(estimate)
            and estimate >= 0
            and math.isclose(estimate, (counts[0] * 0.57 + counts[1] * 2.85) / 1e6)
        ):
            # Opt-in accounting estimate from the pinned native evaluator, not a bill or
            # guaranteed bound. The caller must have explicit authorization for this policy.
            return 2 * estimate
        return None
    approved_native_transport = (
        row.get("native_transport_reserve_policy") == "user_20260921_two_retained_native_failures"
        and row.get("error_type") == "transient_transport"
        and row.get("native_receipt", {}).get("error_class") == "transient_transport"
        and bool(row.get("request_id"))
        and row["request_id"] == row.get("native_receipt", {}).get("request_id")
    )
    if (
        (
            row.get("error_type") == "ReadTimeout"
            or approved_native_transport
            or (
                row.get("error_type") == "ReadError"
                and row.get("read_error_reserve_policy")
                == "user_20260921_readerror_existing_limits"
            )
            or (
                row.get("error_type") == "ConnectionError"
                and row.get("connection_error_reserve_policy")
                == "user_20260923_connectionerror_existing_limits"
            )
        )
        and row.get("status_code") is None
        and not row.get("usage")
        and not row.get("timeout_reservation_denied")
    ):
        return 5.0
    return None


def clip_generation_spend(row: dict) -> float | None:
    """Return ordinary clip spend, keeping approved error reserves in the separate pool.

    This is an admission measure, not a bill. Confirmed failed-call charges and successful
    call estimates count here. Approved unpriced failures still count in full against the
    stage/campaign allocation through attempt_budget_cost; they must not make the ordinary
    per-clip threshold prohibit an otherwise authorized transport retry. Unknown liabilities
    remain unknown and stop dispatch.
    """
    amount = attempt_budget_cost(row)
    if amount is None:
        return None
    if row["cost_usd"] is None and row.get("status_code") != 200:
        return 0.0
    return amount


def timeout_reservations_path(budget: dict) -> Path:
    """Locate one compact timeout allowance shared by all campaign phases and workers."""
    identity = hashlib.sha256(f"{budget['owner']}:{budget['campaign']}".encode()).hexdigest()
    return Path(budget["ledger"]).with_name(f"timeout_reservations_{identity}.json")


def reserve_read_timeout(path: Path, row: dict) -> bool:
    """Claim an unresolved $5 estimate within the recorded campaign approval under a lock.

    Replaying the same retained receipt is idempotent. An excess in-flight failure remains
    explicitly unreserved and stops further dispatch; it never silently enlarges the limit.
    """
    if row["cost_usd"] is not None or attempt_budget_cost(row) != 5:
        raise ValueError("timeout reservation requires an eligible unpriced read timeout")
    identity = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        state = (
            json.loads(path.read_text())
            if path.exists()
            else {
                "per_timeout_usd": 5,
                "limit_usd": 25,
                "reserved": [],
                "unreserved": [],
            }
        )
        capacity = _timeout_reserve_capacity(state)
        if row.get("error_type") == "ReadError" and state.get("read_error_authorization") != (
            "user_20260921_readerror_existing_limits"
        ):
            raise ValueError("read-error reserve requires campaign authorization")
        if (
            row.get("error_type") == "ConnectionError"
            and state.get("connection_error_authorization")
            != "user_20260923_connectionerror_existing_limits"
        ):
            raise ValueError("connection-error reserve requires campaign authorization")
        if identity in state["reserved"]:
            return True
        if identity in state["unreserved"]:
            return False
        allowed = len(state["reserved"]) < capacity and not state["unreserved"]
        state["reserved" if allowed else "unreserved"].append(identity)
        write_json_atomic(path, state)
        return allowed


def _timeout_reserve_capacity(state: dict) -> int:
    """Validate trusted local policy tokens; these values are not an access-control boundary."""
    limit = state["limit_usd"]
    extensions = {
        75: "user_20260921_75_usd_shared_reserve",
        200: "user_20260924_200_usd_shared_reserve",
        400: "user_20260928_400_usd_shared_reserve",
    }
    if state["per_timeout_usd"] != 5 or (
        limit != 25 and (limit not in extensions or state.get("authorization") != extensions[limit])
    ):
        raise ValueError("timeout reservation policy changed without approval")
    return limit // 5


def authorize_dispatch(budget: dict) -> None:
    """Check the controller allocation against fresh shared accounting before paid work."""
    from ....infra.budget import BudgetPolicy, CostLedger, authorize_cost

    if timeout_reservations_path(budget).with_suffix(".paused").exists():
        raise AllocationGateError("campaign dispatch is paused")
    envelope = budget["dispatch_envelope_usd"]
    if envelope is None or not math.isfinite(envelope) or envelope <= 0:
        raise ValueError("a finite positive dispatch allocation is required")
    records = CostLedger(Path(budget["ledger"])).records()
    if any(
        row.owner == budget["owner"] and row.counts_as_spend() and row.amount() is None
        for row in records
    ):
        raise RuntimeError("reconcile unknown ledger costs before paid dispatch")
    decision = authorize_cost(
        envelope,
        policy=BudgetPolicy(**budget["policy"]),
        records=records,
        owner=budget["owner"],
        campaign=budget["campaign"],
    )
    if decision.verdict != "authorized":
        raise RuntimeError(f"budget {decision.verdict}: {decision.reason}")
    if budget["accounting_verified_at"] is None:
        raise RuntimeError("refresh provider accounting before paid dispatch")
    verified = budget["accounting_verified_at"]
    if isinstance(verified, str):
        verified = datetime.fromisoformat(verified)
    if not isinstance(verified, datetime) or verified.utcoffset() is None:
        raise ValueError("accounting_verified_at requires an ISO timestamp with a timezone")
    if not 0 <= (datetime.now(timezone.utc) - verified).total_seconds() <= 3600:
        raise RuntimeError("provider accounting snapshot is stale")


@contextmanager
def generation_allocation(root: Path, budget: dict, request_admission=None, allocation_owned=False):
    """Own one lazy campaign reservation; workers reuse their controller's admission.

    Cached and deterministic work reserve nothing. Measured receipts settle once in the
    shared ledger; interrupted requests retain their liability until reconciliation.
    """
    if allocation_owned:
        if request_admission is None:
            raise ValueError("an owned allocation requires its controller admission")
        yield request_admission
        return
    allocation = _GenerationAllocation(root, budget, request_admission)
    try:
        if allocation.state_path.exists():
            allocation._recover()
        yield allocation
    except BaseException as error:
        try:
            allocation.finish()
        except Exception as settlement_error:
            error.add_note(
                "Allocation settlement also failed with "
                f"{type(settlement_error).__name__}; preserve state for reconciliation."
            )
        raise
    else:
        allocation.finish()


class _GenerationAllocation:
    """Persist standalone controller liability and settle its actual transport receipts."""

    def __init__(self, root: Path, budget: dict, request_admission):
        """Bind one output directory and optionally its concurrent request scheduler."""
        self.root = root.resolve()
        self.budget = budget
        self.request_admission = request_admission
        self.lock = threading.Lock()
        self.controller_lock = None
        self.reservation = None
        self.state_path = self.root / "campaign_allocation.json"
        self.state = None
        self.binding_verified = False

    def _receipts(self):
        """Merge registered recovery receipts before concurrently copied worker journals arrive."""
        roots = []
        current = self.root
        while current is not None:
            if current in roots:
                raise ValueError("generation recovery receipts contain a cycle")
            roots.append(current)
            serial = current / "generation_identity.json"
            parallel = current / "parallel_identity.json"
            if serial.exists():
                source = json.loads(serial.read_text()).get("recovery", {}).get("source")
            elif parallel.exists():
                source = json.loads(parallel.read_text()).get("recovery_source")
            else:
                source = None
            current = Path(source).resolve() if source is not None else None
        receipts = {}
        for source in reversed(roots):
            for path in sorted(source.rglob("usage.jsonl")):
                location = str(path.relative_to(source))
                for index, line in enumerate(path.read_text().splitlines()):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    key = (location, index)
                    if key in receipts and receipts[key][3] != row:
                        raise ValueError("registered recovery receipt content changed")
                    receipts.setdefault(key, (source, location, index, row))
        return list(receipts.values())

    def _book(self, receipts, requests_complete=False):
        """Book receipts once and atomically settle a fully priced controller allocation."""
        from ....infra.budget import CostLedger, CostRecord

        ledger = CostLedger(Path(self.budget["ledger"]))
        records = [
            CostRecord(
                owner=self.budget["owner"],
                project="alparsg",
                campaign=self.budget["campaign"],
                stage="generation",
                candidate=receipt_candidate(source, location, index, row),
                provider="configured_registry",
                model=row["model"],
                estimated_usd=attempt_budget_cost(row),
                confirmed_usd=row["cost_usd"],
                requests=1,
                status="completed" if row.get("status_code") == 200 else "failed",
                recorded_at=row.get("recorded_at", ""),
            )
            for source, location, index, row in receipts
        ]
        settled = ledger.book_and_release(records, self.state["reservation_id"], requests_complete)
        return ledger, settled

    def _recover(self):
        """Lock the controller and reconcile only fully recorded, priced prior requests."""
        from ....infra.budget import CostLedger

        handle = (self.root / "campaign_controller.lock").open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            handle.close()
            raise
        self.controller_lock = handle
        receipts = self._receipts()
        ledger = CostLedger(Path(self.budget["ledger"]))
        binding = {
            "root": str(self.root),
            "ledger": str(ledger.path.resolve()),
            "owner": self.budget["owner"],
            "campaign": self.budget["campaign"],
        }
        reservation_id = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
            if self.state["binding"] != binding or self.state["reservation_id"] != reservation_id:
                raise ValueError("campaign allocation identity changed")
            self.binding_verified = True
            ledger, _ = self._book(receipts, self.state["requests"] == len(receipts))
            if self.state["requests"] != len(receipts):
                raise AllocationGateError("reconcile outstanding requests without durable receipts")
            prices = {record.candidate: record.amount() for record in ledger.records()}
            if any(
                prices[receipt_candidate(source, location, index, row)] is None
                for source, location, index, row in receipts
            ):
                raise AllocationGateError("reconcile unknown generation receipt costs")
            self.state["in_flight"] = 0
        else:
            self.state = {
                "binding": binding,
                "reservation_id": reservation_id,
                "requests": len(receipts),
                "in_flight": 0,
            }
        self.binding_verified = True
        self._book(receipts)
        write_json_atomic(self.state_path, self.state)
        return ledger

    def before_attempt(self, before_dispatch=None):
        """Reserve shared headroom before the first provider request, then admit a worker."""
        from ....infra.budget import BudgetPolicy, CostLedger, reserve_cost

        try:
            with self.lock:
                if self.reservation is None:
                    if self.controller_lock is not None and not self.binding_verified:
                        raise RuntimeError(
                            "campaign allocation startup failed; reconcile before retry"
                        )
                    ledger = (
                        self._recover()
                        if self.controller_lock is None
                        else CostLedger(Path(self.budget["ledger"]))
                    )
                    authorize_dispatch(self.budget)
                    reservation = reserve_cost(
                        ledger,
                        policy=BudgetPolicy(**self.budget["policy"]),
                        owner=self.budget["owner"],
                        campaign=self.budget["campaign"],
                        reservation_id=self.state["reservation_id"],
                        estimated_usd=self.budget["dispatch_envelope_usd"],
                    )
                    reservation.__enter__()
                    self.reservation = reservation
        except (RuntimeError, ValueError, BlockingIOError) as error:
            raise AllocationGateError(str(error)) from error
        if self.request_admission is not None:
            self.request_admission.before_attempt(before_dispatch)
        elif before_dispatch is not None:
            before_dispatch()
        with self.lock:
            self.state["requests"] += 1
            self.state["in_flight"] += 1
            write_json_atomic(self.state_path, self.state)

    def record(self, event):
        """Reconcile a locally journaled receipt without double-debiting worker allocations."""
        if self.request_admission is not None:
            self.request_admission.record(event)
        with self.lock:
            if self.state is None or self.state["in_flight"] <= 0:
                raise ValueError("generation receipt has no controller admission")
            self.state["in_flight"] -= 1
            write_json_atomic(self.state_path, self.state)

    def finish(self):
        """Settle known receipts, retaining the reservation for any missing or unknown bill."""
        try:
            if self.controller_lock is None and self.state_path.exists():
                self._recover()
            if self.controller_lock is None or not self.binding_verified:
                return
            receipts = self._receipts()
            _, settled = self._book(receipts, self.state["requests"] == len(receipts))
            if self.reservation is not None:
                if settled:
                    self.reservation.__exit__(None, None, None)
                else:
                    error = RuntimeError("generation requests require accounting reconciliation")
                    self.reservation.__exit__(type(error), error, None)
        finally:
            if self.controller_lock is not None:
                self.controller_lock.close()


def receipt_candidate(source: Path, location: str, index: int, row: dict) -> str:
    """Identify one immutable transport receipt across explicitly registered recovery copies."""
    identity = (
        {"attempt_id": row["attempt_id"]}
        if row.get("attempt_id")
        else {"root": str(source), "journal": location, "line": index, "receipt": row}
    )
    return hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
        ).encode()
    ).hexdigest()


def resolved_attempts(path: Path, budget: dict) -> list[dict]:
    """Read usage with explicit ledger reconciliation prices, preserving original receipt bytes."""
    from ....infra.budget import CostLedger

    if not path.exists():
        return []
    root = next((p for p in path.parents if (p / "parallel_identity.json").exists()), path.parent)
    receipts = _GenerationAllocation(root, budget, None)._receipts()
    prices = {r.candidate: r for r in CostLedger(Path(budget["ledger"])).records()}
    resolved = {}
    for source, location, index, row in receipts:
        price = prices.get(receipt_candidate(source, location, index, row))
        if price is not None and (
            row["cost_usd"] is None
            and price.confirmed_usd is not None
            or attempt_budget_cost(row) is None
            and price.amount() is not None
        ):
            row = {
                **row,
                **(
                    {"cost_usd": price.confirmed_usd}
                    if price.confirmed_usd is not None
                    else {"reconciled_estimated_usd": price.estimated_usd}
                ),
            }
        resolved[location, index] = row
    location = str(path.relative_to(root))
    return [
        resolved[location, index]
        for index, line in enumerate(path.read_text().splitlines())
        if line.strip()
    ]


class RequestAdmission:
    """Controller-owned admission across concurrent workers in one reserved stage.

    Persist known spend and outstanding request allowances in one file. An unknown bill or
    interrupted in-flight request blocks resume until explicit accounting reconciliation.
    The request allowance is conservative admission headroom, not a provider price guarantee.
    """

    def __init__(
        self, path: Path, limit_usd: float, request_allowance_usd: float, initial_attempts=()
    ):
        """Bind a single controller's stage allocation, retaining earlier usage on resume."""
        if not 0 < request_allowance_usd <= limit_usd or not math.isfinite(limit_usd):
            raise ValueError("invalid concurrent request allocation")
        self.path = path
        self.lock = threading.Condition()
        self.state = {
            "limit_usd": limit_usd,
            "request_allowance_usd": request_allowance_usd,
            "known_cost_usd": 0.0,
            "unknown_cost_requests": 0,
            "unreserved_unknown_requests": 0,
            "reserved_unknown_cost_usd": 0.0,
            "in_flight": 0,
            "requests": 0,
        }
        if path.exists():
            prior = json.loads(path.read_text())
            if any(prior[k] != self.state[k] for k in ("limit_usd", "request_allowance_usd")):
                raise ValueError("request admission allocation changed")
            if prior["in_flight"] or prior["unreserved_unknown_requests"]:
                raise RuntimeError("reconcile outstanding or unknown requests before resume")
            self.state = prior
        elif initial_attempts:
            summary = summarize_attempts(list(initial_attempts), 0, 0)
            for key in (
                "known_cost_usd",
                "unknown_cost_requests",
                "unreserved_unknown_requests",
                "reserved_unknown_cost_usd",
                "requests",
            ):
                self.state[key] = summary[key]
            if self.state["unreserved_unknown_requests"]:
                raise RuntimeError("reconcile unbounded seed request costs before recovery")
        write_json_atomic(path, self.state)

    def before_attempt(self, before_dispatch: Callable[[], None] | None = None) -> None:
        """Wait for temporary headroom, but reject exhausted or unbounded allocations."""
        with self.lock:
            s = self.state
            while True:
                if before_dispatch is not None:
                    before_dispatch()
                if s["unreserved_unknown_requests"]:
                    raise AllocationGateError("reconcile unknown parallel request cost")
                next_request = (
                    s["known_cost_usd"]
                    + s["reserved_unknown_cost_usd"]
                    + s["request_allowance_usd"]
                )
                if next_request > s["limit_usd"]:
                    raise AllocationGateError("parallel request allocation exhausted")
                if next_request + s["in_flight"] * s["request_allowance_usd"] <= s["limit_usd"]:
                    break
                self.lock.wait()
            s["in_flight"] += 1
            write_json_atomic(self.path, s)

    def record(self, event: dict) -> None:
        """Reconcile one dispatched request without writing the shared campaign ledger."""
        with self.lock:
            if self.state["in_flight"] <= 0:
                raise ValueError("request receipt has no admission")
            self.state["in_flight"] -= 1
            self.state["requests"] += 1
            amount = event["cost_usd"]
            if amount is None or not math.isfinite(amount) or amount < 0:
                self.state["unknown_cost_requests"] += 1
                reserved = attempt_budget_cost(event)
                if reserved is None:
                    self.state["unreserved_unknown_requests"] += 1
                else:
                    self.state["reserved_unknown_cost_usd"] += reserved
            else:
                self.state["known_cost_usd"] += amount
            write_json_atomic(self.path, self.state)
            self.lock.notify_all()


class RequestAccounting:
    """Run-owned request cap and append-only per-attempt usage log.

    A count cap is not a dollar guarantee. The generation controller separately
    authorizes and reserves a measured conservative dollar envelope before dispatch.
    """

    def __init__(  # noqa: PLR0917 -- explicit run, clip, and shared accounting context
        self,
        path: Path,
        clip_id: str,
        stage: str,
        max_requests: int,
        admission: RequestAdmission | None = None,
        timeout_reservations: Path | None = None,
        max_cost_usd: float | None = None,
    ):
        """Bind the run's shared append-only log and an explicit positive attempt cap."""
        if max_requests <= 0:
            raise ValueError("accounting requires a positive cap")
        if max_cost_usd is not None and (not math.isfinite(max_cost_usd) or max_cost_usd <= 0):
            raise ValueError("per-clip dollar allocation must be positive and finite")
        self.path = path
        self.clip_id = clip_id
        self.stage = stage
        self.max_requests = max_requests
        self.reserved = 0
        self.lock = threading.Lock()
        self.attempts: list[dict] = []
        self.admission = admission
        self.timeout_reservations = timeout_reservations
        self.max_cost_usd = max_cost_usd
        self.inherited_cost_usd = 0.0

    def before_attempt(self) -> None:
        """Reserve one request atomically, including failed HTTP attempts."""
        with self.lock:

            def check_limits():
                """Recheck campaign and clip limits whenever queued admission wakes."""
                if (
                    self.timeout_reservations is not None
                    and self.timeout_reservations.with_suffix(".paused").exists()
                ):
                    raise AllocationGateError("campaign dispatch is paused")
                if self.timeout_reservations is not None and self.timeout_reservations.exists():
                    state = json.loads(self.timeout_reservations.read_text())
                    if (
                        len(state["reserved"]) >= _timeout_reserve_capacity(state)
                        or state["unreserved"]
                    ):
                        raise AllocationGateError("campaign timeout reservation limit exhausted")
                if self.reserved >= self.max_requests:
                    raise RuntimeError("request cap reached")
                if self.max_cost_usd is not None:
                    amounts = [clip_generation_spend(row) for row in self.attempts]
                    if any(value is None for value in amounts):
                        raise RuntimeError("reconcile unknown per-clip cost")
                    if self.inherited_cost_usd + sum(amounts) >= self.max_cost_usd:
                        raise RuntimeError("per-clip generation allocation exhausted")

            if self.admission is not None:
                self.admission.before_attempt(check_limits)
            else:
                check_limits()
            self.reserved += 1

    def record(self, event: dict) -> None:
        """Append transport-sanitized usage without prompts or credential metadata."""
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            row = {"clip_id": self.clip_id, "stage": self.stage, **event}
            if row.get("error_type") == "ReadError" and self.timeout_reservations is not None:
                state = (
                    json.loads(self.timeout_reservations.read_text())
                    if self.timeout_reservations.exists()
                    else {}
                )
                if (
                    state.get("read_error_authorization")
                    == "user_20260921_readerror_existing_limits"
                ):
                    row["read_error_reserve_policy"] = state["read_error_authorization"]
            if row.get("error_type") == "ConnectionError" and self.timeout_reservations is not None:
                state = (
                    json.loads(self.timeout_reservations.read_text())
                    if self.timeout_reservations.exists()
                    else {}
                )
                if (
                    state.get("connection_error_authorization")
                    == "user_20260923_connectionerror_existing_limits"
                ):
                    row["connection_error_reserve_policy"] = state["connection_error_authorization"]
            if (
                row["cost_usd"] is None
                and row.get("error_type")
                in {"ReadTimeout", "ReadError", "ConnectionError", "transient_transport"}
                and attempt_budget_cost(row) == 5
            ):
                row["timeout_reservation_id"] = uuid.uuid4().hex
                if self.timeout_reservations is None or not reserve_read_timeout(
                    self.timeout_reservations, row
                ):
                    row["timeout_reservation_denied"] = True
            with self.path.open("a") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            self.attempts.append(row)
            if self.admission is not None:
                self.admission.record(row)


def summarize_attempts(attempts: list[dict], attempted_clips: int, valid_clips: int) -> dict:
    """Report complete cost only if every attempt has provider-confirmed billing.

    Unknown failed requests remain unknown, not zero. Separate evaluation logs can
    be summarized independently and then included in an explicitly labeled campaign bill.
    """
    if not 0 <= valid_clips <= attempted_clips:
        raise ValueError("invalid clip denominators")
    amounts = [row["cost_usd"] for row in attempts if row["cost_usd"] is not None]
    if any(not math.isfinite(value) or value < 0 for value in amounts):
        raise ValueError("invalid provider cost")
    known = sum(amounts)
    unknown = len(attempts) - len(amounts)
    total = known if not unknown else None
    budget_amounts = [attempt_budget_cost(row) for row in attempts]
    unreserved = sum(amount is None for amount in budget_amounts)
    reserved = sum(
        amount
        for row, amount in zip(attempts, budget_amounts, strict=True)
        if row["cost_usd"] is None and amount is not None
    )
    token_totals = {}
    for row in attempts:
        usage = row["usage"]
        if not isinstance(usage, dict):
            continue
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if type(value) is int and value >= 0:
                token_totals[key] = token_totals.get(key, 0) + value
        for group, names in (
            ("prompt_tokens_details", ("cached_tokens",)),
            ("completion_tokens_details", ("reasoning_tokens",)),
        ):
            for key in names:
                details = usage.get(group)
                value = details.get(key) if isinstance(details, dict) else None
                if type(value) is int and value >= 0:
                    token_totals[key] = token_totals.get(key, 0) + value
    result = {
        "attempted_clips": attempted_clips,
        "valid_clips": valid_clips,
        "requests": len(attempts),
        "unknown_cost_requests": unknown,
        "unreserved_unknown_requests": unreserved,
        "reserved_unknown_cost_usd": reserved,
        "budget_cost_usd": known + reserved if not unreserved else None,
        "unknown_usage_requests": sum(row["usage"] is None for row in attempts),
        "known_cost_usd": known,
        "total_cost_usd": total,
        "usd_per_attempted_clip": total / attempted_clips
        if total is not None and attempted_clips
        else None,
        "usd_per_valid_clip": total / valid_clips if total is not None and valid_clips else None,
        "request_elapsed_s": sum(row["elapsed_s"] for row in attempts),
        "tokens": token_totals,
    }
    estimated = [
        amount
        for row, amount in zip(attempts, budget_amounts, strict=True)
        if row["cost_usd"] is None
        and row.get("status_code") == 200
        and row.get("cost_estimate_policy") == "configured_token_rates_v1"
        and amount is not None
    ]
    if estimated:
        result.update(estimated_cost_usd=sum(estimated), estimated_cost_requests=len(estimated))
    return result
