"""Public OpenAI transport must preserve honest usage estimates and dispatch gates."""

import json
from datetime import datetime, timezone

import pytest
import requests

from ....infra.budget import CostLedger
from ....infra.model import ChatClient
from ..generation.cost import (
    AllocationGateError,
    RequestAccounting,
    RequestAdmission,
    attempt_budget_cost,
    generation_allocation,
    summarize_attempts,
)


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """Bind fake credentials and explicit test prices without touching shared key state."""
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-public-profile-key")
    monkeypatch.setenv("ALPAMAYO_INFERENCE_STATE_DIR", str(tmp_path / "key-state"))
    path = tmp_path / "registry"
    path.mkdir()
    record = {
        "status": "ready",
        "model": "gpt-6.1-sol",
        "url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "request_profile": "openai_reasoning",
        "reasoning_effort": "medium",
        "pricing": {"input_usd_per_million_tokens": 2.0, "output_usd_per_million_tokens": 8.0},
    }
    (path / "r0.json").write_text(json.dumps(record))
    return path, record


def response(usage, cost=None):
    """Return synthetic provider usage; the fixture rates are not published model prices."""
    value = requests.Response()
    value.status_code = 200
    value._content = json.dumps(
        {"choices": [{"message": {"content": "{}"}}], "usage": usage, "service_tier": "default"}
    ).encode()
    if cost is not None:
        value.headers["x-litellm-response-cost"] = cost
    return value


def test_public_profile_prices_usage_without_claiming_confirmed_billing(
    registry, tmp_path, monkeypatch
):
    """Two real accounted calls settle estimated liability and retain schema/sampling rules."""
    path, record = registry
    sent = []

    def post(url, **kwargs):
        """Capture the full public wire contract at the only external boundary."""
        assert url == "https://api.openai.com/v1/chat/completions"
        assert kwargs["headers"] == {"Authorization": "Bearer synthetic-public-profile-key"}
        sent.append(kwargs["json"])
        return response(
            {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_tokens_details": {"cached_tokens": 60},
                "completion_tokens_details": {"reasoning_tokens": 20},
            }
        )

    monkeypatch.setattr(requests, "post", post)
    out = tmp_path / "out"
    out.mkdir()
    budget = {
        "owner": "test",
        "campaign": "public-profile",
        "ledger": str(tmp_path / "ledger"),
        "accounting_verified_at": datetime.now(timezone.utc).isoformat(),
        "dispatch_envelope_usd": 10,
        "policy": {
            "monthly_limit_usd": 100,
            "reserve_usd": 0,
            "campaign_limit_usd": 50,
            "daily_limit_usd": 50,
            "large_job_usd": 50,
        },
    }
    with generation_allocation(out, budget) as admission:
        accounting = RequestAccounting(
            out / "usage.jsonl", "clip", "generation", 4, admission=admission, max_cost_usd=1
        )
        client = ChatClient(
            path,
            max_tokens=2048,
            on_attempt=accounting.record,
            before_attempt=accounting.before_attempt,
        )
        assert client.complete([{"role": "user", "content": "frame context"}]) == "{}"
        schema = {"type": "object", "properties": {}, "additionalProperties": False}
        assert client.complete([], response_schema=schema, temperature=0) == "{}"
    for request in sent:
        assert request["max_completion_tokens"] == 2048
        assert request["reasoning_effort"] == "medium"
        assert request["service_tier"] == "default"
        assert not {"max_tokens", "temperature", "top_p", "chat_template_kwargs"} & request.keys()
    assert sent[1]["response_format"]["json_schema"]["schema"] == schema
    rows = [json.loads(line) for line in (out / "usage.jsonl").read_text().splitlines()]
    assert all(row["cost_usd"] is None for row in rows)
    assert [attempt_budget_cost(row) for row in rows] == pytest.approx([0.0006, 0.0006])
    ledger = CostLedger(tmp_path / "ledger")
    assert [row.confirmed_usd for row in ledger.records()] == [None, None]
    assert [row.estimated_usd for row in ledger.records()] == pytest.approx([0.0006, 0.0006])
    totals = summarize_attempts(rows, attempted_clips=1, valid_clips=1)
    assert totals["total_cost_usd"] is None
    assert totals["estimated_cost_usd"] == pytest.approx(0.0012)
    assert totals["budget_cost_usd"] == pytest.approx(0.0012)
    assert "synthetic-public-profile-key" not in json.dumps(rows)


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"prompt_tokens": True, "completion_tokens": 1},
        {"prompt_tokens": 10, "completion_tokens": -1},
        {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 1},
    ],
)
def test_invalid_public_usage_keeps_next_dispatch_blocked(registry, tmp_path, monkeypatch, usage):
    """Missing/malformed accounting cannot be replaced by a zero-cost success."""
    path, _ = registry
    calls = []

    def post(*args, **kwargs):
        """Return an unpriced response exactly once."""
        calls.append(1)
        return response(usage)

    monkeypatch.setattr(requests, "post", post)
    admission = RequestAdmission(tmp_path / "admission.json", 5, 1)
    client = ChatClient(path, on_attempt=admission.record, before_attempt=admission.before_attempt)
    client.complete([])
    with pytest.raises(AllocationGateError, match="unknown"):
        client.complete([])
    assert len(calls) == 1


@pytest.mark.parametrize("price", [None, 0, -1, True, float("nan"), float("inf")])
def test_invalid_rates_reject_before_admission_or_http(registry, monkeypatch, price):
    """A public operator must configure valid prices before any paid attempt."""
    path, record = registry
    record["pricing"]["input_usd_per_million_tokens"] = price
    (path / "r0.json").write_text(json.dumps(record))
    calls = []
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: calls.append("http"))
    client = ChatClient(path, before_attempt=lambda: calls.append("admission"))
    with pytest.raises(ValueError, match="pricing"):
        client.complete([])
    assert not calls


def test_provider_charge_overrides_estimate_and_unexpected_tier_remains_unknown(
    registry, monkeypatch
):
    """A real dollar receipt wins; standard-rate estimates cannot price another service tier."""
    path, _ = registry
    usage = {"prompt_tokens": 100, "completion_tokens": 50}
    billed = response(usage, cost="0.004")
    priority = response(usage)
    body = priority.json()
    body["service_tier"] = "priority"
    priority._content = json.dumps(body).encode()
    replies = [billed, priority]
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: replies.pop(0))
    attempts = []
    client = ChatClient(path, on_attempt=attempts.append)
    client.complete([])
    client.complete([])
    assert attempts[0]["cost_usd"] == 0.004
    assert attempt_budget_cost(attempts[0]) == 0.004
    assert attempts[1]["cost_usd"] is None and attempt_budget_cost(attempts[1]) is None


def test_profile_and_reasoning_effort_bind_request_identity(registry):
    """Resume must reject a different request profile while retaining captured historical rates."""
    from ..generation.configuration import endpoint_settings

    path, record = registry
    before = endpoint_settings(str(path))
    record["reasoning_effort"] = "low"
    (path / "r0.json").write_text(json.dumps(record))
    after = endpoint_settings(str(path))
    assert before != after
    record["pricing"]["input_usd_per_million_tokens"] = 3
    (path / "r0.json").write_text(json.dumps(record))
    assert endpoint_settings(str(path)) == after


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_profile_transient_response_preserves_reserved_retry(
    registry, tmp_path, monkeypatch, status
):
    """A usage-free transient error reserves one dollar and permits a bounded retry."""
    from ....infra import model as model_module

    path, _ = registry
    replies = [response(None), response({"prompt_tokens": 10, "completion_tokens": 10})]
    replies[0].status_code = status
    monkeypatch.setattr(model_module.time, "sleep", lambda delay: None)
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: replies.pop(0))
    admission = RequestAdmission(tmp_path / "admission.json", 10, 1)
    accounting = RequestAccounting(
        tmp_path / "usage.jsonl", "clip", "generation", 3, admission=admission
    )
    client = ChatClient(
        path,
        transient_retries=1,
        before_attempt=accounting.before_attempt,
        on_attempt=accounting.record,
    )
    assert client.complete([]) == "{}"
    assert len(accounting.attempts) == 2
    failed, completed = accounting.attempts
    assert "cost_estimate_policy" not in failed and failed["cost_usd"] is None
    assert attempt_budget_cost(failed) == 1
    assert completed["cost_usd"] is None and attempt_budget_cost(completed) == pytest.approx(0.0001)
    failed["cost_estimate_policy"] = "configured_token_rates_v1"
    assert attempt_budget_cost(failed) == 1, (
        "Retained receipts from the previous profile must also resume"
    )
    totals = summarize_attempts(accounting.attempts, 1, 1)
    assert totals["budget_cost_usd"] == pytest.approx(1.0001)
    assert totals["estimated_cost_usd"] == pytest.approx(0.0001)
    assert totals["total_cost_usd"] is None


@pytest.mark.parametrize(
    "usage",
    [
        None,
        ["bad"],
        "bad",
        {"prompt_tokens": None, "completion_tokens": 10},
        {"prompt_tokens": "10", "completion_tokens": 10},
        {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20.5},
        {"prompt_tokens": 10, "completion_tokens": -1},
        {
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": "bad"},
        },
        {"prompt_tokens": 10, "completion_tokens": 10, "completion_tokens_details": ["bad"]},
    ],
)
def test_summary_keeps_valid_token_counts_without_trusting_malformed_usage(usage):
    """Retained malformed receipts must remain summarizable after explicit price reconciliation."""
    row = {"cost_usd": 0.25, "usage": usage, "elapsed_s": 1, "status_code": 200}
    summary = summarize_attempts([row], 1, 1)
    assert summary["total_cost_usd"] == 0.25
    assert all(type(value) is int and value >= 0 for value in summary["tokens"].values())
    assert "cached_tokens" not in summary["tokens"]


@pytest.mark.parametrize("hosted", [False, True])
@pytest.mark.parametrize("locked", [False, True])
@pytest.mark.parametrize("temperature", [None, 0.2])
@pytest.mark.parametrize("refit", [False, True])
def test_unprofiled_transport_preserves_request_shapes(
    registry, monkeypatch, hosted, locked, temperature, refit
):
    """Pin legacy local/hosted, ordinary/schema and context-refit wire contracts in the release."""
    from copy import deepcopy

    path, record = registry
    for field in ["request_profile", "reasoning_effort", "pricing"]:
        record.pop(field)
    record.update(url="https://legacy.example.invalid", model="legacy-model", temperature=1.0)
    if not hosted:
        record.pop("api_key_env")
    (path / "r0.json").write_text(json.dumps(record))
    calls = []

    def post(url, **kwargs):
        """Capture actual dispatches without sharing mutable payload objects."""
        calls.append((url, deepcopy(kwargs)))
        value = response({"prompt_tokens": 10, "completion_tokens": 10}, cost="0.001")
        if refit and len(calls) == 1:
            value.status_code = 400
            value._content = (
                b"maximum context length is 16000 tokens; "
                b"prompt contains at least 10000 input tokens"
            )
        return value

    monkeypatch.setattr(requests, "post", post)
    schema = {"type": "object", "properties": {}, "additionalProperties": False} if locked else None
    assert ChatClient(path).complete([], temperature=temperature, response_schema=schema) == "{}"
    expected = {"model": "legacy-model", "messages": []}
    if locked:
        expected.update(
            max_completion_tokens=8192,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "reply", "schema": schema, "strict": True},
            },
        )
        if temperature is not None:
            expected["temperature"] = temperature
    else:
        expected.update(max_tokens=8192, temperature=1.0)
    if not hosted:
        expected["chat_template_kwargs"] = {"enable_thinking": False}
    assert calls[0][0] == "https://legacy.example.invalid/v1/chat/completions"
    assert calls[0][1]["json"] == expected
    assert calls[0][1]["headers"] == (
        {"Authorization": "Bearer synthetic-public-profile-key"} if hosted else {}
    )
    assert len(calls) == (2 if refit else 1)
    if refit:
        expected["max_completion_tokens" if locked else "max_tokens"] = 5936
        assert calls[1][1]["json"] == expected


def test_unqualified_response_tier_cannot_receive_a_token_estimate(registry, tmp_path, monkeypatch):
    """Check the actual receipt, not just a manually constructed budget row."""
    path, _ = registry
    value = response({"prompt_tokens": 10, "completion_tokens": 10})
    body = value.json()
    body["service_tier"] = "priority"
    value._content = json.dumps(body).encode()
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: value)
    admission = RequestAdmission(tmp_path / "admission.json", 10, 1)
    accounting = RequestAccounting(
        tmp_path / "usage.jsonl", "clip", "generation", 3, admission=admission
    )
    client = ChatClient(
        path, on_attempt=accounting.record, before_attempt=accounting.before_attempt
    )
    assert client.complete([]) == "{}"
    assert accounting.attempts[0]["estimated_cost_usd"] is None
    assert attempt_budget_cost(accounting.attempts[0]) is None
    with pytest.raises(AllocationGateError):
        client.complete([])


def test_confirmed_ledger_price_replaces_estimate_without_rewriting_usage(
    registry, tmp_path, monkeypatch
):
    """A later confirmed charge becomes the summary price and keeps the original receipt."""
    from ..generation.cost import resolved_attempts

    path, _ = registry
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: response({"prompt_tokens": 10, "completion_tokens": 10}),
    )
    budget = {
        "owner": "test",
        "campaign": "test",
        "ledger": str(tmp_path / "ledger"),
        "accounting_verified_at": datetime.now(timezone.utc).isoformat(),
        "dispatch_envelope_usd": 10,
        "policy": {
            "monthly_limit_usd": 100,
            "reserve_usd": 0,
            "campaign_limit_usd": 50,
            "daily_limit_usd": 50,
            "large_job_usd": 50,
        },
    }
    out = tmp_path / "run"
    out.mkdir()
    with generation_allocation(out, budget) as admission:
        accounting = RequestAccounting(
            out / "usage.jsonl", "clip", "generation", 3, admission=admission
        )
        assert (
            ChatClient(
                path, on_attempt=accounting.record, before_attempt=accounting.before_attempt
            ).complete([])
            == "{}"
        )
    original = (out / "usage.jsonl").read_bytes()
    ledger = CostLedger(tmp_path / "ledger")
    receipt = ledger.records()[0]
    assert receipt.confirmed_usd is None and receipt.estimated_usd == pytest.approx(0.0001)
    ledger.reconcile(receipt, estimated_usd=None, confirmed_usd=0.25, requests=1)
    rows = resolved_attempts(out / "usage.jsonl", budget)
    assert rows[0]["cost_usd"] == 0.25
    assert summarize_attempts(rows, 1, 1)["total_cost_usd"] == 0.25
    assert (out / "usage.jsonl").read_bytes() == original
