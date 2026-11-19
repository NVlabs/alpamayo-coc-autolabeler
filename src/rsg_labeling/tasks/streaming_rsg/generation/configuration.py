"""Resolved task policy and immutable generation cache identities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SEARCH_KNOBS = frozenset(
    {
        "policy.interval_prompt",
        "policy.coc_prompt",
        "policy.max_refinements",
        "max_output_tokens",
        "temperature",
        "policy.include_neighbors",
        "policy.interpreter_max_frames",
        "policy.coverage_review",
        "policy.interval_construction",
    }
)


class GenerationSetupError(ValueError):
    """Unfilled local configuration reported as CLI usage without masking runtime errors."""


def load_config(path: Path, overrides=()) -> dict:
    """Apply only declared experimental knobs, never evaluation/data/accounting overrides."""
    import yaml

    config = yaml.safe_load(Path(path).read_text())
    for override in overrides or ():
        key, separator, value = override.partition("=")
        if not separator or key not in SEARCH_KNOBS:
            raise ValueError(
                f"override is outside frozen experiment surface: {key}; "
                f"allowed keys: {', '.join(sorted(SEARCH_KNOBS))}"
            )
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = yaml.safe_load(value)
    if "policy" in config and config["policy"].get("interval_construction", "joint_v1") not in (
        "joint_v1",
        "skeleton_v1",
    ):
        raise ValueError("interval_construction must be joint_v1 or skeleton_v1")
    if (
        type(config.get("max_resume_attempts", 1)) is not int
        or config.get("max_resume_attempts", 1) < 0
    ):
        raise ValueError("max_resume_attempts must be a non-negative integer")
    return config


def validate_setup(config: dict, preparation: bool = False) -> None:
    """Name unfilled run settings before paths, output directories or accounting are used.

    Preparation needs a compiler and registry, but no generated input manifest or paid
    authorization. Saved-graph modes need a source publication; only native compilation
    and boundary replay need a compiler configuration. Existing budget checks retain
    ownership of numerical limits, freshness and admission.
    """
    paths = ["endpoint_registry"]
    if preparation or config.get("mode") in {"native", "boundary_replay"}:
        paths.append("compiler_config")
    if not preparation:
        paths.append("input_manifest")
        if config.get("mode") != "native":
            paths.append("source_run")
    missing = []
    for key in paths:
        value = config.get(key)
        if (
            not isinstance(value, (str, Path))
            or not str(value).strip()
            or str(value).startswith(("/absolute/path/to/", "/path/to/"))
        ):
            missing.append(key)
    if not preparation:
        if config.get("allow_paid_dispatch") is not True:
            missing.append("allow_paid_dispatch (enable only for an owner-approved budget)")
        budget = config.get("budget") or {}
        for key in (
            "owner",
            "campaign",
            "ledger",
            "accounting_verified_at",
            "dispatch_envelope_usd",
            "policy",
        ):
            if budget.get(key) is None or budget.get(key) == "":
                missing.append(f"budget.{key}")
        if str(budget.get("ledger", "")).startswith(("/absolute/path/to/", "/path/to/")):
            missing.append("budget.ledger")
    if missing:
        raise GenerationSetupError("unfilled generation settings: " + ", ".join(missing))


def digest(value) -> str:
    """Hash a canonical structured artifact, excluding no undeclared fields."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def resolved_model(registry: str) -> str:
    """Require a homogeneous registry; do not persist endpoint URLs or credentials."""
    rows = [json.loads(p.read_text()) for p in Path(registry).glob("r*.json")]
    models = {row["model"] for row in rows if row.get("status") == "ready"}
    if len(models) != 1:
        raise ValueError("one resolved ready model is required")
    return next(iter(models))


def endpoint_settings(registry: str) -> dict:
    """Bind nonsecret request semantics that can override the run's configured temperature."""
    rows = [json.loads(p.read_text()) for p in Path(registry).glob("r*.json")]
    settings = [
        {
            "model": r["model"],
            "temperature_override": r.get("temperature"),
            "hosted_chat": bool(r.get("api_key_env")),
            **(
                {
                    "request_profile": r["request_profile"],
                    "reasoning_effort": r.get("reasoning_effort"),
                    "service_tier": "default",
                }
                if r.get("request_profile") == "openai_reasoning"
                else {}
            ),
        }
        for r in rows
        if r.get("status") == "ready"
    ]
    if not settings or any(row != settings[0] for row in settings):
        raise ValueError("ready endpoints disagree on request semantics")
    return settings[0]


def generation_code_paths(parent: Path) -> list[Path]:
    """List generator source files for both resume and optimization-stage identity."""
    return [
        *parent.joinpath("streaming_rsg").rglob("*.py"),
        *parent.joinpath("common").rglob("*.py"),
        *(
            path
            for path in parent.joinpath("tasks/streaming_rsg").rglob("*.py")
            if not {"tests", "scripts", "docs"}.intersection(path.parts)
        ),
        *parent.joinpath("infra").glob("*.py"),
        *parent.joinpath("infra").glob("*.yaml"),
        parent / "infra/middleware/__init__.py",
        parent / "infra/middleware/parsing.py",
    ]


def generation_identity(config: dict, manifest: dict, sources: dict) -> dict:
    """Bind resume to input content, resolved model/config, prompts, source and task code.

    Accounting freshness/allocation and scratch placement are operational, not cache keys.
    A changed generator, input population, upstream record or model requires a new run.
    """
    policy = {
        key: value
        for key, value in config.items()
        if key
        not in {
            "budget",
            "scratch_dir",
            "endpoint_registry",
            "reference_manifest",
            "graph_gold",
            "judge",
            "allow_paid_dispatch",
            "task_interface",
            "agentic",
            "max_resume_attempts",
        }
    }
    parent = Path(__file__).resolve().parents[3]
    paths = generation_code_paths(parent)
    code = {
        str(p.relative_to(parent)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
        if "tests" not in p.parts
    }
    compiler = None
    if config["mode"] in {"native", "boundary_replay"}:
        from ....streaming_rsg.runtime.config import load_config as compiler_config

        compiler = compiler_config(Path(config["compiler_config"])).model_dump(mode="json")
        sensor_path = compiler["playground"]["env"].get("sensor_manifest")
        if sensor_path is not None:
            compiler["sensor_manifest_sha256"] = digest(json.loads(Path(sensor_path).read_text()))
        navigation_path = compiler["playground"]["env"]["navigation_manifest"]
        if navigation_path is not None:
            compiler["navigation_manifest_sha256"] = digest(
                json.loads(Path(navigation_path).read_text())
            )
        action_path = compiler["playground"]["env"]["meta_action_manifest"]
        if action_path is not None:
            compiler["meta_action_manifest_sha256"] = digest(
                json.loads(Path(action_path).read_text())
            )
        video_path = compiler["playground"]["env"]["video_manifest"]
        if video_path is not None:
            compiler["video_manifest_sha256"] = digest(json.loads(Path(video_path).read_text()))
    payload = {
        "policy": policy,
        "model": resolved_model(config["endpoint_registry"]),
        "endpoint_settings": endpoint_settings(config["endpoint_registry"]),
        "inputs": digest(manifest),
        "sources": digest(sources),
        "compiler": compiler,
        "code": digest(code),
    }
    return {"sha256": digest(payload), "components": payload}
