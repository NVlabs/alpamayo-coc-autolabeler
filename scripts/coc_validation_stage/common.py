#!/usr/bin/env python3
"""Shared runtime helpers for CoC video validation stages."""

from __future__ import annotations

import argparse
import base64
import bisect
import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    import yaml
except ImportError:  # pragma: no cover - validated in tests/runtime
    yaml = None

DEFAULT_API_BASE = "auto"
DEFAULT_MODEL_NAME = "gpt5.5"
DEFAULT_CAMERA_NAME = "camera_front_wide_120fov"
DEFAULT_RELATIVE_TIMESTAMP_MAX_SEC = 600.0
COT_YAML_GLOB = "cot_*.yaml"
CAMERA_NAME_RE = re.compile(r"(camera_front_[a-z0-9_]+)")

SUPPORTED_MODEL_NAMES = (
    "qwen3_vl_235b_awq",
    "qwen3.5_35b",
    "qwen3.5_397b_fp8",
    "gpt5",
    "gpt5.5",
)
QWEN_HF_MODEL_MAP = {
    "qwen3_vl_235b_awq": "QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ",
    "qwen3.5_35b": "Qwen/Qwen3.5-35B-A3B",
    "qwen3.5_397b_fp8": "Qwen/Qwen3.5-397B-A17B-FP8",
}
NV_INFERENCE_MODEL_MAP = {
    "gpt5": "us/azure/openai/gpt-5",
    "gpt-5": "us/azure/openai/gpt-5",
    "gpt5.5": "openai/openai/gpt-5.5",
    "gpt-5.5": "openai/openai/gpt-5.5",
}
NV_INFERENCE_URL = "https://inference-api.nvidia.com"
NV_AZURE_API_VERSION = "2025-01-01-preview"

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_THINKING_CLOSE_TAG = "</think>"

LABEL_COLUMN_CANDIDATES = (
    "effect_on_ego_behavior",
    "final_content_ego_behavior_schema_effect_on_ego_behavior",
    "vlm",
    "labeled effect_on_ego_behavior",
)
CLIP_ID_COLUMN_CANDIDATES = ("clip_id", "clip_id_gt", "clip_id_pred")
TIMESTAMP_COLUMN_CANDIDATES = (
    "event_start_timestamp",
    "start_timestamp",
    "timestamp",
    "keyframe_range_start_timestamp",
)
CLASS_COLUMN_CANDIDATES = ("label_class_identifier", "raw_final_content_label_class_identifier")
VIDEO_PATH_COLUMN_CANDIDATES = (
    "video_path",
    "fpv_video_path",
    "front_wide_video_path",
    "camera_front_wide_120fov_path",
)
# Internal record marker set by the autolabeler loader when ``video_path`` points at a
# pre-saved 8-second event segment (as opposed to a legacy full raw camera video).
SAVED_SEGMENT_PROVENANCE_KEY = "_video_is_saved_segment"


@dataclass
class NormalizedRow:
    """Normalized CoC record consumed by the validation stages."""

    row_index: int
    clip_id: str
    event_timestamp: int | None
    coc_label: str
    label_class_identifier: str
    yaml_path: str
    video_path_hint: str
    camera_name: str = ""
    hint_is_saved_segment: bool = False


@dataclass(frozen=True)
class ModelRuntime:
    """Resolved evaluator client, model name, and backend settings."""

    client: Any
    model_name: str
    backend: str
    omit_temperature: bool
    qwen_disable_thinking: bool
    default_temperature: float | None
    uses_max_completion_tokens: bool = False


@dataclass(frozen=True)
class StageSpec:
    """Static configuration for a single validation stage."""

    stage: int
    name: str
    default_jsonl: str
    default_json: str
    video_segment: str
    start_offset_sec: float
    duration_sec: float


def is_missing(value: Any) -> bool:
    """Return True when a value is empty or a null-like token."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def to_text(value: Any) -> str:
    """Convert a value to a trimmed string, empty when missing."""
    if is_missing(value):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def first_present(record: dict[str, Any], candidates: tuple[str, ...]) -> Any:
    """Return the first non-missing value among candidate keys."""
    for column in candidates:
        if column in record and not is_missing(record[column]):
            return record[column]
    return None


def as_dict(value: Any) -> dict[str, Any]:
    """Return the value when it is a dict, otherwise an empty dict."""
    return value if isinstance(value, dict) else {}


def to_valid_timestamp(value: Any) -> int | None:
    """Coerce a value to a non-negative integer timestamp or None."""
    try:
        timestamp = int(value)
    except (TypeError, ValueError):
        return None
    if timestamp < 0:
        return None
    return timestamp


def filename_timestamp(yaml_path: Path) -> int | None:
    """Parse the timestamp from a cot_<timestamp>.yaml file name."""
    stem = yaml_path.stem
    if not stem.startswith("cot_"):
        return None
    return to_valid_timestamp(stem[4:].split("_", 1)[0])


def infer_camera_name_from_prompt(prompt: Any) -> str:
    """Infer the camera name from the saved prompt payload."""
    if not isinstance(prompt, list):
        return ""
    for item in prompt:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, str):
            continue
        match = CAMERA_NAME_RE.search(content)
        if match:
            return match.group(1)
    return ""


def infer_clip_id(yaml_path: Path, entry: dict[str, Any]) -> str:
    """Infer the clip id from the YAML payload or parent directory."""
    clip_id = to_text(entry.get("clip_id"))
    if clip_id:
        return clip_id
    if yaml_path.parent.name:
        return yaml_path.parent.name
    return ""


def extract_coc_label(entry: dict[str, Any]) -> str:
    """Extract the effect_on_ego_behavior text from a CoC entry."""
    final_content = as_dict(entry.get("final_content"))
    ego_schema = as_dict(final_content.get("ego_behavior_schema"))
    return to_text(
        first_present(
            {
                "effect_on_ego_behavior": ego_schema.get("effect_on_ego_behavior"),
                "final_content_effect": final_content.get("effect_on_ego_behavior"),
                "entry_effect": entry.get("effect_on_ego_behavior"),
            },
            ("effect_on_ego_behavior", "final_content_effect", "entry_effect"),
        )
    )


def extract_event_timestamp(entry: dict[str, Any], yaml_path: Path) -> int | None:
    """Extract the event start timestamp from a CoC entry."""
    final_content = as_dict(entry.get("final_content"))
    ego_schema = as_dict(final_content.get("ego_behavior_schema"))
    return first_valid_timestamp(
        entry.get("event_start_timestamp"),
        final_content.get("event_start_timestamp"),
        ego_schema.get("event_start_timestamp"),
        filename_timestamp(yaml_path),
        ego_schema.get("keyframe_range_start_timestamp"),
        final_content.get("keyframe_range_start_timestamp"),
        entry.get("keyframe_range_start_timestamp"),
    )


def first_valid_timestamp(*values: Any) -> int | None:
    """Return the first valid timestamp among candidates."""
    for value in values:
        timestamp = to_valid_timestamp(value)
        if timestamp is not None:
            return timestamp
    return None


def yaml_entries(content: Any) -> list[dict[str, Any]]:
    """Yield candidate result entries from a parsed YAML document."""
    if isinstance(content, list):
        return [item for item in content if isinstance(item, dict)]
    if isinstance(content, dict):
        results = content.get("results")
        if isinstance(results, list):
            return [item for item in results if isinstance(item, dict)]
        if isinstance(results, dict):
            return [results]
        return [content]
    return []


def parse_coc_yaml(yaml_path: Path) -> dict[str, Any] | None:
    """Parse a cot_*.yaml file into a normalized record dict."""
    if yaml is None:
        raise RuntimeError("PyYAML is required to read CoC autolabeler YAML outputs")
    try:
        content = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None

    for entry in yaml_entries(content):
        clip_id = infer_clip_id(yaml_path, entry)
        coc_label = extract_coc_label(entry)
        event_timestamp = extract_event_timestamp(entry, yaml_path)
        if not clip_id or not coc_label:
            continue

        final_content = as_dict(entry.get("final_content"))
        ego_schema = as_dict(final_content.get("ego_behavior_schema"))
        raw_content = as_dict(as_dict(entry.get("raw")).get("final_content"))
        return {
            "clip_id": clip_id,
            "event_start_timestamp": event_timestamp,
            "effect_on_ego_behavior": coc_label,
            "label_class_identifier": to_text(
                first_present(
                    {
                        "ego": ego_schema.get("label_class_identifier"),
                        "final": final_content.get("label_class_identifier"),
                        "entry": entry.get("label_class_identifier"),
                        "raw": raw_content.get("label_class_identifier"),
                    },
                    ("ego", "final", "entry", "raw"),
                )
            ),
            "yaml_path": str(yaml_path),
            "camera_name": infer_camera_name_from_prompt(entry.get("prompt")),
        }
    return None


def discover_coc_yaml_files(coc_result_dir: Path) -> list[Path]:
    """Return all cot_*.yaml files under a result directory."""
    if not coc_result_dir.is_dir():
        raise FileNotFoundError(f"CoC result directory does not exist: {coc_result_dir}")
    return sorted(path for path in coc_result_dir.rglob(COT_YAML_GLOB) if path.is_file())


def infer_segment_video_roots(coc_result_dir: Path, explicit_root: Path | None) -> list[Path]:
    """Build candidate roots for saved 8s segment videos."""
    candidates: list[Path] = []
    if explicit_root is not None:
        candidates.append(explicit_root.expanduser())
    candidates.extend(
        [
            coc_result_dir.parent / "video_segment",
            coc_result_dir / "video_segment",
            coc_result_dir.parent.parent / "experiments" / "video_segment",
        ]
    )
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(candidate.expanduser().resolve())
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(candidate.expanduser())
    return deduped


def resolve_segment_video_path(
    clip_id: str,
    event_timestamp: int | None,
    segment_video_roots: list[Path],
) -> Path | None:
    """Resolve the saved segment video path for a record."""
    if event_timestamp is None:
        return None

    timestamp = str(event_timestamp)
    candidates: list[Path] = []
    for root in segment_video_roots:
        base = root.expanduser()
        candidates.extend(
            [
                base / clip_id / f"{clip_id}_{timestamp}.mp4",
                base / clip_id / f"{timestamp}.mp4",
                base / clip_id[:4] / clip_id / f"{clip_id}_{timestamp}.mp4",
                base / clip_id[:4] / clip_id / f"{timestamp}.mp4",
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_coc_autolabeler_records(
    coc_result_dir: Path,
    *,
    segment_video_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Load records from a CoC autolabeler result directory."""
    segment_roots = infer_segment_video_roots(coc_result_dir, segment_video_root)
    records: list[dict[str, Any]] = []
    for yaml_path in discover_coc_yaml_files(coc_result_dir):
        record = parse_coc_yaml(yaml_path)
        if record is None:
            continue
        event_timestamp = record.get("event_start_timestamp")
        clip_id = to_text(record.get("clip_id"))
        segment_video = resolve_segment_video_path(
            clip_id,
            to_valid_timestamp(event_timestamp),
            segment_roots,
        )
        if segment_video is not None:
            record["video_path"] = str(segment_video)
            record[SAVED_SEGMENT_PROVENANCE_KEY] = True
        records.append(record)
    return records


def read_table(path: Path) -> list[dict[str, Any]]:
    """Read a CSV, JSON, JSONL, or parquet table into dict rows."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict) and isinstance(data.get("rows"), list):
            return [row for row in data["rows"] if isinstance(row, dict)]
        raise ValueError(f"{path}: expected JSON array or object with rows list")
    if suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
        return rows
    if suffix in {".parquet", ".yaml", ".yml"}:
        if suffix in {".yaml", ".yml"}:
            record = parse_coc_yaml(path)
            return [record] if record is not None else []
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "Reading parquet requires pandas plus pyarrow or fastparquet"
            ) from exc
        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"Unsupported input suffix: {path.suffix}")


def load_input_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Load input records from a result directory or table."""
    if args.coc_result_dir is not None:
        return load_coc_autolabeler_records(
            args.coc_result_dir.resolve(),
            segment_video_root=args.segment_video_root,
        )
    assert args.input_table is not None
    return read_table(args.input_table.resolve())


def normalize_record(
    record: dict[str, Any], row_index: int
) -> tuple[NormalizedRow | None, str | None]:
    """Normalize a raw record into a NormalizedRow."""
    clip_id = to_text(first_present(record, CLIP_ID_COLUMN_CANDIDATES))
    label = to_text(first_present(record, LABEL_COLUMN_CANDIDATES))
    if not clip_id:
        return None, "missing_clip_id"
    if not label:
        return None, "missing_coc_label"

    event_timestamp = None
    ts_raw = first_present(record, TIMESTAMP_COLUMN_CANDIDATES)
    if not is_missing(ts_raw):
        try:
            event_timestamp = int(float(str(ts_raw)))
        except ValueError:
            return None, f"invalid_timestamp:{ts_raw}"

    return (
        NormalizedRow(
            row_index=row_index,
            clip_id=clip_id,
            event_timestamp=event_timestamp,
            coc_label=label,
            label_class_identifier=to_text(first_present(record, CLASS_COLUMN_CANDIDATES)),
            yaml_path=to_text(record.get("yaml_path")),
            video_path_hint=to_text(first_present(record, VIDEO_PATH_COLUMN_CANDIDATES)),
            camera_name=to_text(record.get("camera_name")),
            hint_is_saved_segment=bool(record.get(SAVED_SEGMENT_PROVENANCE_KEY, False)),
        ),
        None,
    )


def is_qwen_model_name(model_name: str) -> bool:
    """Return True when the model name is a local Qwen alias."""
    return model_name in QWEN_HF_MODEL_MAP


def is_gpt_cloud_model_name(model_name: str) -> bool:
    """Return True when the model name is a cloud GPT alias."""
    return model_name in {"gpt5", "gpt5.5", "gpt-5", "gpt-5.5"}


def is_gpt5_family_model_name(model_name: str) -> bool:
    """Return True for GPT-5-family names (including provider-qualified forms).

    GPT-5-family chat completion requests must use ``max_completion_tokens``
    instead of ``max_tokens``. This matches the resolved model/backend name,
    e.g. the alias ``gpt5.5`` or NVIDIA inference's ``openai/openai/gpt-5.5``.
    """
    lowered = model_name.lower()
    return "gpt-5" in lowered or "gpt5" in lowered


def resolve_provider_model_name(
    model_name: str, backend: str, provider_model: str | None = None
) -> str:
    """Resolve the provider-side model id for a backend."""
    if provider_model:
        return provider_model
    if is_qwen_model_name(model_name):
        return os.environ.get("LOCAL_OPENAI_MODEL", QWEN_HF_MODEL_MAP[model_name])
    if backend == "nvidia_inference":
        return os.environ.get(
            "NV_INFERENCE_MODEL", NV_INFERENCE_MODEL_MAP.get(model_name, model_name)
        )
    if model_name in {"gpt5", "gpt5.5"}:
        return {"gpt5": "gpt-5", "gpt5.5": "gpt-5.5"}[model_name]
    return model_name


def _has_nv_azure_credentials() -> bool:
    """Return True when NVIDIA-hosted Azure credentials are set."""
    return bool(
        os.environ.get("NVHOST_OAI_CLIENT_ID") and os.environ.get("NVHOST_OAI_CLIENT_SECRET")
    )


def _resolve_api_key(api_key_env: str | None, candidates: tuple[str, ...]) -> str:
    """Return the first available API key from candidate env vars."""
    names: list[str] = []
    if api_key_env:
        names.append(api_key_env)
    names.extend(candidates)
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return "EMPTY"


def create_model_runtime(
    model_name: str,
    *,
    api_base: str | None = None,
    api_key_env: str | None = None,
    provider_model: str | None = None,
) -> ModelRuntime:
    """Create an evaluator runtime for the requested model."""
    from openai import OpenAI

    if is_qwen_model_name(model_name):
        base_url = (
            api_base or os.environ.get("LOCAL_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        )
        if not base_url:
            raise ValueError(
                "Local Qwen evaluation requires LOCAL_OPENAI_BASE_URL pointing to an "
                "OpenAI-compatible vLLM server."
            )
        api_key = _resolve_api_key(api_key_env, ("LOCAL_OPENAI_API_KEY", "OPENAI_API_KEY"))
        resolved_model = resolve_provider_model_name(model_name, "local", provider_model)
        return ModelRuntime(
            client=OpenAI(base_url=base_url.rstrip("/"), api_key=api_key),
            model_name=resolved_model,
            backend="local",
            omit_temperature=False,
            qwen_disable_thinking=True,
            default_temperature=0.0,
            uses_max_completion_tokens=False,
        )

    if api_base is None and _has_nv_azure_credentials():
        repo_src = Path(__file__).resolve().parents[2] / "src"
        if repo_src.is_dir() and str(repo_src) not in sys.path:
            sys.path.insert(0, str(repo_src))
        try:
            from coc_labeling.model_clients.openai_client import create_cloud_openai_client
        except ImportError as exc:
            raise RuntimeError(
                "NVHOST_OAI_CLIENT_ID/NVHOST_OAI_CLIENT_SECRET are set, but the "
                "coc_labeling.model_clients.openai_client module is unavailable. Install the "
                "coc_label_oss package (pip install -e projects/coc_label_oss) to use the "
                "NVIDIA-hosted Azure route, or set NVIDIA_API_KEY / OPENAI_API_KEY instead."
            ) from exc

        cloud = create_cloud_openai_client(api_version=NV_AZURE_API_VERSION)
        resolved_model = resolve_provider_model_name(model_name, cloud.backend, provider_model)
        return ModelRuntime(
            client=cloud.client,
            model_name=resolved_model,
            backend=cloud.backend,
            omit_temperature=is_gpt_cloud_model_name(model_name),
            qwen_disable_thinking=False,
            default_temperature=0.2,
            uses_max_completion_tokens=is_gpt5_family_model_name(resolved_model),
        )

    if api_base is not None and api_key_env:
        api_key = _resolve_api_key(api_key_env, ())
        if api_key != "EMPTY":
            resolved_model = resolve_provider_model_name(model_name, "openai", provider_model)
            return ModelRuntime(
                client=OpenAI(base_url=api_base.rstrip("/"), api_key=api_key),
                model_name=resolved_model,
                backend="openai_compatible",
                omit_temperature=is_gpt_cloud_model_name(model_name),
                qwen_disable_thinking=False,
                default_temperature=0.2,
                uses_max_completion_tokens=is_gpt5_family_model_name(resolved_model),
            )

    if os.environ.get("NVIDIA_API_KEY"):
        base_url = api_base or os.environ.get("NV_INFERENCE_URL", NV_INFERENCE_URL)
        api_key = _resolve_api_key(api_key_env, ("NVIDIA_API_KEY",))
        resolved_model = resolve_provider_model_name(model_name, "nvidia_inference", provider_model)
        return ModelRuntime(
            client=OpenAI(base_url=base_url.rstrip("/"), api_key=api_key),
            model_name=resolved_model,
            backend="nvidia_inference",
            omit_temperature=is_gpt_cloud_model_name(model_name),
            qwen_disable_thinking=False,
            default_temperature=0.2,
            uses_max_completion_tokens=is_gpt5_family_model_name(resolved_model),
        )

    if os.environ.get("OPENAI_API_KEY"):
        client_kwargs: dict[str, Any] = {
            "api_key": _resolve_api_key(api_key_env, ("OPENAI_API_KEY",)),
        }
        base_url = api_base or os.environ.get("OPENAI_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url.rstrip("/")
        resolved_model = resolve_provider_model_name(model_name, "openai", provider_model)
        return ModelRuntime(
            client=OpenAI(**client_kwargs),
            model_name=resolved_model,
            backend="openai",
            omit_temperature=is_gpt_cloud_model_name(model_name),
            qwen_disable_thinking=False,
            default_temperature=0.2,
            uses_max_completion_tokens=is_gpt5_family_model_name(resolved_model),
        )

    raise ValueError(
        "Please configure evaluator credentials. For cloud GPT models, set "
        "NVHOST_OAI_CLIENT_ID/NVHOST_OAI_CLIENT_SECRET, NVIDIA_API_KEY, or OPENAI_API_KEY. "
        "For any OpenAI-compatible endpoint, pass --api-base together with --api-key-env "
        "naming the env var that holds the key. For local Qwen models, set "
        "LOCAL_OPENAI_BASE_URL and LOCAL_OPENAI_API_KEY."
    )


def apply_model_defaults(args: argparse.Namespace) -> ModelRuntime | None:
    """Apply per-model defaults and build the runtime."""
    if getattr(args, "omit_temperature", None) is None:
        args.omit_temperature = is_gpt_cloud_model_name(args.model_name)
    if getattr(args, "qwen_disable_thinking", None) is None:
        args.qwen_disable_thinking = is_qwen_model_name(args.model_name)
    if args.dry_run:
        return None
    return create_model_runtime(
        args.model_name,
        api_base=None if args.api_base == "auto" else args.api_base,
        api_key_env=args.api_key_env,
        provider_model=args.provider_model,
    )


def completion_finish_reason(resp: Any) -> str | None:
    """Return the finish reason from a chat completion response."""
    try:
        return resp.choices[0].finish_reason
    except (AttributeError, IndexError, TypeError):
        return None


def call_chat_completion(
    runtime: ModelRuntime,
    *,
    user_content: list[dict[str, Any]],
    max_tokens: int,
    temperature: float | None,
    qwen_disable_thinking: bool,
) -> tuple[Any, str, int]:
    """Call the chat completion endpoint, retrying on truncation."""
    tokens = max_tokens
    last_resp = None
    raw_text = ""
    finish_reason = None
    token_param = "max_completion_tokens" if runtime.uses_max_completion_tokens else "max_tokens"
    for attempt in range(2):
        request_kwargs: dict[str, Any] = {
            "model": runtime.model_name,
            "messages": [{"role": "user", "content": user_content}],
            token_param: tokens,
        }
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if qwen_disable_thinking:
            request_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        if hasattr(runtime.client, "chat_completion"):
            last_resp = runtime.client.chat_completion(**request_kwargs)
        else:
            last_resp = runtime.client.chat.completions.create(**request_kwargs)

        finish_reason = completion_finish_reason(last_resp)
        raw_text = strip_thinking_suffix(last_resp.choices[0].message.content or "")
        if finish_reason != "length" or attempt == 1:
            break
        tokens = min(tokens * 2, 8192)
    assert last_resp is not None
    return last_resp, raw_text, tokens


def resolve_api_key(api_key_env: str | None) -> str:
    """Return the first available API key from the environment."""
    candidates = []
    if api_key_env:
        candidates.append(api_key_env)
    candidates.extend(("OPENAI_API_KEY", "NVIDIA_API_KEY", "LOCAL_OPENAI_API_KEY"))
    for name in candidates:
        value = os.environ.get(name)
        if value:
            return value
    return "EMPTY"


def resolve_video_path(
    row: NormalizedRow,
    video_root: Path | None,
    camera_name: str,
) -> tuple[Path | None, str | None]:
    """Resolve the source video path and its provenance for a record.

    Returns a ``(path, source)`` pair where ``source`` is ``"hint"`` when the
    record's own ``video_path`` was used, or ``"video_root"`` when the path was
    discovered under ``--video-root``. A ``"video_root"`` match is always a full
    raw camera video and therefore requires raw-video event alignment.
    """
    if row.video_path_hint:
        hinted = Path(row.video_path_hint).expanduser()
        if hinted.is_file():
            return hinted, "hint"

    clip_id = row.clip_id
    if video_root is None:
        return None, None

    root = video_root.expanduser()
    camera_file = f"{camera_name}.mp4"
    candidates = [
        root / "camera" / f"{clip_id}.{camera_file}",
        root / f"{clip_id}.{camera_file}",
        root / f"{clip_id}_fpv.mp4",
        root / f"{clip_id}.mp4",
        root / clip_id[:4] / clip_id / "recordings" / "recorder-00" / camera_file,
        root / clip_id / "recordings" / "recorder-00" / camera_file,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate, "video_root"

    if root.exists():
        for pattern in (
            f"{clip_id[:4]}/{clip_id}/recordings/recorder-*/{camera_file}",
            f"*/{clip_id}/recordings/recorder-*/{camera_file}",
            f"{clip_id}/**/{camera_file}",
            f"{clip_id[:4]}/{clip_id}/**/{camera_file}",
        ):
            match = next(root.glob(pattern), None)
            if match is not None and match.is_file():
                return match, "video_root"
    return None, None


def resolve_row_video_mode(
    configured_mode: str,
    *,
    hint_is_saved_segment: bool,
    video_source: str | None,
) -> str:
    """Choose the per-row video mode from the resolved video's provenance.

    An explicit ``--video-mode`` (``raw`` or ``segment_8s``) always wins. Under
    ``auto`` a row is treated as a pre-saved 8-second segment only when its own
    ``video_path`` hint is a tracked saved segment; any row that falls back to
    ``--video-root`` resolves a full raw camera video and uses ``raw`` event
    alignment.
    """
    if configured_mode != "auto":
        return configured_mode
    if video_source == "hint" and hint_is_saved_segment:
        return "segment_8s"
    return "raw"


def find_timestamp_sidecar(video_path: Path) -> Path | None:
    """Find a timestamp sidecar file next to a video."""
    base = video_path.with_suffix("")
    for candidate in (
        Path(str(base) + ".timestamps.parquet"),
        Path(str(base) + ".timestamps"),
        Path(str(video_path) + ".timestamps.parquet"),
        Path(str(video_path) + ".timestamps"),
    ):
        if candidate.is_file():
            return candidate
    return None


def load_timestamp_mapping(sidecar_path: Path) -> list[tuple[int, int]]:
    """Load a frame-to-timestamp mapping from a sidecar file."""
    if sidecar_path.suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Reading timestamp parquet requires pandas plus pyarrow") from exc
        df = pd.read_parquet(sidecar_path)
        if df.index.name is not None:
            df = df.reset_index()
        frame_col = next(
            (
                c
                for c in ("frame_index", "frame_idx", "index_in_video", "frame", "idx")
                if c in df.columns
            ),
            None,
        )
        ts_col = next(
            (
                c
                for c in (
                    "timestamp_micros",
                    "timestamp_us",
                    "timestamp",
                    "ts",
                    "timestamp_microseconds",
                )
                if c in df.columns
            ),
            None,
        )
        if frame_col is None or ts_col is None:
            raise ValueError(
                f"Cannot infer timestamp columns from {sidecar_path}: {list(df.columns)}"
            )
        return sorted(
            (int(frame), int(ts))
            for frame, ts in df[[frame_col, ts_col]].itertuples(index=False, name=None)
        )

    mapping = []
    with sidecar_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                mapping.append((int(parts[0]), int(parts[1])))
    return sorted(mapping)


def nearest_frame_index(mapping: list[tuple[int, int]], query_ts: int) -> int:
    """Return the frame index nearest to a query timestamp."""
    if not mapping:
        raise IndexError("empty_timestamp_mapping")
    frames = [frame for frame, _ in mapping]
    timestamps = [ts for _, ts in mapping]
    if query_ts < timestamps[0] or query_ts > timestamps[-1]:
        raise IndexError(
            f"timestamp_out_of_range:{query_ts}:range={timestamps[0]}..{timestamps[-1]}"
        )
    idx = bisect.bisect_left(timestamps, query_ts)
    if idx == 0:
        return frames[0]
    if idx >= len(timestamps):
        return frames[-1]
    before_delta = query_ts - timestamps[idx - 1]
    after_delta = timestamps[idx] - query_ts
    return frames[idx] if after_delta < before_delta else frames[idx - 1]


def get_video_fps(video_path: Path) -> float:
    """Return the frame rate of a video, defaulting to 30 fps."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        rate = json.loads(result.stdout)["streams"][0]["r_frame_rate"]
        if "/" in rate:
            num, den = rate.split("/", 1)
            fps = float(num) / float(den)
        else:
            fps = float(rate)
        if fps > 0:
            return fps
    except Exception:
        pass
    return 30.0


def event_time_seconds(
    row: NormalizedRow, video_path: Path, timestamp_mode: str, relative_timestamp_max_sec: float
) -> tuple[float | None, str | None]:
    """Convert an event timestamp to seconds within the video."""
    if row.event_timestamp is None:
        return None, "missing_event_timestamp"

    if timestamp_mode in {"auto", "sidecar"}:
        sidecar = find_timestamp_sidecar(video_path)
        if sidecar is not None:
            try:
                frame_index = nearest_frame_index(
                    load_timestamp_mapping(sidecar), row.event_timestamp
                )
                return frame_index / get_video_fps(video_path), None
            except Exception as exc:
                return None, f"timestamp_sidecar_error:{exc}"
        if timestamp_mode == "sidecar":
            return None, "missing_timestamp_sidecar"

    if timestamp_mode in {"auto", "relative"}:
        if 0 <= row.event_timestamp <= relative_timestamp_max_sec * 1_000_000:
            return row.event_timestamp / 1_000_000.0, None
        return None, "timestamp_requires_sidecar"

    return None, f"unsupported_timestamp_mode:{timestamp_mode}"


def extract_segment_ffmpeg(src: Path, start_sec: float, duration_sec: float, dst: Path) -> None:
    """Cut a video segment with ffmpeg."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{max(0.0, start_sec):.6f}",
            "-i",
            str(src),
            "-t",
            f"{duration_sec:.6f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-an",
            str(dst),
        ],
        check=True,
    )


def prepare_stage_video(
    row: NormalizedRow,
    video_path: Path,
    spec: StageSpec,
    args: argparse.Namespace,
    temp_dir: Path,
    video_mode: str,
) -> tuple[Path | None, dict[str, Any], str | None]:
    """Prepare the stage video segment for a record using the per-row mode."""
    metadata: dict[str, Any] = {
        "source_video_path": str(video_path),
        "timestamp_mode": args.timestamp_mode,
        "video_mode": video_mode,
        "segment_start_sec": None,
        "segment_duration_sec": None,
    }
    if video_mode == "segment_8s":
        base_start = 0.0
    else:
        event_sec, err = event_time_seconds(
            row, video_path, args.timestamp_mode, args.relative_timestamp_max_sec
        )
        if err:
            return None, metadata, err
        assert event_sec is not None
        base_start = event_sec - 2.0

    start_sec = max(0.0, base_start + spec.start_offset_sec)
    metadata["segment_start_sec"] = start_sec
    metadata["segment_duration_sec"] = spec.duration_sec

    if video_mode == "segment_8s" and spec.start_offset_sec == 0.0 and spec.duration_sec >= 8.0:
        return video_path, metadata, None

    segment_path = (
        temp_dir / f"{row.clip_id}_{row.event_timestamp or row.row_index}_stage{spec.stage}.mp4"
    )
    try:
        extract_segment_ffmpeg(video_path, start_sec, spec.duration_sec, segment_path)
    except subprocess.CalledProcessError as exc:
        return None, metadata, f"ffmpeg_segment_error:{exc}"
    return segment_path, metadata, None


def is_frame_based_vision_model(model: str) -> bool:
    """Return True when the model expects sampled image frames."""
    leaf = str(model).lower().replace("_", "-").rsplit("/", 1)[-1]
    return leaf.startswith("gpt-5") or leaf.startswith("gpt5") or "gemini" in leaf


def extract_video_frame_jpegs(video_path: Path, fps: float, max_frames: int | None) -> list[bytes]:
    """Extract sampled JPEG frames from a video."""
    with tempfile.TemporaryDirectory(prefix="coc_frames_") as temp_dir:
        pattern = str(Path(temp_dir) / "frame_%03d.jpg")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps={fps}",
        ]
        if max_frames is not None:
            cmd.extend(["-frames:v", str(max_frames)])
        cmd.append(pattern)
        subprocess.run(cmd, check=True)
        frame_paths = sorted(Path(temp_dir).glob("frame_*.jpg"))
        if not frame_paths:
            raise RuntimeError(f"no_frames_extracted:{video_path}")
        return [frame_path.read_bytes() for frame_path in frame_paths]


def build_user_content(
    video_path: Path, text: str, args: argparse.Namespace, *, model_name: str
) -> list[dict[str, Any]]:
    """Build the user message content for the evaluator request."""
    parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
    content_type = args.video_content_type
    if content_type == "auto":
        content_type = "image_frames" if is_frame_based_vision_model(model_name) else "video_url"

    if content_type == "image_frames":
        if args.video_payload != "base64":
            raise ValueError("image_frames content requires --video-payload base64")
        for frame_bytes in extract_video_frame_jpegs(
            video_path, args.video_frame_fps, args.video_max_frames
        ):
            frame_b64 = base64.standard_b64encode(frame_bytes).decode("ascii")
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}", "detail": "high"},
                }
            )
        return parts

    if args.video_payload == "base64":
        encoded = base64.standard_b64encode(video_path.read_bytes()).decode("ascii")
        payload_url = f"data:video/mp4;base64,{encoded}"
    elif args.video_payload == "file_url":
        payload_url = video_path.resolve().as_uri()
    elif args.video_payload == "http_url":
        if not args.http_video_url:
            raise ValueError("video_payload=http_url requires --http-video-url")
        payload_url = args.http_video_url
    else:
        raise ValueError(f"unknown_video_payload:{args.video_payload}")

    if content_type == "file":
        parts.append(
            {"type": "file", "file": {"filename": video_path.name, "file_data": payload_url}}
        )
    elif content_type == "video_url":
        parts.append({"type": "video_url", "video_url": {"url": payload_url}})
    else:
        raise ValueError(f"unknown_video_content_type:{content_type}")
    return parts


def strip_thinking_suffix(text: str) -> str:
    """Strip any thinking-mode prefix from model output."""
    value = text.strip()
    if _THINKING_CLOSE_TAG in value:
        value = value.split(_THINKING_CLOSE_TAG, 1)[-1].strip()
    return value


def parse_model_json(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse a JSON object from raw model output."""
    value = raw.strip()
    if not value:
        return None, "empty_response"
    candidates = [value]
    match = _JSON_FENCE_RE.search(value)
    if match:
        candidates.append(match.group(1).strip())
    lb, rb = value.find("{"), value.rfind("}")
    if lb != -1 and rb > lb:
        candidates.append(value[lb : rb + 1])
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj, None
        except json.JSONDecodeError:
            pass
    return None, "json_parse_failed"


def is_score_eval_key(key: str) -> bool:
    """Return True when an eval key is a numeric score expected in [0, 1]."""
    return key.endswith("_score") or key == "decision_supported_by_video"


def is_valid_score_value(value: Any) -> bool:
    """Return True when ``value`` is a finite number in ``[0, 1]``."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        number = float(value)
        return math.isfinite(number) and 0.0 <= number <= 1.0
    if isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return False
        return math.isfinite(number) and 0.0 <= number <= 1.0
    return False


def validate_parsed_evaluation(
    parsed: dict[str, Any],
    eval_keys: tuple[str, ...],
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate required score fields and build the evaluation payload.

    Every score key listed in ``eval_keys`` must be present, numeric, finite,
    and within ``[0, 1]``. Non-score keys such as ``explanation`` are copied
    through when present but do not gate ``ok``.
    """
    for key in eval_keys:
        if not is_score_eval_key(key):
            continue
        if key not in parsed:
            return None, f"validation_error:missing_score:{key}"
        if not is_valid_score_value(parsed[key]):
            return None, f"validation_error:invalid_score:{key}"

    evaluation: dict[str, Any] = {}
    for key in eval_keys:
        if key not in parsed:
            continue
        if is_score_eval_key(key):
            evaluation[key] = float(parsed[key])
        else:
            evaluation[key] = parsed[key]
    return evaluation, None


def is_completed_result_row(row: dict[str, Any]) -> bool:
    """Return True only for a successfully parsed and validated evaluation.

    Rows carrying ``error`` (e.g. ``api_error``/``missing_video``), a
    ``parse_error``, ``ok=False``, or an empty ``evaluation`` represent
    retryable failures and must stay eligible for ``--resume``.
    """
    if row.get("error"):
        return False
    if row.get("parse_error"):
        return False
    if not row.get("ok"):
        return False
    evaluation = row.get("evaluation")
    return isinstance(evaluation, dict) and bool(evaluation)


def load_done_keys(jsonl_path: Path) -> set[str]:
    """Load row keys that already hold a validated evaluation.

    Only completed rows are returned so that rows with retryable failures
    (API errors, parse errors, missing videos, etc.) remain eligible for
    reprocessing on ``--resume``.
    """
    done = set()
    if not jsonl_path.exists():
        return done
    for line in jsonl_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = row.get("row_key")
        if isinstance(key, str) and is_completed_result_row(row):
            done.add(key)
    return done


def _should_prefer_result_row(candidate: dict[str, Any], existing: dict[str, Any]) -> bool:
    """Return True when ``candidate`` should replace ``existing`` for one row_key."""
    candidate_done = is_completed_result_row(candidate)
    existing_done = is_completed_result_row(existing)
    if candidate_done != existing_done:
        return candidate_done
    return True


def dedupe_result_rows_by_key(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one row per ``row_key``, preferring completed evaluations.

    When multiple rows share a key, a completed row wins over retryable failures.
    Otherwise the later JSONL line wins. Rows without a string ``row_key`` are kept
    in encounter order.
    """
    keyed_index: dict[str, int] = {}
    deduped: list[dict[str, Any]] = []

    for row in rows:
        key = row.get("row_key")
        if not isinstance(key, str):
            deduped.append(row)
            continue
        if key not in keyed_index:
            keyed_index[key] = len(deduped)
            deduped.append(row)
            continue
        idx = keyed_index[key]
        if _should_prefer_result_row(row, deduped[idx]):
            deduped[idx] = row

    return deduped


def finalize_json_from_jsonl(jsonl_path: Path, json_path: Path) -> None:
    """Write a finalized JSON array from a JSONL file.

    Duplicate ``row_key`` entries are collapsed so ``--resume`` retries do not
    leave stale failed rows alongside newer successful ones in the output JSON.
    """
    rows = []
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    rows = dedupe_result_rows_by_key(rows)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append a row to a JSONL file and flush."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def add_common_args(parser: argparse.ArgumentParser, spec: StageSpec) -> None:
    """Register the CLI arguments shared by all stages."""
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--coc-result-dir",
        type=Path,
        help="CoC autolabeler experiment dir with <clip_id>/cot_<timestamp>.yaml files.",
    )
    input_group.add_argument(
        "--input-table",
        type=Path,
        help="Optional legacy CSV/JSON/JSONL/parquet table input.",
    )
    parser.add_argument(
        "--segment-video-root",
        type=Path,
        default=None,
        help="Root of saved 8s segment videos (default: auto-detect experiments/video_segment).",
    )
    parser.add_argument(
        "--video-root", type=Path, default=None, help="Root containing raw full camera videos."
    )
    parser.add_argument("--camera-name", default=DEFAULT_CAMERA_NAME)
    parser.add_argument(
        "--video-mode",
        choices=("auto", "raw", "segment_8s"),
        default="auto",
        help=(
            "auto (default) picks the mode per row from segment provenance: rows whose "
            "video_path is a saved 8s segment use segment_8s, rows resolved via --video-root "
            "use raw event alignment. Pass raw or segment_8s to force one mode for all rows."
        ),
    )
    parser.add_argument("--timestamp-mode", choices=("auto", "sidecar", "relative"), default="auto")
    parser.add_argument(
        "--relative-timestamp-max-sec", type=float, default=DEFAULT_RELATIVE_TIMESTAMP_MAX_SEC
    )
    parser.add_argument(
        "--model-name",
        choices=SUPPORTED_MODEL_NAMES,
        default=DEFAULT_MODEL_NAME,
        help="Evaluator model alias, matching coc_labeling model_name values.",
    )
    parser.add_argument(
        "--provider-model",
        default=None,
        help="Optional provider-side model override, similar to NV_INFERENCE_MODEL.",
    )
    parser.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help="Optional API base URL override. Defaults to auto-detect from environment.",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Environment variable containing the API key when an override is needed.",
    )
    parser.add_argument("--jsonl-out", type=Path, default=Path(spec.default_jsonl))
    parser.add_argument("--json-out", type=Path, default=Path(spec.default_json))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--video-payload", choices=("base64", "file_url", "http_url"), default="base64"
    )
    parser.add_argument(
        "--video-content-type",
        choices=("auto", "video_url", "file", "image_frames"),
        default="auto",
    )
    parser.add_argument("--video-frame-fps", type=float, default=2.0)
    parser.add_argument("--video-max-frames", type=int, default=None)
    parser.add_argument("--http-video-url", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--omit-temperature", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--qwen-disable-thinking", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--keep-temp-segments", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")


def resolve_runtime_video_settings(args: argparse.Namespace, records: list[dict[str, Any]]) -> None:
    """Resolve the timestamp alignment mode.

    ``--video-mode`` is intentionally left untouched (``auto`` stays ``auto``) so
    that :func:`resolve_row_video_mode` can pick the mode per row from each
    record's segment provenance instead of collapsing to one global mode.
    """
    if args.timestamp_mode == "auto":
        args.timestamp_mode = "relative" if args.coc_result_dir is not None else "sidecar"


def base_result_row(spec: StageSpec, row: NormalizedRow | None, raw_index: int) -> dict[str, Any]:
    """Build the base result row for a record."""
    row_key = f"{row.clip_id}_{row.event_timestamp}_{raw_index}" if row else f"row_{raw_index}"
    return {
        "stage": spec.stage,
        "row_key": row_key,
        "row_index": raw_index,
        "clip_id": row.clip_id if row else None,
        "event_timestamp": row.event_timestamp if row else None,
        "video_segment": spec.video_segment,
        "source_video_path": None,
        "segment_mp4_path": None,
        "coc_label": row.coc_label if row else None,
        "label_class_identifier": row.label_class_identifier if row else None,
        "yaml_path": row.yaml_path if row else None,
        "evaluator_model_name": None,
        "evaluator_backend": None,
        "ok": False,
        "error": None,
        "model_response_raw": None,
        "evaluation": None,
        "parse_error": None,
        "finish_reason": None,
        "max_tokens_used": None,
    }


def run_stage(
    spec: StageSpec,
    prompt_builder: Callable[[NormalizedRow], str],
    eval_keys: tuple[str, ...],
    argv: list[str] | None = None,
) -> int:
    """Run a validation stage over all input records."""
    parser = argparse.ArgumentParser(description=f"CoC validation {spec.name}.")
    add_common_args(parser, spec)
    args = parser.parse_args(argv)

    if args.finalize_only:
        finalize_json_from_jsonl(args.jsonl_out, args.json_out)
        print(f"wrote {args.json_out} from {args.jsonl_out}", file=sys.stderr)
        return 0

    records = load_input_records(args)
    if not records:
        print("No input rows found.", file=sys.stderr)
        return 1

    resolve_runtime_video_settings(args, records)
    runtime = apply_model_defaults(args)
    done = load_done_keys(args.jsonl_out) if args.resume else set()

    processed = 0
    temp_dir = Path(tempfile.mkdtemp(prefix=f"coc_stage{spec.stage}_"))

    def emit(result: dict[str, Any]) -> None:
        if args.dry_run:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            append_jsonl(args.jsonl_out, result)

    try:
        for raw_index, record in enumerate(records):
            if args.limit is not None and processed >= args.limit:
                break
            row, row_error = normalize_record(record, raw_index)
            result = base_result_row(spec, row, row.row_index if row is not None else raw_index)
            if result["row_key"] in done:
                continue

            if row_error:
                result["error"] = row_error
                emit(result)
                processed += 1
                continue
            assert row is not None

            camera_name = row.camera_name or args.camera_name
            video_path, video_source = resolve_video_path(row, args.video_root, camera_name)
            if video_path is None:
                result["error"] = "missing_video"
                emit(result)
                processed += 1
                continue
            result["source_video_path"] = str(video_path)

            row_video_mode = resolve_row_video_mode(
                args.video_mode,
                hint_is_saved_segment=row.hint_is_saved_segment,
                video_source=video_source,
            )
            segment_path, segment_meta, err = prepare_stage_video(
                row, video_path, spec, args, temp_dir, row_video_mode
            )
            result.update(segment_meta)
            if err:
                result["error"] = err
                emit(result)
                processed += 1
                continue
            assert segment_path is not None
            result["segment_mp4_path"] = str(segment_path)

            prompt = prompt_builder(row)
            if args.dry_run:
                result["prompt_preview"] = prompt[:900]
                emit(result)
                if not args.keep_temp_segments and segment_path != video_path:
                    segment_path.unlink(missing_ok=True)
                processed += 1
                continue

            try:
                assert runtime is not None
                user_content = build_user_content(
                    segment_path, prompt, args, model_name=runtime.model_name
                )
                max_tokens = args.max_tokens
                if max_tokens <= 1024 and is_frame_based_vision_model(runtime.model_name):
                    max_tokens = 4096
                temperature = args.temperature
                if temperature is None and not args.omit_temperature:
                    temperature = runtime.default_temperature
                resp, raw_text, max_tokens_used = call_chat_completion(
                    runtime,
                    user_content=user_content,
                    max_tokens=max_tokens,
                    temperature=None if args.omit_temperature else temperature,
                    qwen_disable_thinking=bool(args.qwen_disable_thinking),
                )
                finish_reason = completion_finish_reason(resp)
                result["evaluator_model_name"] = args.model_name
                result["evaluator_backend"] = runtime.backend
            except Exception as exc:
                result["error"] = f"api_error:{exc}"
                emit(result)
                if not args.keep_temp_segments and segment_path != video_path:
                    segment_path.unlink(missing_ok=True)
                processed += 1
                continue

            if not args.keep_temp_segments and segment_path != video_path:
                segment_path.unlink(missing_ok=True)

            parsed, parse_error = parse_model_json(raw_text)
            result["model_response_raw"] = raw_text
            result["finish_reason"] = finish_reason
            result["max_tokens_used"] = max_tokens_used
            if parsed is None:
                result["parse_error"] = parse_error
            else:
                evaluation, validation_error = validate_parsed_evaluation(parsed, eval_keys)
                if validation_error:
                    result["parse_error"] = validation_error
                else:
                    result["evaluation"] = evaluation
                    result["ok"] = True
            emit(result)
            processed += 1
    finally:
        if not args.keep_temp_segments:
            for path in temp_dir.glob("*.mp4"):
                path.unlink(missing_ok=True)
            try:
                temp_dir.rmdir()
            except OSError:
                pass

    if not args.dry_run:
        finalize_json_from_jsonl(args.jsonl_out, args.json_out)
        print(f"wrote {args.jsonl_out} and {args.json_out}", file=sys.stderr)
    else:
        print("dry-run: no json writes", file=sys.stderr)
    return 0
