"""Exercise released recipes on tiny local media, with HTTP and ffprobe mocked."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.abc
import importlib.machinery
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

os.environ["OPENCV_FFMPEG_THREADS"] = "1"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "threads;1"


def native_fixture(root, base, ids):
    """Bind the native policy to prepared sensor files and fake HTTP."""
    import cv2
    import numpy as np
    import pandas as pd
    import yaml

    cv2.setNumThreads(1)

    base.mkdir(parents=True, exist_ok=True)
    (base / "scratch").mkdir(exist_ok=True)
    camera = "camera_front_wide_120fov"
    stamps = list(range(0, 4_000_000, 200_000))
    sensors = {}
    for cid in ids:
        folder = base / "data" / cid
        folder.mkdir(parents=True, exist_ok=True)
        video = folder / (camera + ".mp4")
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (640, 480))
        assert writer.isOpened(), "Native fixture video encoder unavailable"
        for _ in stamps:
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.circle(image, (320, 90), 25, (0, 0, 255), -1)
            writer.write(image)
        writer.release()
        np.savetxt(
            str(video) + ".timestamps", np.column_stack((range(len(stamps)), stamps)), fmt="%d"
        )
        ego = pd.DataFrame(
            {
                "t_us": [*stamps, 4_000_000],
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "qw": 1.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
            }
        )
        ego_path = folder / "ego.parquet"
        ego.to_parquet(ego_path, index=False)
        capture_path = Path(str(video) + ".timestamps")
        sensors[cid] = {
            "clock": {"origin_us": 0, "end_us": 4_000_000},
            "ego": {
                "path": str(ego_path),
                "sha256": hashlib.sha256(ego_path.read_bytes()).hexdigest(),
            },
            "camera": {
                "video": str(video),
                "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                "timestamps": str(capture_path),
                "timestamps_sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(),
            },
            "source": {"fixture": "measured_ego_poses"},
        }
    (base / "sensors.json").write_text(
        json.dumps({"schema": "alparsg_sensor_inputs_v1", "clips": sensors})
    )
    navigation = {
        cid: {
            "available": True,
            "source": {"fixture": "route"},
            "samples": [
                {
                    "timestamp_us": 0,
                    "maneuver": "STRAIGHT",
                    "remaining_m": 100.0,
                    "instruction_status": "active",
                }
            ],
        }
        for cid in ids
    }
    actions = {
        cid: {
            "available": True,
            "source": {"fixture": "production-actions"},
            "coverage_gaps_us": [],
            "coverage_overlaps_us": [],
            "reason": "fixture",
            "record_id": "fixture-actions",
            "clock": {"origin_us": 0, "end_us": 4_000_000},
            "actions": [
                {"action": "Stop", "group": "longitudinal", "start_us": 0, "end_us": 4_000_000}
            ],
        }
        for cid in ids
    }
    (base / "navigation.json").write_text(
        json.dumps({"schema": "alparsg_navigation_timeline_v1", "clips": navigation})
    )
    (base / "actions.json").write_text(
        json.dumps({"schema": "alparsg_production_meta_actions_v1", "clips": actions})
    )
    (base / "inputs.json").write_text(
        json.dumps(
            {
                "clips": [
                    {
                        "clip_id": cid,
                        "clock": {"origin_us": 0, "end_us": 4_000_000},
                        "clock_verified": True,
                        "sample_anchors_us": [],
                    }
                    for cid in ids
                ]
            }
        )
    )
    config_dir = root / __package__.split(".")[0] / "tasks/streaming_rsg/configs"
    config = yaml.safe_load((config_dir / "wl_readable_context.yaml").read_text())
    compiler = yaml.safe_load((config_dir / "wl_readable_context.compiler.yaml").read_text())
    endpoint = json.loads((config_dir / "wl_readable_context_registry/r0.json").read_text())
    assert not compiler["infra"].get("data_dir") and not compiler["infra"].get("metadb")
    compiler["playground"]["env"].update(
        sensor_manifest=str(base / "sensors.json"),
        navigation_manifest=str(base / "navigation.json"),
        meta_action_manifest=str(base / "actions.json"),
    )
    (base / "compiler.yaml").write_text(yaml.safe_dump(compiler))
    (base / "registry").mkdir(exist_ok=True)
    (base / "registry/r0.json").write_text(
        json.dumps(
            {
                **endpoint,
                "status": "ready",
                "model": "offline-native-recipe",
                "url": endpoint["url"],
                "api_key_env": "RELEASE_SMOKE_API_KEY",
                "pricing": {"input_usd_per_million_tokens": 2, "output_usd_per_million_tokens": 8},
            }
        )
    )
    config.update(
        allow_paid_dispatch=True,
        input_manifest=str(base / "inputs.json"),
        compiler_config=str(base / "compiler.yaml"),
        endpoint_registry=str(base / "registry"),
        scratch_dir=str(base / "scratch"),
    )
    config["budget"] = {
        "owner": "offline-native-audit",
        "campaign": "offline-native-audit",
        "ledger": str(base / "ledger"),
        "accounting_verified_at": datetime.now(timezone.utc),
        "dispatch_envelope_usd": 100,
        "policy": {
            "monthly_limit_usd": 1000,
            "reserve_usd": 0,
            "campaign_limit_usd": 1000,
            "daily_limit_usd": 1000,
            "large_job_usd": 1000,
        },
    }
    (base / "generation.yaml").write_text(yaml.safe_dump(config))
    return base / "generation.yaml"


def video_fixture(root, base, ids):
    """Prepare hash-bound camera-only MP4 inputs with the released video recipe."""
    import cv2
    import numpy as np
    import yaml

    cv2.setNumThreads(1)

    base.mkdir(parents=True, exist_ok=True)
    (base / "scratch").mkdir(exist_ok=True)
    videos = base / "videos"
    videos.mkdir()
    for index in range(2):
        path = videos / f"clip{index}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (640, 480))
        assert writer.isOpened()
        for _ in range(60):
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.circle(image, (320 + index, 90), 25, (0, 0, 255), -1)
            writer.write(image)
        writer.release()
    configs = root / __package__.split(".")[0] / "tasks/streaming_rsg/configs"
    config = yaml.safe_load((configs / "video_wl_readable.generation.yaml").read_text())
    endpoint = json.loads((configs / "video_registry/r0.json").read_text())
    compiler_path = configs / "video_wl_readable.compiler.yaml"
    (base / "registry").mkdir()
    (base / "registry/r0.json").write_text(
        json.dumps(
            {
                **endpoint,
                "status": "ready",
                "model": "offline-native-recipe",
                "url": endpoint["url"],
                "api_key_env": "RELEASE_SMOKE_API_KEY",
                "pricing": {"input_usd_per_million_tokens": 2, "output_usd_per_million_tokens": 8},
            }
        )
    )
    config.update(
        allow_paid_dispatch=True,
        compiler_config=str(compiler_path),
        endpoint_registry=str(base / "registry"),
        scratch_dir=str(base / "scratch"),
    )
    config["budget"] = {
        "owner": "offline-native-audit",
        "campaign": "offline-native-audit",
        "ledger": str(base / "ledger"),
        "accounting_verified_at": datetime.now(timezone.utc).isoformat(),
        "dispatch_envelope_usd": 100,
        "policy": {
            "monthly_limit_usd": 1000,
            "reserve_usd": 0,
            "campaign_limit_usd": 1000,
            "daily_limit_usd": 1000,
            "large_job_usd": 1000,
        },
    }
    config_path = base / "generation.yaml"
    config_path.write_text(yaml.safe_dump(config))

    from ..generation.__main__ import main as cli

    with patch.object(
        sys,
        "argv",
        [
            "streaming_rsg",
            "prepare-video",
            "--config",
            str(config_path),
            "--videos",
            str(videos),
            "--out",
            str(base / "prepared"),
        ],
    ):
        with patch("subprocess.run", side_effect=probe_response):
            cli()
    prepared = base / "prepared"
    ids[:] = [row["clip_id"] for row in json.loads((prepared / "inputs.json").read_text())["clips"]]
    assert len(ids) == 2
    return prepared / "generation.json"


def texts(messages):
    """Read text parts without retaining synthetic image blobs in diagnostics."""
    result = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            result.append(content)
        else:
            result.extend(part["text"] for part in content if part.get("type") == "text")
    return result


def probe_response(command, **kwargs):
    """Return the generated 5fps clock at the external ffprobe boundary."""
    assert command[0] == "ffprobe", repr(command)
    probe = {
        "streams": [{"time_base": "1/1000", "nb_frames": "60"}],
        "packets": [{"pts": index * 200, "duration": 200} for index in range(60)],
    }
    probe["packets"][-1]["duration"] = 100
    return subprocess.CompletedProcess(command, 0, stdout=json.dumps(probe))


class Gateway:
    """Return deterministic valid labels at the real HTTP transport boundary."""

    def __init__(self, video, guarded_id, invalid_copyedit=False, registry_record=None, fault=None):
        """Bind deterministic replies and the guarded clip to this test process."""
        self.video = video
        self.registry_record = registry_record
        self.fault = fault
        self.public = bool(
            registry_record and registry_record.get("request_profile") == "openai_reasoning"
        )
        self.invalid_copyedit = invalid_copyedit
        self.guarded_id = guarded_id
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, *args, **kwargs):
        """Return a scripted response at the real pipeline HTTP boundary."""
        import requests

        assert kwargs["headers"]["Authorization"] == "Bearer offline-audit-placeholder"
        body = kwargs["json"]
        public = self.public
        if self.registry_record is not None:
            expected_url = self.registry_record["url"].rstrip("/")
            if public:
                assert expected_url.endswith("/v1")
                expected_url += "/chat/completions"
            else:
                expected_url += "/v1/chat/completions"
            assert args[0] == expected_url
        if public:
            assert body["reasoning_effort"] == self.registry_record["reasoning_effort"]
            assert not {"temperature", "max_tokens", "top_p", "chat_template_kwargs"} & body.keys()
            assert body["max_completion_tokens"] > 0 and body["service_tier"] == "default"
        with self.lock:
            fault, self.fault = self.fault, None
        if fault == "transient":
            response = requests.Response()
            response.status_code = 429
            response._content = json.dumps({"error": {"code": "rate_limit_exceeded"}}).encode()
            with self.lock:
                self.calls.append({"phase": "transient", "status_code": 429})
            return response
        strings = texts(body["messages"])
        joined = "\n".join(strings)
        packets = []
        for string in strings:
            try:
                value = json.loads(string)
            except (ValueError, TypeError):
                continue
            if isinstance(value, dict):
                packets.append(value)
        if "Report driving-relevant visual observations" in joined:
            import re

            known = re.findall(r"observed:\d+(?:#\d+)?", joined)
            result = [
                {
                    "component": known[-1] if known else None,
                    "tracklet": None,
                    "category": "traffic_control",
                    "description": "A red traffic signal is visible ahead.",
                }
            ]
            if self.video:
                ego_ids = re.findall(r"(?<![a-z])ego(?:#\d+)?", joined)
                result.append(
                    {
                        "component": ego_ids[-1] if ego_ids else "ego",
                        "tracklet": None,
                        "category": "ego_vehicle",
                        "description": "The vehicle appears stopped in the front camera.",
                    }
                )
            phase = "observe"
        elif "Observe what ego ACTUALLY DOES" in joined:
            packet = next(p for p in packets if "sensor_samples" in p)
            samples = [
                p
                for p in packet["sensor_samples"]
                if (p["stream"].startswith("camera/") if self.video else p["stream"] == "ego_pose")
            ]
            result = {
                "phases": [
                    {
                        "action": "stop",
                        "kind": "longitudinal",
                        "before_id": None,
                        "after_id": None,
                        "support_ids": [samples[0]["id"], samples[-1]["id"]],
                        "confidence": "supported",
                        "reason": "The measured vehicle remains stationary.",
                    }
                ]
            }
            phase = "physical-actions"
        elif any("refinement_rounds_left" in packet for packet in packets):
            packet = next(p for p in packets if "refinement_rounds_left" in p)
            red = [
                (fid, fact)
                for fid, fact in packet["facts"].items()
                if "red" in json.dumps(fact).lower()
            ]
            assert red, "Observed signal absent from actual compiler snapshot"
            fid, fact = red[0]
            inspect = packet["refinement_rounds_left"] > 0
            guarded = packet["clip_id"] == self.guarded_id
            result = {
                "intervals": [
                    {
                        "interval_start_us": 0,
                        "interval_end_us": packet["clock"]["end_us"],
                        "action": "stop",
                        "cause": "A red signal is visible; its relevance to stopping is uncertain."
                        if guarded
                        else "<NOMINAL>",
                        "cause_refs": [fid] if guarded else [],
                        "outcome_refs": []
                        if guarded
                        else [
                            next(
                                fid
                                for fid, f in packet["facts"].items()
                                if (
                                    "appears stopped" in json.dumps(f).lower()
                                    if self.video
                                    else f["component_id"] == "ego#maneuver"
                                    and f["label"] == "Stop"
                                )
                            )
                        ],
                        "tags": [],
                        "status": "review_needed" if inspect else "validated",
                    }
                ],
                "inspections": [
                    {
                        "scope": "component",
                        "component_id": fact["component_id"],
                        "window_us": [0, 3_000_000],
                        "reason": "Confirm the observed signal",
                        "frames": 3,
                    }
                ]
                if inspect
                else [],
            }
            phase = "interval-refinement" if inspect else "interval-final"
        elif any("committed_action" in packet and "outcome_facts" in packet for packet in packets):
            result = {
                "decision_kind": "nominal",
                "action_evidence": "stop",
                "reason": "Synthetic stationary state.",
            }
            phase = "nominal-review"
        elif any("committed_action" in packet and "facts" in packet for packet in packets):
            result = {"factors": []}
            phase = "causal-export"
        elif packets and "annotations" in packets[-1]:
            rows = packets[-1]["annotations"]
            if isinstance(rows, dict):
                rows = [{"id": key, **value} for key, value in rows.items()]
            if '"fact_ids"' in strings[0] and "supplemental context" in strings[0]:
                result = {
                    "annotations": [
                        {
                            "id": row["id"],
                            "text": (
                                "The red signal is visible, "
                                "but its relevance to stopping is uncertain."
                            ),
                            "fact_ids": [next(iter(row["selected_facts"]))],
                        }
                        for row in rows
                    ]
                }
                phase = "context-writer"
            elif '"supported"' in strings[0] and "proposed context" in strings[0]:
                result = {
                    "annotations": [
                        {"id": row["id"], "supported": True, "readable": True, "issues": []}
                        for row in rows
                    ]
                }
                phase = "context-verifier"
            elif '"faithful"' in strings[0]:
                result = {
                    "annotations": [
                        {"id": row["id"], "faithful": True, "readable": True, "issues": []}
                        for row in rows
                    ]
                }
                phase = "readable-verifier"
            else:
                result = {"annotations": [{"id": row["id"], "text": row["text"]} for row in rows]}
                phase = "readable-writer"
        else:
            raise AssertionError(
                "Unhandled offline phase: "
                + joined[:240]
                + " packet keys="
                + repr([list(p) for p in packets])
            )
        if phase == "readable-writer" and self.invalid_copyedit:
            result = {}
            self.invalid_copyedit = False
        response = requests.Response()
        response.status_code = 200
        if not public:
            response.headers["x-litellm-response-cost"] = "0.001"
        response._content = json.dumps(
            {
                "choices": [{"message": {"content": json.dumps(result)}}],
                "usage": None
                if fault == "malformed"
                else {"prompt_tokens": 10, "completion_tokens": 10},
                "service_tier": "default",
            }
        ).encode()
        with self.lock:
            self.calls.append(
                {
                    "phase": phase,
                    "request_sha256": hashlib.sha256(
                        json.dumps(body, sort_keys=True).encode()
                    ).hexdigest(),
                    "response": result,
                }
            )
        return response


def main(mode: str, out_dir: Path | None = None) -> None:
    """Run real compilation, readable export, exact resume and publication in isolation."""

    def offline_only(event, arguments):
        """Fail on real network access in this isolated test worker."""
        if event == "socket.connect":
            raise AssertionError("Release pipeline tests must stay offline")

    sys.addaudithook(offline_only)
    root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(root))

    class GenerationDependencies(importlib.abc.MetaPathFinder):
        """Reject optional evaluation and development packages in this child process."""

        def find_spec(self, fullname, path=None, target=None):
            """Reject dependencies outside the released generation pipeline."""
            excluded = (
                "alpax",
                "alpaloop",
                "alpamayo",
                f"{__package__.split('.')[0]}.eval",
                f"{__package__.split('.')[0]}.keyframe_rsg",
                f"{__package__.split('.')[0]}.tasks.registry",
                f"{__package__.split('.')[0]}.tasks.keyframe_rsg",
                f"{__package__.split('.')[0]}.tasks.keyframe_rsg_posttraining",
                f"{__package__.split('.')[0]}.tasks.streaming_rsg.evaluation.scorecard",
                f"{__package__.split('.')[0]}.tasks.streaming_rsg.evaluation.parallel",
                f"{__package__.split('.')[0]}.tasks.streaming_rsg.evaluation.judging",
                f"{__package__.split('.')[0]}.tasks.streaming_rsg.evaluation.metrics",
                f"{__package__.split('.')[0]}.tasks.streaming_rsg.evaluation.benchmarks.run",
            )
            if any(fullname == name or fullname.startswith(name + ".") for name in excluded):
                raise ModuleNotFoundError("Outside generation release: " + fullname)
            if fullname.split(".", 1)[0] == __package__.split(".")[0]:
                spec = importlib.machinery.PathFinder.find_spec(fullname, path)
                locations = [] if spec is None else list(spec.submodule_search_locations or ())
                if spec is not None and spec.origin not in {None, "namespace"}:
                    locations.append(spec.origin)
                if not locations or any(
                    not Path(name).resolve().is_relative_to(root) for name in locations
                ):
                    raise ModuleNotFoundError("Outside selected release sources: " + fullname)
                return spec

    sys.meta_path.insert(0, GenerationDependencies())
    import pyarrow.parquet as pq
    import requests
    import yaml

    from ....streaming_rsg.runtime import run as compiler  # noqa: F401
    from ..generation import runtime  # noqa: F401
    from ..generation.__main__ import main as cli
    from ..generation.export.storage import load_clip_records
    from ..generation.interface import run
    from ..generation.parallel import run_parallel

    if mode == "archive-rejected":
        from ....streaming_rsg.env.env import build_env_native

        with patch("requests.post") as transport:
            for metadb, data_dir, sensor in (
                ("metadb.parquet", "archives", None),
                (None, None, None),
                ("metadb.parquet", "archives", "sensors.json"),
            ):
                try:
                    build_env_native(
                        metadb,
                        data_dir,
                        "clip",
                        sensor_manifest=sensor,
                        anchor_mode="discovery",
                        include_external_labels=False,
                    )
                except ValueError as error:
                    assert "sensor_manifest" in str(error) and "development" in str(error)
                else:
                    raise AssertionError("Unsupported archive inputs were accepted")
            transport.assert_not_called()
        print("Public input boundary and dependency isolation passed")
        return

    if out_dir is not None:
        out_dir = out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=False)
    fixtures = (
        tempfile.TemporaryDirectory(prefix="streaming-release-", dir="/tmp")
        if out_dir is None
        else contextlib.nullcontext(str(out_dir))
    )
    with fixtures as directory:
        base = Path(directory)
        os.environ["ALPAMAYO_INFERENCE_STATE_DIR"] = str(base / "inference-state")
        os.environ["RELEASE_SMOKE_API_KEY"] = "offline-audit-placeholder"
        ids = ["native0", "native1"]
        config_path = (video_fixture if mode == "video" else native_fixture)(root, base, ids)
        config = yaml.safe_load(config_path.read_text())
        fixture_endpoint = json.loads((base / "registry/r0.json").read_text())
        if mode == "native":
            incomplete = deepcopy(config)
            incomplete.update(
                allow_paid_dispatch=False,
                compiler_config=None,
                input_manifest=None,
                endpoint_registry=None,
            )
            incomplete["budget"] = {}
            unfilled = base / "unfilled.yaml"
            unfilled.write_text(yaml.safe_dump(incomplete))
            rejected = base / "unfilled-run"
            with patch.object(requests, "post") as transport:
                for parallel in (False, True):
                    try:
                        if parallel:
                            run_parallel(incomplete, rejected, 2)
                        else:
                            run(unfilled, out_dir=rejected)
                    except ValueError as error:
                        for name in (
                            "compiler_config",
                            "endpoint_registry",
                            "input_manifest",
                            "allow_paid_dispatch",
                            "budget.ledger",
                        ):
                            assert name in str(error)
                    else:
                        raise AssertionError("Unfilled template reached generation")
                    assert not rejected.exists()
                transport.assert_not_called()
        if mode == "legacy":
            for field in ("request_profile", "reasoning_effort", "pricing"):
                fixture_endpoint.pop(field, None)
            fixture_endpoint["url"] = "https://offline.invalid"
            (base / "registry/r0.json").write_text(json.dumps(fixture_endpoint))
        if mode == "transient":
            config["transient_retries"] = 1
            config_path.write_text(yaml.safe_dump(config))
        if mode == "native":
            from ....streaming_rsg.agent.orchestrator import Orchestrator
            from ....streaming_rsg.runtime.config import RunConfig

            RunConfig.model_validate(
                {
                    "playground": {"task": "none", "env": {"reference": "model"}},
                    "infra": {"grounding_registry": ""},
                }
            )
            tools = Orchestrator(env=None)
            tools.tools["fixture"] = lambda graph, argument: argument
            value, record_id = tools.call_registered(
                "fixture", "fixture", "recorded", None, None, None
            )
            assert value == "recorded" and tools.records[-1]["id"] == record_id
            for unsupported in (
                {"playground": {"task": "keyframe_decision"}},
                {
                    "playground": {"task": "none", "env": {"reference": "model"}},
                    "infra": {"grounding_registry": str(base / "grounding")},
                },
            ):
                try:
                    RunConfig.model_validate(unsupported)
                except ValueError as error:
                    assert "not included in this package" in str(error)
                else:
                    raise AssertionError("Unavailable public compiler feature was accepted")
            try:
                run_parallel(config, base / "unavailable-refresh", 2, refresh_accounting=True)
            except ValueError as error:
                assert "provider refresh is unavailable" in str(error)
            else:
                raise AssertionError("Private provider refresh was accepted")

            from ..generation.export.coc_copyedit import copyedit_interval_cocs

            actions = {
                "clock": "At 5.2 s, brake within 2 s of the cut-in.",
                "duration": "The ego waits for about 6 s at the crosswalk.",
                "window": "The signal is observed red at 1, 2, and 3 s; later within "
                "4–6 s and 8 s, and again near 9.2 s.",
            }
            editor_requests = []

            def public_copyeditor(messages):
                """Keep real durations and repair a retained clock using rejection feedback."""
                packet = json.loads(messages[1]["content"])
                if '"faithful":boolean' in messages[0]["content"]:
                    assert all("retry_feedback" not in row for row in packet["annotations"])
                    rows = [
                        {"id": row["id"], "faithful": True, "readable": True, "issues": []}
                        for row in packet["annotations"]
                    ]
                else:
                    editor_requests.append(packet)
                    if len(editor_requests) == 2:
                        assert [row["id"] for row in packet["annotations"]] == ["clock"]
                        feedback = packet["annotations"][0]["retry_feedback"]
                        assert feedback["remaining_clock_references"] == ["At 5.2 s"]
                    rows = [
                        {
                            "id": row["id"],
                            "text": (
                                "Brake within 2 s of the cut-in."
                                if len(editor_requests) == 2
                                else "The signal is observed red at early observation points, "
                                "in later windows, and again later."
                                if row["id"] == "window"
                                else row["text"]
                            ),
                        }
                        for row in packet["annotations"]
                    ]
                return json.dumps({"annotations": rows})

            edited = copyedit_interval_cocs(
                {
                    "bundles": [
                        {
                            "interval_id": key,
                            "coc": action,
                            "coc_export": {
                                "public_action": {"rendered": action},
                                "factors": [],
                            },
                        }
                        for key, action in actions.items()
                    ]
                },
                "verified_readable_account_v1",
                public_copyeditor,
                omit_clock_times=True,
            )
            assert [row["coc"] for row in edited["bundles"]] == [
                "Brake within 2 s of the cut-in.",
                actions["duration"],
                "The signal is observed red at early observation points, in later windows, "
                "and again later.",
            ]
            assert len(editor_requests) == 2 and editor_requests[0] != editor_requests[1]

            invalid = deepcopy(config)
            del invalid["budget"]
            invalid["policy"]["retain_refinement_context"] = "true"
            invalid_path = base / "invalid-policy.yaml"
            invalid_path.write_text(yaml.safe_dump(invalid))
            rejected = base / "invalid-policy-run"
            with (
                patch.object(
                    sys,
                    "argv",
                    ["streaming_rsg", "run", "--config", str(invalid_path), "--out", str(rejected)],
                ),
                patch(
                    f"{__package__.split('.')[0]}.tasks.streaming_rsg.generation.interface.generation_allocation",
                    side_effect=AssertionError("invalid policy reached budget allocation"),
                ),
                patch.object(
                    requests, "post", side_effect=AssertionError("invalid policy sent HTTP")
                ),
            ):
                try:
                    cli()
                except ValueError as error:
                    assert "retain_refinement_context must be Boolean" in str(error)
                else:
                    raise AssertionError("Invalid generation policy was accepted")
            assert not rejected.exists()
            for feature, mode_override, policy_override in (
                ("boundary replay", "boundary_replay", {}),
                ("constraint replay", "constraint_replay", {}),
                ("boundary localization", None, {"boundary_localization": {"enabled": True}}),
                ("constraint review", None, {"constraint_review": True}),
            ):
                unavailable = deepcopy(config)
                del unavailable["budget"]
                unavailable["policy"].update(policy_override)
                if mode_override is not None:
                    unavailable["mode"] = mode_override
                path = base / (feature.replace(" ", "-") + ".yaml")
                path.write_text(yaml.safe_dump(unavailable))
                rejected = base / (feature.replace(" ", "-") + "-run")
                with (
                    patch(
                        f"{__package__.split('.')[0]}.tasks.streaming_rsg.generation.interface.generation_allocation",
                        side_effect=AssertionError("unavailable feature reached allocation"),
                    ),
                    patch.object(
                        requests,
                        "post",
                        side_effect=AssertionError("unavailable feature sent HTTP"),
                    ),
                ):
                    try:
                        run(path, out_dir=rejected)
                    except ValueError as error:
                        assert feature + " is not included" in str(error)
                    else:
                        raise AssertionError("Unavailable feature was accepted: " + feature)
                assert not rejected.exists()
            compiler_path = Path(config["compiler_config"])
            compiler_text = compiler_path.read_text()
            compiler_config = yaml.safe_load(compiler_text)
            compiler_config["policy"].setdefault("focus", {})["mode"] = "agentic"
            try:
                compiler_path.write_text(yaml.safe_dump(compiler_config))
                rejected = base / "unavailable-planner"
                with patch.object(
                    requests, "post", side_effect=AssertionError("unavailable planner sent HTTP")
                ):
                    try:
                        run(config_path, out_dir=rejected)
                    except ValueError as error:
                        assert "agentic focus is not included" in str(error)
                    else:
                        raise AssertionError("Unavailable planner was accepted")
                assert not rejected.exists()
            finally:
                compiler_path.write_text(compiler_text)
        if mode == "native" and fixture_endpoint.get("request_profile") == "openai_reasoning":
            variants = [
                (
                    "null-prices",
                    {
                        "pricing": {
                            "input_usd_per_million_tokens": None,
                            "output_usd_per_million_tokens": None,
                        }
                    },
                    "token pricing",
                ),
                ("unknown-profile", {"request_profile": "unknown"}, "unknown request profile"),
                ("missing-effort", {"reasoning_effort": None}, "reasoning effort"),
                ("unsupported-effort", {"reasoning_effort": "none"}, "reasoning effort"),
                ("missing-key", {"api_key_env": "UNSET_RELEASE_PROFILE_KEY"}, "not set"),
            ]
            assert "UNSET_RELEASE_PROFILE_KEY" not in os.environ
            for name, overrides, message in variants:
                try:
                    (base / "registry/r0.json").write_text(
                        json.dumps({**fixture_endpoint, **overrides})
                    )
                    for parallel in (False, True):
                        rejected = base / (name + str(parallel))
                        with patch.object(
                            requests,
                            "post",
                            side_effect=AssertionError("invalid registry sent HTTP"),
                        ):
                            try:
                                if parallel:
                                    run_parallel(config, rejected, 2)
                                else:
                                    run(config_path, out_dir=rejected)
                            except ValueError as error:
                                assert message in str(error)
                            else:
                                raise AssertionError("Invalid public registry was accepted")
                        assert not rejected.exists()
                        assert not Path(config["budget"]["ledger"]).exists()
                finally:
                    (base / "registry/r0.json").write_text(json.dumps(fixture_endpoint))
        gateway = Gateway(
            video=mode == "video",
            guarded_id=ids[0],
            registry_record=fixture_endpoint,
            fault=mode if mode in {"transient", "malformed"} else None,
        )
        output = base / "run"
        with (
            patch.object(requests, "post", side_effect=gateway),
            patch("subprocess.run", side_effect=probe_response)
            if mode == "video"
            else contextlib.nullcontext(),
        ):
            if mode == "malformed":
                from ....infra.budget import CostLedger
                from ..generation.cost import AllocationGateError

                try:
                    run(config_path, out_dir=output)
                except AllocationGateError:
                    pass
                else:
                    raise AssertionError("A 200 without usage authorized further dispatch")
                before = (output / "usage.jsonl").read_bytes()
                ledger = CostLedger(Path(config["budget"]["ledger"]))
                pending = [
                    row
                    for row in ledger.records()
                    if row.provider == "configured_registry" and row.amount() is None
                ]
                assert len(pending) == 1
                ledger.reconcile(pending[0], estimated_usd=None, confirmed_usd=0.001, requests=1)
                run(config_path, out_dir=output, resume=True)
                assert (output / "usage.jsonl").read_bytes().startswith(before)
                assert (output / "clips/manifest.json").exists() and (
                    output / "cost_summary.json"
                ).exists()
            elif mode == "parallel":
                assert run_parallel(config, output, 2)["failures"] == {}
            else:
                with patch.object(
                    sys,
                    "argv",
                    ["streaming_rsg", "run", "--config", str(config_path), "--out", str(output)],
                ):
                    cli()
        from ....infra.budget import CostLedger

        rows = [
            json.loads(line)
            for path in output.rglob("usage.jsonl")
            for line in path.read_text().splitlines()
        ]
        summary = json.loads((output / "cost_summary.json").read_text())
        receipts = CostLedger(Path(config["budget"]["ledger"])).records()
        assert len(receipts) == len(rows)
        if fixture_endpoint.get("request_profile") == "openai_reasoning":
            priced = [row for row in rows if isinstance(row["usage"], dict)]
            assert all(row["cost_usd"] is None for row in rows)
            assert all(row["cost_estimate_policy"] == "configured_token_rates_v1" for row in priced)
            assert all(row["estimated_cost_usd"] == 0.0001 for row in priced)
            expected = len(priced) * 0.0001
            assert math.isclose(summary["estimated_cost_usd"], expected)
            assert math.isclose(
                sum(row.estimated_usd or 0 for row in receipts),
                expected + (1 if mode == "transient" else 0),
            )
            assert math.isclose(
                sum(row.confirmed_usd or 0 for row in receipts), 0.001 if mode == "malformed" else 0
            )
            assert summary["total_cost_usd"] is None
            if mode == "transient":
                failed = [row for row in rows if row["status_code"] == 429]
                assert len(failed) == 1 and "cost_estimate_policy" not in failed[0]
        else:
            assert all(row["cost_usd"] == 0.001 for row in rows)
            assert math.isclose(summary["total_cost_usd"], len(rows) * 0.001)
        assert math.isclose(summary["budget_cost_usd"], sum(row.amount() for row in receipts))
        records = load_clip_records(output / "clips")
        assert set(records) == set(ids)
        assert all(
            row["status"] in {"complete", "review_needed"} and not row.get("error_type")
            for row in records.values()
        ), {
            cid: {key: row.get(key) for key in ("status", "error_type", "error_trace")}
            for cid, row in records.items()
        }
        assert records[ids[0]]["status"] == "review_needed"
        assert records[ids[1]]["status"] == "complete"
        assert all(row["result"]["bundles"] for row in records.values())
        assert {call["phase"] for call in gateway.calls} >= {
            "observe",
            "physical-actions",
            "interval-refinement",
            "interval-final",
            "readable-writer",
            "readable-verifier",
        }
        if mode != "video":
            assert records[ids[0]]["result"]["bundles"][0]["coc_support_context"]["accepted"]
            assert {call["phase"] for call in gateway.calls} >= {
                "context-writer",
                "context-verifier",
            }
        else:
            assert all(not row["result"]["clock_verified"] for row in records.values())
            assert all(row["result"]["samples"] for row in records.values())
            assert all(
                sample["trajectory_status"] == "unavailable_video_only"
                for row in records.values()
                for sample in row["result"]["samples"]
            )
        assert all(
            "<NOMINAL>" not in bundle["coc"]
            for row in records.values()
            for bundle in row["result"]["bundles"]
        )
        with patch.object(requests, "post", side_effect=AssertionError("resume sent HTTP")):
            if mode == "parallel":
                assert run_parallel(config, output, 2)["failures"] == {}
                operational = deepcopy(config)
                operational["scratch_dir"] = str(base / "alternate-scratch")
                operational["max_resume_attempts"] = config.get("max_resume_attempts", 1) + 1
                assert run_parallel(operational, output, 2)["failures"] == {}
            else:
                run(config_path, out_dir=output, resume=True)
        assert load_clip_records(output / "clips") == records
        if mode == "native" and fixture_endpoint.get("request_profile") == "openai_reasoning":
            registry_file = base / "registry/r0.json"
            usage_before = (output / "usage.jsonl").read_bytes()
            ledger_before = CostLedger(Path(config["budget"]["ledger"])).records()
            try:
                for index, override in enumerate(
                    (
                        {"reasoning_effort": "high"},
                        {"model": "changed-model"},
                        {"request_profile": None},
                    )
                ):
                    registry_file.write_text(json.dumps({**fixture_endpoint, **override}))
                    with patch.object(
                        requests,
                        "post",
                        side_effect=AssertionError("semantic recovery drift sent HTTP"),
                    ):
                        try:
                            run(
                                config_path,
                                out_dir=base / ("semantic-drift-" + str(index)),
                                recovery_source=output,
                            )
                        except ValueError as error:
                            assert "endpoint request semantics changed" in str(error)
                        else:
                            raise AssertionError(
                                "Changed endpoint semantics relabeled recovered records"
                            )
                registry_file.write_text(
                    json.dumps(
                        {
                            **fixture_endpoint,
                            "pricing": {
                                "input_usd_per_million_tokens": 4,
                                "output_usd_per_million_tokens": 16,
                            },
                        }
                    )
                )
                recovered = base / "repriced-recovery"
                with patch.object(
                    requests, "post", side_effect=AssertionError("price-only recovery sent HTTP")
                ):
                    run(config_path, out_dir=recovered, recovery_source=output)
                recovered_records = load_clip_records(recovered / "clips")
                assert all(
                    recovered_records[cid]["result"] == records[cid]["result"] for cid in ids
                )
                assert (output / "usage.jsonl").read_bytes() == usage_before
                assert CostLedger(Path(config["budget"]["ledger"])).records() == ledger_before
            finally:
                registry_file.write_text(json.dumps(fixture_endpoint))
        for cid in ids:
            publication = base / ("public-" + cid)
            with patch.object(
                sys,
                "argv",
                [
                    "streaming_rsg",
                    "export-public",
                    "--publication",
                    str(output / "clips"),
                    "--clip",
                    cid,
                    "--out",
                    str(publication),
                ],
            ):
                cli()
                rows = pq.read_table(publication / "labels_cot_intervals.parquet").to_pylist()
                assert len(rows) == 1 and rows[0]["clipid"] == cid and rows[0]["cot"]
                assert rows[0]["start_us"] == 0
                assert rows[0]["end_us"] == (11_900_000 if mode == "video" else 4_000_000)
                assert (publication / "labels_cot_keyframes.csv").exists()
            annotation_dir = base / ("annotation-" + cid)
            with patch.object(
                sys,
                "argv",
                [
                    "streaming_rsg",
                    "export-annotation",
                    "--publication",
                    str(output / "clips"),
                    "--clip",
                    cid,
                    "--out",
                    str(annotation_dir),
                    *(["--diagnostics"] if cid == ids[0] else []),
                ],
            ):
                cli()
            annotation = json.loads((annotation_dir / "annotation.json").read_text())
            original = records[cid]
            assert annotation["graph_snapshot"] == original["snapshot"]
            assert annotation["clock"] == original["result"]["timeline"]["clock"]
            assert annotation["generation_status"] == "complete"
            assert "status" not in annotation
            assert not {"training_ready", "training_eligible"} & annotation.keys()
            for exported, interval, bundle in zip(
                annotation["intervals"],
                original["result"]["timeline"]["intervals"],
                original["result"]["bundles"],
                strict=True,
            ):
                assert {k: v for k, v in exported.items() if k not in {"coc", "evidence_view"}} == {
                    k: v for k, v in interval.items() if k != "status"
                }
                assert exported["coc"] == bundle["coc"]
                assert exported["evidence_view"] == bundle["view"]
            diagnostics = annotation_dir / "diagnostics.json"
            if cid == ids[0]:
                assert json.loads(diagnostics.read_text()) == original
            else:
                assert not diagnostics.exists()
        from ..generation.export.publication import write_annotation, write_public_store

        normalized = deepcopy(records[ids[0]]["result"])
        normalized["timeline"]["intervals"][0]["tags"] = [
            " nav_missing_not_default ",
            " interesting ",
        ]
        raw = deepcopy(normalized)
        write_public_store(normalized, base / "normalized-tags")
        assert (
            pq.read_table(base / "normalized-tags/labels_cot_intervals.parquet")[
                "tags"
            ].to_pylist()[0]
            == "nav_not_default,interesting"
        )
        assert normalized == raw

        for status in ("failed", "not_run"):
            rejected = deepcopy(records[ids[0]])
            rejected["status"] = rejected["result"]["status"] = status
            destination = base / ("rejected-" + status)
            for writer, item in (
                (write_annotation, rejected),
                (write_public_store, rejected["result"]),
            ):
                try:
                    writer(item, destination)
                except ValueError as error:
                    assert "unsuccessful" in str(error)
                else:
                    raise AssertionError("Unsuccessful generation was exported")
                assert not destination.exists()
        malformed = deepcopy(records[ids[0]])
        malformed["result"]["timeline"]["intervals"][0]["interval_end_us"] -= 1
        destination = base / "malformed-annotation"
        try:
            write_annotation(malformed, destination)
        except ValueError:
            assert not destination.exists()
        else:
            raise AssertionError("Truncated annotation was exported")
        assert load_clip_records(output / "clips") == records
        if mode == "video":
            from ....streaming_rsg.env.video import build_env_video_manifest
            from ....streaming_rsg.runtime.config import RunConfig

            compiler_config = json.loads((base / "prepared/compiler.json").read_text())
            manifest_path = Path(compiler_config["playground"]["env"]["video_manifest"])
            video_path = Path(json.loads(manifest_path.read_text())["clips"][ids[0]]["video"])
            with video_path.open("ab") as handle:
                handle.write(b"changed content")
            try:
                build_env_video_manifest(
                    manifest_path,
                    ids[0],
                    base / "cache",
                    RunConfig.model_validate(compiler_config).playground.env,
                )
            except ValueError as error:
                assert "content changed" in str(error)
            else:
                raise AssertionError("Changed MP4 escaped content binding")
        if mode == "native":
            from ....infra.budget import CostLedger
            from ..generation.parallel import main as parallel_cli

            receipts = CostLedger(Path(config["budget"]["ledger"])).records()
            local_rows = (output / "usage.jsonl").read_text().splitlines()
            assert len(receipts) == len(local_rows)
            assert sum(row.amount() for row in receipts) > 0
            videos = [base / "data" / cid / "camera_front_wide_120fov.mp4" for cid in ids]
            for path in videos:
                path.rename(path.with_suffix(".missing"))
            try:
                for parallel in (False, True):
                    failed = base / ("missing-parallel" if parallel else "missing-serial")
                    argv = (
                        [
                            "streaming_rsg",
                            "--config",
                            str(config_path),
                            "--out",
                            str(failed),
                            "--workers",
                            "2",
                        ]
                        if parallel
                        else [
                            "streaming_rsg",
                            "run",
                            "--config",
                            str(config_path),
                            "--out",
                            str(failed),
                        ]
                    )
                    with (
                        patch.object(sys, "argv", argv),
                        patch.object(
                            requests, "post", side_effect=AssertionError("missing input sent HTTP")
                        ),
                    ):
                        try:
                            (parallel_cli if parallel else cli)()
                        except SystemExit as error:
                            assert error.code == 1
                        else:
                            raise AssertionError("Failed clips returned CLI success")
                    assert set(load_clip_records(failed / "clips")) == set(ids)
                    assert {
                        row["status"] for row in load_clip_records(failed / "clips").values()
                    } == {"failed"}
            finally:
                for path in videos:
                    path.with_suffix(".missing").rename(path)

            review_inputs = base / "review-inputs.json"
            population = json.loads(Path(config["input_manifest"]).read_text())
            review_inputs.write_text(json.dumps({"clips": population["clips"][:1]}))
            review_config = {**config, "input_manifest": str(review_inputs)}
            review_path = base / "review-only.yaml"
            review_path.write_text(yaml.safe_dump(review_config))
            review_run = base / "review-only"
            for resume in (False, True):
                argv = [
                    "streaming_rsg",
                    "run",
                    "--config",
                    str(review_path),
                    "--out",
                    str(review_run),
                ]
                if resume:
                    argv.append("--resume")
                with (
                    patch.object(sys, "argv", argv),
                    patch.object(
                        requests,
                        "post",
                        side_effect=AssertionError("cached review sent HTTP")
                        if resume
                        else Gateway(
                            video=False, guarded_id=ids[0], registry_record=fixture_endpoint
                        ),
                    ),
                ):
                    cli()
            assert load_clip_records(review_run / "clips")[ids[0]]["status"] == "review_needed"

            invalid = base / "invalid-copyedit"
            bad_gateway = Gateway(
                video=False,
                guarded_id=ids[0],
                invalid_copyedit=True,
                registry_record=fixture_endpoint,
            )
            with (
                patch.object(
                    sys,
                    "argv",
                    ["streaming_rsg", "run", "--config", str(config_path), "--out", str(invalid)],
                ),
                patch.object(requests, "post", side_effect=bad_gateway),
            ):
                try:
                    cli()
                except SystemExit as error:
                    assert error.code == 1
                else:
                    raise AssertionError("Invalid copyedit returned CLI success")
            invalid_records = load_clip_records(invalid / "clips")
            assert invalid_records[ids[0]]["status"] == "failed"
            assert (
                invalid_records[ids[1]]["status"] == "complete"
            )  # One malformed reply cannot invalidate another clip's checked prose.
            for row in invalid_records.values():
                if row["status"] != "failed":
                    continue
                assert any(call["response"] == "{}" for call in row["copyedit_calls"])
                assert "offline-audit-placeholder" not in json.dumps(row["copyedit_calls"])
                assert "data:image" not in json.dumps(row["copyedit_calls"])
            assert not list((base / "scratch").glob("alparsg-coc-clip-*"))

            from ....infra.artifacts import CompactArtifactJournal
            from ..generation import interface
            from ..generation.cost import AllocationGateError, RequestAccounting

            interrupted = invalid
            record = CompactArtifactJournal.record

            def interrupt_archived_replay(journal, clip_id, payload):
                """Interrupt after archiving the old manifest and before restoring its cache."""
                if journal.root == interrupted / "clips" and clip_id == ids[1]:
                    raise KeyboardInterrupt
                record(journal, clip_id, payload)

            with (
                patch.object(CompactArtifactJournal, "record", new=interrupt_archived_replay),
                patch.object(
                    requests, "post", side_effect=AssertionError("archive replay sent HTTP")
                ),
            ):
                try:
                    run(config_path, out_dir=interrupted, resume=True)
                except KeyboardInterrupt:
                    pass
                else:
                    raise AssertionError("Archive-replay interruption was swallowed")
            assert (interrupted / "prior_publications/attempt_000/clips/manifest.json").is_file()
            cached_lines = None
            for _ in range(2):
                with (
                    patch.object(
                        RequestAccounting,
                        "before_attempt",
                        side_effect=AllocationGateError("synthetic pause before physical dispatch"),
                    ),
                    patch.object(
                        requests, "post", side_effect=AssertionError("paused resume sent HTTP")
                    ),
                ):
                    try:
                        run(config_path, out_dir=interrupted, resume=True)
                    except AllocationGateError:
                        pass
                    else:
                        raise AssertionError("Paused resume unexpectedly dispatched")
                lines = [
                    line
                    for line in (interrupted / "clips/.resume.jsonl").read_text().splitlines()
                    if json.loads(line)["record_id"] == ids[1]
                ]
                assert len(lines) == 1, "Cached success was replayed into its own live journal"
                assert cached_lines is None or lines == cached_lines
                cached_lines = lines
                assert load_clip_records(interrupted / "clips")[ids[1]] == invalid_records[ids[1]]
            write = interface.write_json_atomic

            def interrupt_metadata(path, payload):
                """Interrupt after durable labels and before final run metadata is complete."""
                if path == interrupted / "cost_summary.json":
                    raise KeyboardInterrupt
                write(path, payload)

            retry_gateway = Gateway(False, ids[0], registry_record=fixture_endpoint)

            def retry_failed_only(*args, **kwargs):
                """A cached successful clip cannot generate another paid request."""
                assert ids[1] not in "\n".join(texts(kwargs["json"]["messages"]))
                return retry_gateway(*args, **kwargs)

            with (
                patch.object(interface, "write_json_atomic", side_effect=interrupt_metadata),
                patch.object(requests, "post", side_effect=retry_failed_only),
            ):
                try:
                    run(config_path, out_dir=interrupted, resume=True)
                except KeyboardInterrupt:
                    pass
                else:
                    raise AssertionError("Finalization interruption was swallowed")
            durable = load_clip_records(interrupted / "clips")
            usage = (interrupted / "usage.jsonl").read_bytes()
            charged = len(CostLedger(Path(config["budget"]["ledger"])).records())
            assert not (interrupted / "storage_summary.json").exists()
            assert not list((base / "scratch").glob("alparsg-coc-clip-*"))
            with patch.object(
                requests, "post", side_effect=AssertionError("finalization resume sent HTTP")
            ):
                run(config_path, out_dir=interrupted, resume=True)
                run(config_path, out_dir=interrupted, resume=True)
            assert load_clip_records(interrupted / "clips") == durable
            assert (interrupted / "usage.jsonl").read_bytes() == usage
            assert (interrupted / "cost_summary.json").is_file()
            assert (interrupted / "storage_summary.json").is_file()
            assert (
                json.loads((interrupted / "cost_summary.json").read_text())
                == json.loads((interrupted / "clips/manifest.json").read_text())["metadata"][
                    "cost_summary"
                ]
            )
            assert len(CostLedger(Path(config["budget"]["ledger"])).records()) == charged
            coverage = [
                json.loads(line) for line in (interrupted / "clips.jsonl").read_text().splitlines()
            ]
            assert len(coverage) == len(ids) + 1
            assert coverage[0] == {"clips_total": len(ids)}
            assert {row["clip_id"] for row in coverage[1:]} == set(ids)

            capped = deepcopy(config)
            capped["budget"]["policy"]["campaign_limit_usd"] = (
                sum(row.amount() for row in CostLedger(Path(config["budget"]["ledger"])).records())
                / 2
                + capped["budget"]["dispatch_envelope_usd"]
            )
            capped_path = base / "capped.yaml"
            capped_path.write_text(yaml.safe_dump(capped))
            with patch.object(
                requests, "post", side_effect=AssertionError("campaign cap sent HTTP")
            ):
                try:
                    run(capped_path, out_dir=base / "capped")
                except RuntimeError as error:
                    assert "budget" in str(error)
                else:
                    raise AssertionError("A new output directory bypassed shared campaign spend")
            assert not (base / "capped").exists()

        if mode == "native":
            from ....infra.budget import BudgetPolicy, CostLedger, reserve_cost
            from ..generation import interface
            from ..generation.configuration import generation_identity
            from ..generation.cost import AllocationGateError

            # A genuinely empty startup journal resumes from its existing identity.
            startup = base / "startup"
            write = interface.write_json_atomic

            def interrupt_identity(path, payload):
                """Interrupt after identity durability and before the first clip append."""
                write(path, payload)
                if path == startup / "generation_identity.json":
                    (startup / "clips").mkdir()
                    raise KeyboardInterrupt

            with patch.object(interface, "write_json_atomic", side_effect=interrupt_identity):
                try:
                    run(config_path, out_dir=startup)
                except KeyboardInterrupt:
                    pass
                else:
                    raise AssertionError("Startup interruption was swallowed")
            assert not (startup / "clips/.resume.jsonl").exists()
            with patch.object(requests, "post", side_effect=gateway):
                assert run(config_path, out_dir=startup, resume=True)["success"]

            for field in ("clock", "sample_anchors_us"):
                invalid_manifest = json.loads(Path(config["input_manifest"]).read_text())
                for item in invalid_manifest["clips"]:
                    if field == "clock":
                        item["clock"]["end_us"] += 1
                    else:
                        item["sample_anchors_us"] = [2_000_000]
                invalid_inputs = base / (field + ".json")
                invalid_inputs.write_text(json.dumps(invalid_manifest))
                invalid_config = {**config, "input_manifest": str(invalid_inputs)}
                invalid_path = base / (field + ".yaml")
                invalid_path.write_text(yaml.safe_dump(invalid_config))
                with patch.object(
                    requests, "post", side_effect=AssertionError("invalid inputs sent HTTP")
                ):
                    assert not run(invalid_path, out_dir=base / ("invalid-" + field))["success"]

            manifest = json.loads(Path(config["input_manifest"]).read_text())
            identity = generation_identity(config, manifest, {})
            helper = root / __package__.split(".")[0] / "infra/middleware/parsing.py"
            original = helper.read_bytes()
            try:
                helper.write_bytes(original + b"\n# offline source identity mutation\n")
                assert generation_identity(config, manifest, {}) != identity
                with patch.object(
                    requests, "post", side_effect=AssertionError("identity drift sent HTTP")
                ):
                    try:
                        run(config_path, out_dir=output, resume=True)
                    except ValueError as error:
                        assert "generation identity drift (code)" in str(error)
                    else:
                        raise AssertionError("Changed parser source reused a cached publication")
            finally:
                helper.write_bytes(original)

            # Paid compiler records survive a downstream interpreter failure.
            failure = base / "compiler-context-failure"
            with (
                patch.object(requests, "post", side_effect=gateway),
                patch.object(runtime, "interpret_clip", side_effect=ValueError("offline failure")),
            ):
                assert not run(config_path, out_dir=failure)["success"]
            failed_records = load_clip_records(failure / "clips")
            assert all(row["context"] for row in failed_records.values())

            # A competing reservation gates the controller without finalizing failed clips.
            competing = deepcopy(config)
            competing["budget"]["policy"]["campaign_limit_usd"] = 150
            competing_path = base / "competing.yaml"
            competing_path.write_text(yaml.safe_dump(competing))
            for recover in (False, True):
                pending = base / ("pending-recovery" if recover else "pending-resume")
                with reserve_cost(
                    CostLedger(Path(config["budget"]["ledger"])),
                    policy=BudgetPolicy(**config["budget"]["policy"]),
                    owner=config["budget"]["owner"],
                    campaign=config["budget"]["campaign"],
                    reservation_id="offline-competing-" + str(recover),
                    estimated_usd=100,
                ):
                    with patch.object(
                        requests, "post", side_effect=AssertionError("reservation gate sent HTTP")
                    ):
                        try:
                            run(competing_path, out_dir=pending)
                        except AllocationGateError:
                            pass
                        else:
                            raise AssertionError("Concurrent reservation was ignored")
                assert not (pending / "clips/manifest.json").exists()
                assert all(
                    r["status"] == "not_run" for r in load_clip_records(pending / "clips").values()
                )
                with patch.object(requests, "post", side_effect=gateway):
                    if recover:
                        assert run(
                            competing_path,
                            out_dir=base / "recovered-partial",
                            recovery_source=pending,
                        )["success"]
                    else:
                        assert run(competing_path, out_dir=pending, resume=True)["success"]

        from ....infra.model import ChatClient

        received = []
        blocked = ChatClient(Path(config["endpoint_registry"]), on_attempt=received.append)
        response = requests.Response()
        response.status_code = 429
        response._content = b'{"error":{"code":"insufficient_quota"}}'
        with patch.object(requests, "post", return_value=response) as transport:
            for _ in range(2):
                try:
                    blocked.complete([{"role": "user", "content": "offline quota check"}])
                except RuntimeError as error:
                    assert "exhausted" in str(error)
                else:
                    raise AssertionError("Exhausted hosted key was accepted")
            assert transport.call_count == 1
        assert len(received) == 1
        assert received[0]["key_fingerprint"].startswith("sha256:")

        if out_dir is not None:
            (base / "offline_example.json").write_text(
                json.dumps(
                    {
                        "synthetic": True,
                        "mode": mode,
                        "requests": gateway.calls,
                        "annotations": [f"annotation-{cid}/annotation.json" for cid in ids],
                        "cost_summary": "run/cost_summary.json",
                        "note": (
                            "Scripted HTTP responses and prices; "
                            "not measured model quality or provider cost."
                        ),
                    },
                    indent=2,
                )
                + "\n"
            )
        assert not list(output.rglob("judgments.json"))
        assert not list(output.rglob("scorecard.json"))
        print(
            f"{mode}: default recipe, real media/Parquet, exact resume, "
            "guarded public export and dependency isolation passed"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=[
            "native",
            "parallel",
            "video",
            "legacy",
            "transient",
            "malformed",
            "archive-rejected",
        ],
        help="synthetic workflow to exercise without real HTTP",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="retain synthetic inputs, annotations and request hashes in a new directory",
    )
    args = parser.parse_args()
    main(args.mode, args.out)
