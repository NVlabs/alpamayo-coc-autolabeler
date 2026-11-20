"""The env: the per-clip playground contract that supersedes the RSG.

One immutable env instance per clip run. It binds the clip's raw data through native loaders
(camera media, trajectory, labels, map) and exposes it only through mediated reads, so an op
cannot observe outside the playground by construction. Every read registers **source records**
(stable ``env:*`` ids, classed by the source manifest) in the env catalog; components cite
those ids in their provenance, which therefore terminates at the env. The contract -- anchors,
sampling bounds, access policy, budgets, source manifest -- is hashed into the env id, making
"same playground" a checkable property.

``env_from_artifact`` remains as a schema/executor test fixture over a reference reasoning_agent
artifact; it is not the data path.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...common.meta_actions import SCHEMA as META_ACTION_SCHEMA, validate_meta_actions
from ...common.navigation import validate_navigation
from ..data_structure.rsg import SourceClass
from .loaders import (
    ClipData,
    actors_at,
    crosswalks_at,
    decode_frame,
    ego_kinematics_at,
    extract_video_segment,
    lanes_at,
)


@dataclass(frozen=True)
class Anchor:
    """One temporal anchor inside the clip, with its observation horizons."""

    anchor_us: int
    history_us: int
    future_us: int
    trigger: str

    @property
    def window_us(self) -> tuple[int, int]:
        """Inclusive observation window spanned by the horizons."""
        return (self.anchor_us - self.history_us, self.anchor_us + self.future_us)


@dataclass(frozen=True)
class SamplingBounds:
    """Upper bounds on what may be sampled from the clip media."""

    max_fps: float
    max_frames: int
    cameras: tuple[str, ...]
    image_max_height_px: int
    image_overview_height_px: int | None = None

    def __post_init__(self) -> None:
        """Keep overview reads within the immutable detail-resolution ceiling."""
        if self.image_overview_height_px is not None and (
            type(self.image_overview_height_px) is not int
            or not 0 < self.image_overview_height_px < self.image_max_height_px
        ):
            raise ValueError("overview height must be positive and below the image ceiling")


@dataclass(frozen=True)
class AccessPolicy:
    """What the agent role may observe; the eval role always sees the full window."""

    agent_gt_labels: bool = True
    agent_future_ego: bool = False


@dataclass(frozen=True)
class RunBudget:
    """User-set execution bounds; in-run loops (including re-planning) terminate on these."""

    max_ops: int
    wall_clock_s: float


@dataclass(frozen=True)
class Observation:
    """One timestamped record from the clip's observation stream (e.g. an ego reading)."""

    timestamp_us: int
    data: dict[str, Any]


@dataclass(frozen=True)
class SourceRecord:
    """One env source record: the citable unit provenance terminates at.

    Records live in the env catalog, never in the graph; components cite their ids.
    ``stream`` names the user-defined env source the record was drawn from (one per camera,
    the trajectory, the label set, the map), so a citation preserves which stream fed a node,
    not just the data kind. ``derived_from`` supports derived media: an ``augmented_media``
    record (a rendered overlay) cites the raw frame and label records it was drawn from, so
    "was this node visually prompted with an overlay?" is answerable from the record itself.
    """

    id: str
    stream: str
    source_class: SourceClass
    summary: str
    timestamp_us: int | None = None
    path: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    derived_from: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        """Serialize to plain Python types."""
        result: dict[str, Any] = {
            "id": self.id,
            "stream": self.stream,
            "source_class": self.source_class.value,
            "summary": self.summary,
        }
        if self.timestamp_us is not None:
            result["timestamp_us"] = self.timestamp_us
        if self.path is not None:
            result["path"] = self.path
        if self.payload:
            result["payload"] = self.payload
        if self.derived_from:
            result["derived_from"] = list(self.derived_from)
        return result


def default_sources(
    cameras: tuple[str, ...] = (), include_external_labels: bool = True
) -> dict[str, SourceClass]:
    """The canonical env data-source manifest (documented in ``env/sources.md``).

    One entry per user-defined stream: the ego pose stream (per-timestamp poses whose
    sequence is the trajectory), the label set, the map, and **each camera individually** --
    the inventory is fixed at env init and hashed into the env id; reads during the run draw
    records against these streams. ``include_external_labels=False`` removes the
    external_label streams at the manifest level (the label-graduation direction): the
    streams do not exist in the contract, the env id rolls, and label-dependent ops must
    degrade gracefully.
    """
    sources = {"ego_pose": SourceClass.SENSOR}
    if include_external_labels:
        sources["object_labels"] = SourceClass.EXTERNAL_LABEL
        sources["map"] = SourceClass.EXTERNAL_LABEL
    sources.update({f"camera/{name}": SourceClass.SENSOR for name in cameras})
    return sources


@dataclass(frozen=True)
class Env:
    """Immutable per-clip episode contract; all data access is mediated by it.

    ``catalog`` accumulates the source records the run actually read -- part of the artifact,
    not of the contract hash.
    """

    env_id: str
    clip_id: str
    anchors: tuple[Anchor, ...]
    sampling: SamplingBounds
    access: AccessPolicy
    budget: RunBudget
    observations: tuple[Observation, ...]
    sources: dict[str, SourceClass] = field(default_factory=default_sources)
    data: ClipData | None = None
    reference: str = "labels"
    # reference concept set (contract when set): what the reference pass may denote
    reference_concepts: tuple[str, ...] | None = None
    # relevance scope (contract when set): distance <= scope[0] OR ego TTC <= scope[1]
    scope: tuple[float, float] | None = None
    catalog: dict[str, SourceRecord] = field(default_factory=dict)
    navigation: dict | None = None
    meta_actions: dict | None = None
    video_binding: dict | None = None
    sensor_binding: dict | None = None
    # per-timestamp camera-visibility cache for label annotation (not contract state);
    # a declared field so env clones via __dict__ round-trip cleanly
    _visibility_cache: dict = field(default_factory=dict)

    def meta_action_record(self, role: str, anchor: Anchor) -> SourceRecord | None:
        """Expose retrospective action labels only to a whole-clip teacher or evaluation.

        Trimming a label to an early anchor does not make its future-derived computation
        causal. Therefore ordinary anchor-agent reads are forbidden, not merely clipped.
        This context never enters graph facts or the exported policy observation view.
        """
        if role not in ("agent", "eval"):
            raise ValueError(f"unknown role: {role}")
        if self.meta_actions is None:
            return None
        if anchor not in self.anchors or (role == "agent" and anchor.trigger != "discovery"):
            raise ValueError("production action context requires the bound discovery teacher")
        return self._register(
            SourceRecord(
                id="env:production_meta_actions",
                stream="production_meta_actions",
                source_class=SourceClass.EXTERNAL_LABEL,
                summary="Privileged executed-action labels; never route intent or scene causes",
                payload=json.loads(json.dumps(self.meta_actions)),
            )
        )

    def navigation_records(self, role: str, anchor: Anchor) -> list[SourceRecord]:
        """Read timestamp-local route metadata inside the role's observation window.

        Route reconstruction is external metadata, not measured ego motion or permission.
        Future-derived ETA and retrospective decision fields are absent from the source.
        """
        if role not in ("agent", "eval"):
            raise ValueError(f"unknown role: {role}")
        if self.navigation is None or not self.navigation["available"]:
            return []
        lo, hi = anchor.window_us
        if role == "agent" and not self.access.agent_future_ego:
            hi = min(hi, anchor.anchor_us)
        records = []
        for sample in self.navigation["samples"]:
            timestamp = sample["timestamp_us"]
            if not lo <= timestamp <= hi:
                continue
            record = SourceRecord(
                id=f"env:navigation:{timestamp}",
                stream="navigation",
                source_class=SourceClass.EXTERNAL_LABEL,
                timestamp_us=timestamp,
                summary="Reconstructed route intent; not an observed action or scene cause",
                payload={**sample, "source": self.navigation["source"]},
            )
            self.catalog[record.id] = record
            records.append(record)
        return records

    def ego_states(self, role: str, anchor: Anchor) -> list[SourceRecord]:
        """Trajectory readings inside the anchor window, mediated by the access policy.

        The ``agent`` role sees no poses past the anchor unless the policy grants
        ``agent_future_ego``; the ``eval`` role always sees the full window. Each pose is
        registered as an ``env:ego:*`` source record on the ``ego_pose`` stream.
        """
        if role not in ("agent", "eval"):
            raise ValueError(f"unknown role: {role}")
        start_us, end_us = anchor.window_us
        if role == "agent" and not self.access.agent_future_ego:
            end_us = anchor.anchor_us
        records = []
        for observation in self.observations:
            if not start_us <= observation.timestamp_us <= end_us:
                continue
            records.append(
                self._register(
                    SourceRecord(
                        id=f"env:ego:{observation.timestamp_us}",
                        stream="ego_pose",
                        source_class=self.sources["ego_pose"],
                        summary="ego pose: "
                        + ", ".join(f"{k}={v}" for k, v in observation.data.items()),
                        timestamp_us=observation.timestamp_us,
                        payload=dict(observation.data),
                    )
                )
            )
        return records

    def labels(self, anchor: Anchor, t_us: int) -> list[SourceRecord]:
        """GT object labels near ``t_us`` inside the anchor window (``env:label:*`` records)."""
        if "object_labels" not in self.sources:
            raise ValueError("object_labels is not in this env's source manifest")
        self._require_data()
        start_us, end_us = anchor.window_us
        if not start_us <= t_us <= end_us:
            raise ValueError(f"t_us {t_us} outside the anchor window {anchor.window_us}")
        visible_by_camera = self._label_visibility(t_us)
        records = []
        for actor in actors_at(self.data.objects, self.data.ego, t_us):
            if visible_by_camera is not None:
                cameras = sorted(
                    camera for camera, ids in visible_by_camera.items() if actor["id"] in ids
                )
                if not cameras:
                    # the RSG is a world model SHAPED BY THE ENV (user direction): a
                    # front-camera playground reads the front-camera slice of the
                    # labels. Sightings the contracted cameras cannot see never enter
                    # a read -- want rear awareness, contract more cameras.
                    continue
                actor = {**actor, "visible_cameras": cameras}
            summary = (
                f"{actor['type']} {actor['id']}: state={actor['state']}, "
                f"forward={actor['forward_m']}m, lateral={actor['lateral_m']}m, "
                f"distance={actor['distance_m']}m, speed={actor['speed_mps']}m/s"
            )
            records.append(
                self._register(
                    SourceRecord(
                        id=f"env:label:{t_us}:{actor['id']}",
                        stream="object_labels",
                        source_class=self.sources["object_labels"],
                        summary=summary,
                        timestamp_us=t_us,
                        payload=actor,
                    )
                )
            )
        return records

    def _label_visibility(self, t_us: int) -> dict[str, set[str]] | None:
        """Per-contracted-camera visible track ids at ``t_us`` (calibrated projection).

        Labels are 360-degree lidar truth while the contracted view is a camera subset;
        whether a sighting is visible is a fact about this playground, computed once
        here. ``None`` when the clip carries no usable calibration -- annotation absent
        means UNKNOWN, and consumers stay fail-open (the pre-visibility behavior).
        """
        cache = self._visibility_cache
        if cache.get("__unavailable__"):
            return None
        if t_us in cache:
            return cache[t_us]
        if "__projector__" not in cache:
            try:
                cache["__projector__"] = self.projector()
            except (ValueError, KeyError):
                cache["__unavailable__"] = True
                return None
        projector = cache["__projector__"]
        height = self.sampling.image_max_height_px
        width = int(height * 16 / 9)  # nominal plane; the area floor scales with it
        visible = {
            camera: projector.visible_tracks(camera, t_us, width, height)
            for camera in self.sampling.cameras
        }
        cache[t_us] = visible
        return visible

    def map_elements(self, anchor: Anchor, t_us: int) -> list[SourceRecord]:
        """Closest map zones near ``t_us`` (``env:map:*`` records)."""
        if "map" not in self.sources:
            raise ValueError("map is not in this env's source manifest")
        self._require_data()
        records = []
        elements = lanes_at(
            self.data.lanes, self.data.lane_chunks, self.data.ego, t_us
        ) + crosswalks_at(self.data.crosswalks, self.data.ego, t_us)
        for element in elements:
            records.append(
                self._register(
                    SourceRecord(
                        id=f"env:map:{element['id']}",
                        stream="map",
                        source_class=self.sources["map"],
                        summary=f"{element['type']} {element['id']}: "
                        + (f"position={element['position']}, " if "position" in element else "")
                        + f"distance={element['distance_m']}m",
                        timestamp_us=t_us,
                        payload=element,
                    )
                )
            )
        return records

    def frames(
        self, camera: str, timestamps_us: list[int], cache_dir: Path, high_detail: bool = False
    ) -> list[SourceRecord]:
        """Decode frames under the sampling bounds (``env:frame:*`` records).

        Enforces the contract: the camera must be in the sampling set and the request must
        fit ``max_frames``. Decoded jpegs are cached under ``cache_dir``.
        """
        self._require_data()
        if camera not in self.sampling.cameras:
            raise ValueError(f"camera {camera} not in the env sampling set")
        if len(timestamps_us) > self.sampling.max_frames:
            raise ValueError(
                f"{len(timestamps_us)} frames requested, env allows {self.sampling.max_frames}"
            )
        video = self.data.cameras[camera]
        overview = self.sampling.image_overview_height_px
        height = self.sampling.image_max_height_px if high_detail or overview is None else overview
        suffix = f":h{height}" if overview is not None else ""
        cache_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for t_us in timestamps_us:
            frame_ts, jpeg = decode_frame(video["mp4"], video["timestamps"], t_us, height)
            path = cache_dir / f"{camera}_{frame_ts}{suffix.replace(':', '_')}.jpg"
            if not path.is_file():
                path.write_bytes(jpeg)
            records.append(
                self._register(
                    SourceRecord(
                        id=f"env:frame:{camera}:{frame_ts}{suffix}",
                        stream=f"camera/{camera}",
                        source_class=self.sources[f"camera/{camera}"],
                        summary=f"{camera} @ {frame_ts}us",
                        timestamp_us=frame_ts,
                        path=str(path),
                        payload={
                            **({"image_max_height_px": height} if overview is not None else {}),
                            **(
                                {"timing_source": "video_presentation_timestamps"}
                                if self.video_binding is not None
                                else {}
                            ),
                        },
                    )
                )
            )
        return records

    def video_segment(self, camera: str, window_us: tuple[int, int], cache_dir: Path):
        """One mp4 segment over ``window_us`` (an ``env:video:*`` record spanning it).

        The record carries no point timestamp -- its extent lives in the payload window --
        so time-scoped citations validate against the span, not a fictitious instant. Same
        raw-sensor stream as the camera's frames; the same access contract applies.
        """
        self._require_data()
        if camera not in self.sampling.cameras:
            raise ValueError(f"camera {camera} not in the env sampling set")
        video = self.data.cameras[camera]
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{camera}_{window_us[0]}_{window_us[1]}.mp4"
        first_ts, last_ts, count = extract_video_segment(
            video["mp4"],
            video["timestamps"],
            window_us,
            self.sampling.image_overview_height_px or self.sampling.image_max_height_px,
            path,
        )
        return self._register(
            SourceRecord(
                id=f"env:video:{camera}:{window_us[0]}_{window_us[1]}",
                stream=f"camera/{camera}",
                source_class=self.sources[f"camera/{camera}"],
                summary=f"{camera} segment {first_ts}..{last_ts}us ({count} frames)",
                path=str(path),
                payload={"window_us": [first_ts, last_ts], "frames": count},
            )
        )

    def frame_targets(self, anchor: Anchor) -> list[int]:
        """The frame timestamps observation works over: history to anchor, under the budget."""
        start_us, _ = anchor.window_us
        count = self.sampling.max_frames
        return [
            start_us + int(i * (anchor.anchor_us - start_us) / max(count - 1, 1))
            for i in range(count)
        ]

    def projector(self):
        """Calibrated box projection over this env's data (an access capability).

        Fails fast when the clip carries no calibration. The agent's render tool composes
        this with mediated frame/label reads; the env only supplies the geometry.
        """
        self._require_data()
        from .projection import Projector

        return Projector(self.data)

    def register_derived(self, record: SourceRecord) -> SourceRecord:
        """Register a record the run derived (agent-side) into the source catalog.

        Enforces the derivation invariants at the boundary: the stream must not shadow an
        env-carried stream, the class must be a derived one, and the derivation must be
        non-empty and resolve to already-cataloged records — chains terminate at the env.
        """
        if record.stream in self.sources:
            raise ValueError(f"derived record {record.id} shadows env stream {record.stream}")
        if record.source_class in (SourceClass.SENSOR, SourceClass.EXTERNAL_LABEL):
            raise ValueError(f"derived record {record.id} claims an env-carried class")
        if not record.derived_from:
            raise ValueError(f"derived record {record.id} carries no derivation")
        for ref in record.derived_from:
            if ref not in self.catalog:
                raise ValueError(f"derived record {record.id} derives from unknown {ref}")
        return self._register(record)

    def _register(self, record: SourceRecord) -> SourceRecord:
        """Idempotently add one source record to the catalog."""
        return self.catalog.setdefault(record.id, record)

    def _require_data(self) -> None:
        """Fail fast when a native read is attempted on an env without bound clip data."""
        if self.data is None:
            raise RuntimeError("this env has no native clip data bound (fixture env)")

    def contract_dict(self) -> dict:
        """The full contract as plain data (identity; the catalog is serialized separately)."""
        return {
            "env_id": self.env_id,
            "clip_id": self.clip_id,
            "anchors": [
                {
                    "anchor_us": anchor.anchor_us,
                    "history_us": anchor.history_us,
                    "future_us": anchor.future_us,
                    "trigger": anchor.trigger,
                }
                for anchor in self.anchors
            ],
            "sampling": {
                "max_fps": self.sampling.max_fps,
                "max_frames": self.sampling.max_frames,
                "cameras": list(self.sampling.cameras),
                "image_max_height_px": self.sampling.image_max_height_px,
                **(
                    {"image_overview_height_px": self.sampling.image_overview_height_px}
                    if self.sampling.image_overview_height_px is not None
                    else {}
                ),
            },
            "access": {
                "agent_gt_labels": self.access.agent_gt_labels,
                "agent_future_ego": self.access.agent_future_ego,
            },
            "budget": {
                "max_ops": self.budget.max_ops,
                "wall_clock_s": self.budget.wall_clock_s,
            },
            "sources": {name: cls.value for name, cls in self.sources.items()},
            **({"video_binding": self.video_binding} if self.video_binding is not None else {}),
            **({"sensor_binding": self.sensor_binding} if self.sensor_binding is not None else {}),
            **(
                {
                    "navigation_sha256": hashlib.sha256(
                        json.dumps(self.navigation, sort_keys=True).encode()
                    ).hexdigest()
                }
                if self.navigation is not None
                else {}
            ),
            **(
                {
                    "meta_actions_sha256": hashlib.sha256(
                        json.dumps(self.meta_actions, sort_keys=True).encode()
                    ).hexdigest()
                }
                if self.meta_actions is not None
                else {}
            ),
            # omitted at the default so historical artifacts stay byte-comparable
            **({"reference": self.reference} if self.reference != "labels" else {}),
            **(
                {"reference_concepts": list(self.reference_concepts)}
                if self.reference_concepts is not None
                else {}
            ),
            **({"scope": list(self.scope)} if self.scope is not None else {}),
            # label sightings carry per-camera visibility (env_visibility_plan):
            # a different playground from pre-annotation artifacts
            **({"label_visibility": True} if "object_labels" in self.sources else {}),
        }

    def catalog_dict(self) -> list[dict]:
        """The source records the run actually read, for the artifact."""
        return [record.to_dict() for record in self.catalog.values()]


def build_env(
    clip_id: str,
    anchors: tuple[Anchor, ...],
    sampling: SamplingBounds,
    access: AccessPolicy,
    budget: RunBudget,
    observations: tuple[Observation, ...],
    data: ClipData | None = None,
    include_external_labels: bool = True,
    reference: str = "labels",
    reference_concepts: tuple[str, ...] | None = None,
    scope: tuple[float, float] | None = None,
    navigation: dict | None = None,
    meta_actions: dict | None = None,
    video_binding: dict | None = None,
    sensor_binding: dict | None = None,
) -> Env:
    """Assemble an env and stamp its id as a hash of the full contract."""
    sources = default_sources(sampling.cameras, include_external_labels)
    if video_binding is not None:
        if (
            observations
            or navigation is not None
            or meta_actions is not None
            or include_external_labels
        ):
            raise ValueError("video-only binding cannot include hidden nonvisual inputs")
        sources.pop("ego_pose")
    if navigation is not None:
        navigation = json.loads(json.dumps(navigation))
        validate_navigation(navigation)
        sources["navigation"] = SourceClass.EXTERNAL_LABEL
    if meta_actions is not None:
        meta_actions = json.loads(json.dumps(meta_actions))
        validate_meta_actions(meta_actions)
        if any(anchor.trigger != "discovery" for anchor in anchors):
            raise ValueError("production action context requires discovery anchors")
        clock = meta_actions["clock"]
        if len(anchors) != 1 or not (
            anchors[0].window_us[0]
            <= clock["origin_us"]
            < clock["end_us"]
            <= anchors[0].window_us[1]
        ):
            raise ValueError("production action clock must fit the bound discovery window")
        sources["production_meta_actions"] = SourceClass.EXTERNAL_LABEL
    contract = {
        "clip_id": clip_id,
        "sources": {name: cls.value for name, cls in sources.items()},
        "anchors": [[a.anchor_us, a.history_us, a.future_us] for a in anchors],
        "sampling": [
            sampling.max_fps,
            sampling.max_frames,
            list(sampling.cameras),
            sampling.image_max_height_px,
        ],
        "access": [access.agent_gt_labels, access.agent_future_ego],
        "budget": [budget.max_ops, budget.wall_clock_s],
    }
    if sampling.image_overview_height_px is not None:
        contract["image_overview_height_px"] = sampling.image_overview_height_px
    if reference != "labels":
        # key omitted at the default so historical artifact hashes stay stable
        contract["reference"] = reference
    if video_binding is not None:
        contract["video_binding"] = video_binding
    if sensor_binding is not None:
        contract["sensor_binding"] = sensor_binding
    if navigation is not None:
        contract["navigation_sha256"] = hashlib.sha256(
            json.dumps(navigation, sort_keys=True).encode()
        ).hexdigest()
    if meta_actions is not None:
        contract["meta_actions_sha256"] = hashlib.sha256(
            json.dumps(meta_actions, sort_keys=True).encode()
        ).hexdigest()
    if reference_concepts is not None:
        contract["reference_concepts"] = list(reference_concepts)
    if scope is not None:
        contract["scope"] = list(scope)
    if include_external_labels:
        # visibility-annotated labels are a different playground (env_visibility_plan)
        contract["label_visibility"] = True
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    return Env(
        env_id=f"{clip_id}:{digest[:8]}",
        clip_id=clip_id,
        reference=reference,
        reference_concepts=reference_concepts,
        scope=scope,
        anchors=anchors,
        sampling=sampling,
        access=access,
        budget=budget,
        observations=observations,
        sources=sources,
        data=data,
        navigation=navigation,
        meta_actions=meta_actions,
        video_binding=video_binding,
        sensor_binding=sensor_binding,
    )


def native_anchors(
    data: ClipData, anchor_mode: str, history_us: int, future_us: int, max_segments: int
) -> tuple[Anchor, ...]:
    """The anchor set for one clip under the chosen framing (see ``build_env_native``)."""
    if anchor_mode == "discovery":
        return (
            Anchor(
                anchor_us=data.end_us,
                history_us=data.end_us - data.start_us,
                future_us=0,
                trigger="discovery",
            ),
        )
    if anchor_mode != "keyframe":
        raise ValueError(f"unknown anchor mode: {anchor_mode!r}")
    if not data.keyframes_us:
        raise ValueError(f"clip {data.clip_id} has no metadb keyframes to anchor an env on")
    return tuple(
        Anchor(
            anchor_us=keyframe_us,
            history_us=min(history_us, keyframe_us - data.start_us),
            future_us=min(future_us, data.end_us - keyframe_us),
            trigger="metadb keyframe",
        )
        for keyframe_us in data.keyframes_us[:max_segments]
    )


def build_env_native(
    metadb_path: str | Path | None,
    data_dir: str | Path | None,
    clip_id: str,
    *,
    cameras: tuple[str, ...] = ("camera_front_wide_120fov",),
    max_fps: float = 1.0,
    max_frames: int = 3,
    image_max_height_px: int = 720,
    image_overview_height_px: int | None = None,
    history_us: int = 2_000_000,
    future_us: int = 6_000_000,
    max_segments: int = 8,
    anchor_mode: str = "keyframe",
    include_external_labels: bool = True,
    reference: str = "labels",
    reference_concepts: tuple[str, ...] | None = None,
    scope: tuple[float, float] | None = None,
    max_ops: int = 64,
    wall_clock_s: float = 900.0,
    navigation_manifest: str | None = None,
    meta_action_manifest: str | None = None,
    sensor_manifest: str | None = None,
) -> Env:
    """Bind an env from raw clip data: the native path.

    ``anchor_mode``: ``keyframe`` anchors on the clip's metadb keyframes with the given
    horizons (clamped to the clip bounds); ``discovery`` is whole-clip event derivation --
    one window spanning the entire clip, no keyframe dependency and no decision semantics
    (the labeled keyframes never reach the agent; downstream consumers score against them
    through the eval role). The observation stream is the raw trajectory sampled at
    ``max_fps`` across the clip (pose + finite-difference kinematics -- sensor class, no
    rule-parsed tags).
    """
    if sensor_manifest is None or metadb_path is not None or data_dir is not None:
        raise ValueError(
            "Released native inputs require sensor_manifest; "
            "archive/metadb inputs are development-only"
        )
    from .sensor import load_sensor_data

    if (
        include_external_labels
        or anchor_mode != "discovery"
        or cameras != ("camera_front_wide_120fov",)
    ):
        raise ValueError("sensor manifest requires front-wide discovery without external labels")
    data, sensor_binding = load_sensor_data(sensor_manifest, clip_id)
    navigation = None
    if navigation_manifest is not None:
        manifest = json.loads(Path(navigation_manifest).read_text())
        if manifest["schema"] != "alparsg_navigation_timeline_v1":
            raise ValueError("unsupported navigation manifest")
        navigation = manifest["clips"][clip_id]
    meta_actions = None
    if meta_action_manifest is not None:
        manifest = json.loads(Path(meta_action_manifest).read_text())
        if manifest["schema"] != META_ACTION_SCHEMA:
            raise ValueError("unsupported production meta-action manifest")
        meta_actions = manifest["clips"][clip_id]
    for camera in cameras:
        if camera not in data.cameras:
            raise ValueError(f"camera {camera} not found in clip {clip_id}")
    anchors = native_anchors(data, anchor_mode, history_us, future_us, max_segments)
    # The grid starts at the clip start, so without anchoring it almost never lands on a
    # keyframe -- every anchor-time reading (and the labels queried at it) would be missed.
    step_us = max(int(round(1e6 / max_fps)), 1)
    grid = set(range(data.start_us, data.end_us + 1, step_us))
    grid.update(anchor.anchor_us for anchor in anchors)
    if sensor_binding is not None:
        lo, hi = sensor_binding["ego_coverage_us"]
        grid = {t_us for t_us in grid if lo <= t_us <= hi}
    observations = tuple(
        Observation(timestamp_us=t_us, data=ego_kinematics_at(data.ego, t_us))
        for t_us in sorted(grid)
    )
    sampling = SamplingBounds(
        max_fps=max_fps,
        max_frames=max_frames,
        cameras=cameras,
        image_max_height_px=image_max_height_px,
        image_overview_height_px=image_overview_height_px,
    )
    return build_env(
        clip_id=clip_id,
        anchors=anchors,
        sampling=sampling,
        access=AccessPolicy(),
        budget=RunBudget(max_ops=max_ops, wall_clock_s=wall_clock_s),
        observations=observations,
        data=data,
        include_external_labels=include_external_labels,
        reference=reference,
        reference_concepts=reference_concepts,
        scope=scope,
        navigation=navigation,
        meta_actions=meta_actions,
        sensor_binding=sensor_binding,
    )


def env_from_artifact(
    artifact: dict,
    *,
    max_fps: float = 1.0,
    image_max_height_px: int = 720,
    max_ops: int = 64,
    wall_clock_s: float = 900.0,
) -> Env:
    """Bind a fixture env from a reference reasoning_agent run artifact (test fixture only).

    The anchor timestamp is the numeric token of the segment id; horizons come from the
    segment window; the observation stream is the artifact's ego-state evidence; cameras are
    inferred from the frame evidence ids.
    """
    segment = artifact["segment"]
    clip_id = segment["clip_id"]
    start_us, end_us = segment["time_window_us"]
    numeric_tokens = [token for token in segment["segment_id"].split("_") if token.isdigit()]
    if not numeric_tokens:
        raise ValueError(f"segment id carries no anchor timestamp: {segment['segment_id']}")
    anchor_us = int(numeric_tokens[0])
    anchor = Anchor(
        anchor_us=anchor_us,
        history_us=anchor_us - start_us,
        future_us=end_us - anchor_us,
        trigger=segment["trigger_reason"],
    )

    observations = tuple(
        Observation(timestamp_us=item["timestamp_us"], data=dict(item.get("payload", {})))
        for item in artifact["evidence"]
        if item["source"] == "ego_state"
    )
    frame_items = [item for item in artifact["evidence"] if item["source"] == "frame"]
    cameras = tuple(
        sorted(
            {
                item["id"].split(":")[2 if item["id"].startswith("frame:rendered:") else 1]
                for item in frame_items
            }
        )
    )
    sampling = SamplingBounds(
        max_fps=max_fps,
        max_frames=max(len(frame_items), 1),
        cameras=cameras,
        image_max_height_px=image_max_height_px,
    )
    return build_env(
        clip_id=clip_id,
        anchors=(anchor,),
        sampling=sampling,
        access=AccessPolicy(),
        budget=RunBudget(max_ops=max_ops, wall_clock_s=wall_clock_s),
        observations=observations,
    )
