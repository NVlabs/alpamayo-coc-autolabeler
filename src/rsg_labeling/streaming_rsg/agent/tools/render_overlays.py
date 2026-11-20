"""The native render tool: scaffold overlays, the agent's enhanced-visual-context choice.

The first of the two render tools (the second is ``ground_frame``). The env supplies the
mediated ingredients (frames, labels, calibrated projection) and validates the derived
records at registration; this tool composes them and decides nothing about *whether* to
render -- that is the caller's (op or model) policy.

TWO SCAFFOLDS, ONE PATH (agent_tracks_plan): external labels when the env carries them,
VFM TRACKLETS otherwise. They occupy the same epistemic slot -- specialized outputs the
agent references without fully trusting -- so they get the same affordance: a box with an
id the model can point at. Without this a labelless run has nothing to reference, which
made labelled and labelless two different workflows rather than one.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ....common.overlays import draw_numbered_boxes
from ...data_structure.rsg import SourceClass
from ...env.env import Anchor, Env, SourceRecord
from ...env.projection import ProjectedBox

# Which reference concepts render as scaffold. Dynamic agents only: map elements already
# anchor as components, and overlaying everything reintroduces the clutter the numbered-pin
# work removed.
_SCAFFOLD_KINDS = ("vehicle", "pedestrian", "cyclist")


def has_tracklets(env) -> bool:
    """Whether the env carries VFM tracklets that could scaffold an overlay."""
    return any(
        (record.stream or "").startswith("reference/")
        and any(
            obj.get("track_id") is not None
            for obj in (record.payload or {}).get("objects", []) or []
        )
        for record in env.catalog.values()
    )


def tracklet_boxes(env, timestamp_us: int, width: int, height: int):
    """VFM tracklet boxes for one frame: ``(boxes, source_id_by_track)``.

    The tracklet id is what the model points at, so it is the box's ``track_id`` -- the
    same slot a label track occupies. ``source_id_by_track`` carries the reference record
    each box came from, so the render's ``derived_from`` chain stays legal (invariant 4)
    whichever scaffold produced it.
    """
    boxes, sources = [], {}
    for record in env.catalog.values():
        stream = record.stream or ""
        if not stream.startswith("reference/") or record.timestamp_us != timestamp_us:
            continue
        if stream.split("/", 1)[1] not in _SCAFFOLD_KINDS:
            continue
        for obj in (record.payload or {}).get("objects", []) or []:
            track_id = obj.get("track_id")
            if track_id is None or not obj.get("box_xyxy_norm"):
                continue
            x0, y0, x1, y1 = obj["box_xyxy_norm"]
            boxes.append(
                ProjectedBox(
                    track_id=str(track_id),
                    x0=int(x0 * width),
                    y0=int(y0 * height),
                    x1=int(x1 * width),
                    y1=int(y1 * height),
                    distance_m=0.0,  # range is not known at reference time; unused here
                )
            )
            sources[str(track_id)] = record.id
    return boxes, sources


def draw_boxes(image_bgr: np.ndarray, boxes: list[ProjectedBox]) -> np.ndarray:
    """Return a copy of the frame with each box drawn and labeled with its track id."""
    annotated = image_bgr.copy()
    for box in boxes:
        cv2.rectangle(annotated, (box.x0, box.y0), (box.x1, box.y1), (0, 255, 0), 1)
        cv2.putText(
            annotated,
            box.track_id,
            (box.x0, max(10, box.y0 - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


# Distinct marker palette (ported by design from reasoning_agent's
# rendered_marker_declutter_plan; reimplemented -- invariant 7). A pin number picks
# its slot by value, so a run-stable number keeps its color across frames.
_MARKER_PALETTE = (
    (0, 255, 0),
    (255, 128, 0),
    (0, 128, 255),
    (0, 255, 255),
    (255, 0, 255),
    (128, 0, 255),
    (255, 255, 0),
    (0, 0, 255),
)


def render_overlay_records(
    env: Env,
    camera: str,
    timestamps_us: list[int],
    cache_dir: Path,
    anchor: Anchor,
    style: str = "track_id",
    pin_numbers: dict[str, int] | None = None,
) -> list[SourceRecord]:
    """Render label-box overlays for the target frames; register augmented_media records.

    ``style="numbered_pin"`` renders the decluttered form: run-stable pin numbers
    (``pin_numbers``: track_id -> number, growing across calls) instead of corner
    track-id text; the observe roster maps pins back to component ids in text.
    """
    # the label scaffold is consulted only when the env carries it: env.labels() and
    # env.projector() both raise on a sensor-only manifest, which is precisely the run
    # that needs the tracklet scaffold instead
    has_labels = "object_labels" in env.sources
    projector = env.projector() if has_labels else None
    records = []
    # env.frames enforces the per-read sampling bound; batch like the observe rounds
    # do (<= max_frames per read) so a whole-run target list stays contract-legal
    frames = []
    step = max(env.sampling.max_frames, 1)
    for start in range(0, len(timestamps_us), step):
        frames.extend(env.frames(camera, timestamps_us[start : start + step], cache_dir))
    for frame in frames:
        image = cv2.imread(frame.path)
        boxes, label_by_track = [], {}
        if has_labels:
            boxes = projector.boxes_for(camera, frame.timestamp_us, image.shape[1], image.shape[0])
            # The decoded frame snaps to the nearest real frame timestamp, which can fall
            # microseconds outside the window; clamp the label lookup back inside it.
            start_us, end_us = anchor.window_us
            label_t_us = min(max(frame.timestamp_us, start_us), end_us)
            label_by_track = {
                record.payload["id"]: record.id for record in env.labels(anchor, label_t_us)
            }
        drawn = [box for box in boxes if box.track_id in label_by_track]
        source_by_track, scaffold = dict(label_by_track), "labeled track"
        if not drawn:
            # labelless: the VFM tracklets ARE the scaffold, same affordance, same
            # not-fully-trusted status -- the agent points at them and may revise them
            drawn, source_by_track = tracklet_boxes(
                env, frame.timestamp_us, image.shape[1], image.shape[0]
            )
            scaffold = "vfm tracklet"
        suffix = "pins" if style == "numbered_pin" else "boxes"
        path = cache_dir / f"{camera}_{frame.timestamp_us}_{suffix}.jpg"
        if style == "numbered_pin":
            numbers = pin_numbers if pin_numbers is not None else {}
            for box in drawn:
                numbers.setdefault(box.track_id, len(numbers) + 1)
            annotated = draw_numbered_boxes(
                image, drawn, [str(numbers[box.track_id]) for box in drawn], palette=_MARKER_PALETTE
            )
        else:
            annotated = draw_boxes(image, drawn)
        if not path.is_file():
            ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise RuntimeError(f"jpeg encode failed for overlay {path}")
            path.write_bytes(encoded.tobytes())
        records.append(
            env.register_derived(
                SourceRecord(
                    id=f"env:render:{camera}:{frame.timestamp_us}",
                    stream="derived/box_render",
                    source_class=SourceClass.AUGMENTED_MEDIA,
                    summary=f"{camera} @ {frame.timestamp_us}us with {len(drawn)} {scaffold} boxes",
                    timestamp_us=frame.timestamp_us,
                    path=str(path),
                    payload={
                        "tracks": [box.track_id for box in drawn],
                        **(
                            {"pins": {str(numbers[b.track_id]): b.track_id for b in drawn}}
                            if style == "numbered_pin"
                            else {}
                        ),
                    },
                    # deduped: many boxes share one reference record, and repeating a
                    # parent id makes the chain walker read siblings as a cycle
                    derived_from=tuple(
                        dict.fromkeys((frame.id, *(source_by_track[box.track_id] for box in drawn)))
                    ),
                )
            )
        )
    return records


def make_render_tool(env, anchor, cache_dir, style: str = "track_id"):
    """Bind the render tool to one run's env, anchor, overlay cache, and marker style."""
    pin_numbers: dict[str, int] = {}  # run-stable: a track keeps its number and color

    def render(rsg, camera: str = "", timestamps_us: list[int] | None = None) -> str:
        """Render overlays for ``timestamps_us`` (default: the anchor frame grid).

        Observe passes its own round targets: overlays only substitute at EXACT
        frame timestamps, and discovery rounds sample between the anchor grid --
        the mismatch left the overlays rendered-but-unconsumed (found live: the
        VLM associated claims to track ids it had never seen on an image).
        """
        cameras = [camera] if camera else list(env.sampling.cameras)
        targets = timestamps_us if timestamps_us is not None else env.frame_targets(anchor)
        summaries = []
        for name in cameras:
            records = render_overlay_records(
                env, name, targets, cache_dir, anchor, style=style, pin_numbers=pin_numbers
            )
            summaries.extend(record.summary for record in records)
        return "\n".join(summaries) if summaries else "no frames rendered"

    return render
