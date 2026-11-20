"""External-model render tool: text-prompted grounding via a SAM3-style endpoint."""

from __future__ import annotations

from collections.abc import Callable

# per-registry rotation offsets: every ready replica shares the load (mirrors
# ChatClient._candidates); the remaining candidates stay in order as failovers
_ROTATION: dict[str, int] = {}


def _ready_endpoints(registry_dir) -> list[dict]:
    """Ready registry records, round-robin rotated per call.

    A single-replica registry degenerates to the old freshest-first pick; with
    replicas, consecutive calls spread across the fleet (an idle second VFM was
    invisible to the old first-ready selection).
    """
    import json as json_module
    from pathlib import Path

    entries = sorted(Path(registry_dir).glob("r*.json"), reverse=True)
    records = [json_module.loads(entry.read_text()) for entry in entries]
    ready = [record for record in records if record.get("status") == "ready"]
    if not ready:
        return []
    key = str(registry_dir)
    _ROTATION[key] = _ROTATION.get(key, -1) + 1
    offset = _ROTATION[key] % len(ready)
    return ready[offset:] + ready[:offset]


def _post_failing_over(registry_dir, route: str, body: dict, timeout: int, kind: str):
    """POST to each ready replica in turn; return (response_json, endpoint).

    The registry contract is "a file exists iff a replica is ready", but a replica that
    dies without clearing its entry breaks it -- and a first-pick client turns that stale
    file into a failed call even when healthy replicas are registered beside it. Measured
    2026-08-24: a fleet restart left one dead entry and every grounding call in the
    phase-2 harness failed against it. So every call walks the ready set, as the
    appearance judges already do.
    """
    import requests

    endpoints = _ready_endpoints(registry_dir)
    if not endpoints:
        raise RuntimeError(f"no ready {kind} endpoint in {registry_dir}")
    last_error: Exception | None = None
    for endpoint in endpoints:
        try:
            response = requests.post(f"{endpoint['url']}{route}", json=body, timeout=timeout)
            response.raise_for_status()
            return response.json(), endpoint
        except requests.RequestException as error:
            last_error = error
    raise RuntimeError(f"{kind} endpoint unreachable: {last_error}")


def ground_detections(registry_dir, image_path: str, prompt: str) -> list[dict]:
    """Text-prompted detections on one image, as structured boxes.

    Posts the image and prompt to the freshest ready endpoint's ``POST /v1/ground`` and
    returns the raw detections (``box_xyxy_norm``, ``score``). This is the structured core
    both consumers share: the registered tool formats it as text for the model; the
    merge:associate op matches the boxes against projected label boxes.
    """
    import base64
    from pathlib import Path

    image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    # a dead/preempted grounding server must degrade (consumers catch RuntimeError and
    # skip association), never kill the run mid-batch
    body, _ = _post_failing_over(
        registry_dir,
        "/v1/ground",
        {"image_b64": image_b64, "text": prompt},
        timeout=120,
        kind="grounding",
    )
    return body.get("detections", [])


def segment_regions(registry_dir, image_path: str, prompt: str) -> tuple[list[dict], str]:
    """Text-prompted segmentation on one image: detections with full-res RLE masks.

    The UNTRACKED baseline: ``ground:reference`` runs on ``track_regions`` now, and this
    is retained as the comparison arm for the tracked-pass recall measurement
    (tracked_reference_plan phase 2). It has no production consumer.

    Posts to the freshest ready endpoint's ``POST /v1/segment``; returns the detections
    (``box_xyxy_norm``, ``score``, ``mask_rle``) and the serving model identity (the
    reference reproducibility pin). Endpoint failures wrap as RuntimeError so consumers
    degrade instead of crashing runs.
    """
    import base64
    from pathlib import Path

    image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    body, endpoint = _post_failing_over(
        registry_dir,
        "/v1/segment",
        {"image_b64": image_b64, "text": prompt, "score_threshold": 0.3},
        timeout=120,
        kind="segmentation",
    )
    return body.get("detections", []), endpoint.get("model", "unknown")


def track_regions(
    registry_dir, image_paths: list, prompt: str
) -> tuple[dict[int, list[dict]], str]:
    """Text-prompted TRACKED segmentation over an ordered frame window.

    One ``POST /v1/track`` call per concept replaces one ``/v1/segment`` call per frame,
    and the detections come back ALREADY ASSOCIATED: the video model runs its detector
    on every frame and admits new instances mid-window, so this is per-frame detection
    carrying a stable ``obj_id``, not a single box propagated from the prompt frame.

    Returns ``{frame_position: [detection, ...]}`` keyed by index into ``image_paths``,
    each detection shaped like a ``/v1/segment`` one (``box_xyxy_norm``, ``score``,
    ``mask_rle``) plus ``track_id`` -- so a per-frame consumer reads it unchanged and an
    identity-aware consumer reads the correspondence perception already asserted.

    The track id is a CLAIM about correspondence, not a fact: callers are expected to
    check it against an independent mechanism (world-position spread, for elements)
    rather than trust it blindly. Endpoint failures wrap as RuntimeError so consumers
    degrade instead of crashing runs.
    """
    import base64
    from pathlib import Path

    body, endpoint = _post_failing_over(
        registry_dir,
        "/v1/track",
        {
            "frames_b64": [
                base64.b64encode(Path(path).read_bytes()).decode() for path in image_paths
            ],
            "text": prompt,
            "box_xyxy_norm": None,
            "prompt_frame_index": 0,
            "score_threshold": 0.3,
            "with_masks": True,
        },
        timeout=1800,
        kind="tracking",
    )
    by_frame: dict[int, list[dict]] = {}
    for track in body.get("tracks", []):
        for observation in track["observations"]:
            if observation.get("mask_rle") is None:
                continue  # a mask-less observation cannot feed element geometry
            by_frame.setdefault(int(observation["frame_index"]), []).append(
                {
                    "box_xyxy_norm": observation["box_xyxy_norm"],
                    "score": observation["score"],
                    "mask_rle": observation["mask_rle"],
                    "track_id": int(track["obj_id"]),
                }
            )
    return by_frame, endpoint.get("model", "unknown")


def make_grounding_tool(registry_dir, latest_frame_path: Callable[[], str | None]):
    """External-model render tool: text-prompted grounding via a SAM3-style endpoint.

    The second of the two render tools (the first is the native box projector): grounds the
    latest observed frame with a text prompt and returns the detections as text. The result
    enters the context layer as a ``tool_output`` record when an op records the call.
    """

    def ground(rsg, prompt: str) -> str:
        frame_path = latest_frame_path()
        if frame_path is None:
            return "no frame available to ground on"
        try:
            detections = ground_detections(registry_dir, frame_path, prompt)
        except RuntimeError as error:
            return str(error)
        if not detections:
            return f"no detections for {prompt!r}"
        return "\n".join(
            f"box_xyxy_norm={[round(v, 3) for v in d['box_xyxy_norm']]} "
            f"score={round(d.get('score', 0.0), 3)}"
            for d in detections
        )

    return ground


def agent_ground_regions(
    registry_dir, llm_registry_dir, image_path: str, text: str
) -> tuple[list[dict], str, list[dict]]:
    """Agentic referring-expression grounding: the SAM3 /v1/agent VLM loop on one image.

    The loop's VLM resolves through ``llm_registry_dir`` (round-robin over ready
    replicas). Returns detections shaped like ``/v1/segment``, the serving model
    identity, and the loop's prompt trace ([{phrase, selected}] -- the wording that
    actually generated the masks; empty from pre-trace servers). Endpoint failures
    wrap as RuntimeError so consumers degrade.
    """
    import base64
    from pathlib import Path

    import requests

    sam3s = _ready_endpoints(registry_dir)
    llms = _ready_endpoints(llm_registry_dir)
    if not sam3s or not llms:
        raise RuntimeError("no ready sam3/llm endpoint for agent grounding")
    global _AGENT_ROTATION
    start = _AGENT_ROTATION
    _AGENT_ROTATION += 1
    image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    last_error: Exception | None = None
    # Walk DISTINCT (sam3, llm) pairs rather than retrying one pair: measured 2026-08-24,
    # two of four qwen replicas were registered ready but dead, and retrying the same
    # corpse twice turned ~half of every identity wave into skips (33 of 45 claims). The
    # extra pass keeps the transient-500 retry the capability probe called for (~11%).
    attempts = max(len(sam3s), len(llms)) + 1
    for attempt in range(attempts):
        sam3 = sam3s[attempt % len(sam3s)]
        llm = llms[(start + attempt) % len(llms)]
        try:
            response = requests.post(
                f"{sam3['url']}/v1/agent",
                json={
                    "image_b64": image_b64,
                    "text": text,
                    "llm_url": llm["url"],
                    "llm_model": llm["model"],
                    "max_generations": 12,
                },
                timeout=600,
            )
            response.raise_for_status()
            body = response.json()
            return (
                body.get("detections", []),
                sam3.get("model", "unknown"),
                body.get("prompt_trace", []),
            )
        except requests.RequestException as error:
            last_error = error
    raise RuntimeError(
        f"agent grounding failed across {attempts} endpoint pairs "
        f"({len(sam3s)} sam3 x {len(llms)} llm): {last_error}"
    )


_AGENT_ROTATION = 0


def judge_signal_face(llm_registry_dir, image_path: str, box_xyxy_norm: list) -> tuple[bool, str]:
    """Does this signal head FACE the camera? Raw-crop yes/no, for signal groups.

    Co-located heads on one mast serve different approaches, and their world positions
    are 0.6m apart against 1.35m of depth noise -- geometry cannot tell them apart
    (map_reference_plan phase-3 measurement). What distinguishes them is which way each
    faces, and that IS visible in a crop: a head facing the camera shows its round lamp
    faces, an edge-on or rear head shows a thin dark profile or a blank back.

    Returns (faces_camera, reason-words). Deliberately NOT applicability: a head may
    face ego and still govern a different lane (caveat B, still deferred).
    """
    import base64
    import io

    import requests
    from PIL import Image

    endpoints = _ready_endpoints(llm_registry_dir)
    if not endpoints:
        raise RuntimeError(f"no ready chat endpoint in {llm_registry_dir}")
    image = Image.open(image_path)
    width, height = image.size
    x0, y0, x1, y1 = box_xyxy_norm
    crop = image.crop(
        (
            int(max(x0 - 0.02, 0) * width),
            int(max(y0 - 0.02, 0) * height),
            int(min(x1 + 0.02, 1) * width),
            int(min(y1 + 0.02, 1) * height),
        )
    )
    scale = max(1, int(128 / max(crop.height, 1)))  # signal heads are small and far
    if scale > 1:
        crop = crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)
    buffer = io.BytesIO()
    crop.save(buffer, format="JPEG")
    crop_b64 = base64.b64encode(buffer.getvalue()).decode()
    last_error: Exception | None = None
    for llm in endpoints:  # a dead-but-registered replica must not lose the verdict
        try:
            response = requests.post(
                f"{llm['url']}/v1/chat/completions",
                json={
                    "model": llm["model"],
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"},
                                },
                                {
                                    "type": "text",
                                    "text": (
                                        "This crop shows one traffic signal head. Judge ONLY "
                                        "its orientation: is its LAMP FACE pointing toward this "
                                        "camera (round lamps visible head-on), or is it seen "
                                        "edge-on / from behind (thin profile, housing, no lamp "
                                        "faces)? Say nothing about which lane it governs. "
                                        "Answer strictly as: yes/no, then a few words for what "
                                        "you see."
                                    ),
                                },
                            ],
                        }
                    ],
                    "temperature": 0,
                    # thinking is on by default on Qwen3.8 and consumes the budget
                    # before any content appears; disable it and leave room to answer
                    "chat_template_kwargs": {"enable_thinking": False},
                    "max_tokens": 256,
                },
                timeout=120,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            last_error = error  # registered but dead: try the next replica
            continue
        reply = response.json()["choices"][0]["message"]
        text = (reply.get("content") or reply.get("reasoning_content") or "").strip()
        verdict = text.split(",")[0].strip().lower()
        if not (verdict.startswith("yes") or verdict.startswith("no")):
            # An empty or unparsable reply is an ABSTENTION, never a negative verdict:
            # scoring silence as "edge-on" manufactures a judgement out of nothing,
            # which is what this design refuses to do (measured live: thinking-mode
            # replies came back empty and were silently scored edge).
            raise RuntimeError(f"signal-face judge gave no verdict: {text[:80]!r}")
        return verdict.startswith("yes"), text
    raise RuntimeError(f"signal-face judge unreachable: {last_error}")


def judge_appearance(
    llm_registry_dir, image_path: str, box_xyxy_norm: list, claim: str, complete=None
) -> tuple[bool, str]:
    """Appearance cross-check on RAW pixels: does the bound object match the claim?

    Crops the box from the untouched frame and asks the VLM (temperature 0) whether
    the object's appearance matches the claim's description -- never a rendered
    overlay (the loop's 15%-alpha mask tint measurably flips color judgments; the
    red-SUV post-mortem). Returns (matched, actual-appearance words). Endpoint
    failures wrap as RuntimeError so consumers degrade. Batch callers can supply an
    accounted completion callback; it must preserve the judge's 256-token, zero-temperature
    settings. The crop, prompt, and verdict parsing are identical in both transports.
    """
    import base64
    import io

    import requests
    from PIL import Image

    image = Image.open(image_path)
    width, height = image.size
    x0, y0, x1, y1 = box_xyxy_norm
    crop = image.crop(
        (
            int(max(x0 - 0.02, 0) * width),
            int(max(y0 - 0.02, 0) * height),
            int(min(x1 + 0.02, 1) * width),
            int(min(y1 + 0.02, 1) * height),
        )
    )
    scale = max(1, int(96 / max(crop.height, 1)))  # small crops upscale for legibility
    if scale > 1:
        crop = crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)
    buffer = io.BytesIO()
    crop.save(buffer, format="JPEG")
    crop_b64 = base64.b64encode(buffer.getvalue()).decode()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"},
                },
                {
                    "type": "text",
                    "text": (
                        f"Claim: {claim}\n"
                        "Judge ONLY the object's color and type against the "
                        "claim; IGNORE all position, lane, motion, and "
                        "visibility statements (they are checked elsewhere and "
                        "cannot be judged from a crop). If the claim names no "
                        "color or type, answer yes. Answer strictly as: yes/no, "
                        "then the object's actual color and type."
                    ),
                },
            ],
        }
    ]
    if complete is not None:
        return _judge_verdict(complete(messages).strip(), claim)
    endpoints = _ready_endpoints(llm_registry_dir)
    if not endpoints:
        raise RuntimeError(f"no ready chat endpoint in {llm_registry_dir}")
    # a dead-but-registered replica must not cost the verdict: the same failover the
    # signal-face judge already runs, for the same measured reason
    response, last_error = None, None
    for llm in endpoints:
        try:
            response = _judge_post(llm, messages)
            break
        except requests.RequestException as error:
            last_error = error
    if response is None:
        raise RuntimeError(f"appearance judge unreachable: {last_error}")
    text = response.json()["choices"][0]["message"]["content"].strip()
    return _judge_verdict(text, claim)


def _judge_post(llm: dict, messages: list[dict]):
    """One appearance-judge call against one replica."""
    import requests

    response = requests.post(
        f"{llm['url']}/v1/chat/completions",
        json={
            "model": llm["model"],
            "messages": messages,
            "max_tokens": 256,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=120,
    )
    response.raise_for_status()
    return response


def _judge_verdict(text: str, claim: str) -> tuple[bool, str]:
    """Parse the judge's reply into (matched, actual description).

    ``claim`` is load-bearing for the self-consistency rescue below, not decoration.
    """
    matched = text.lower().lstrip().startswith("yes")
    actual = text.split(",", 1)[1].strip() if "," in text else text
    if not matched:
        # the yes/no token is the least stable draw (it measurably contradicts the
        # judge's own description ~30% of the time); trust the description: a "no"
        # whose description words all appear in the claim is self-inconsistent -> match
        import re

        actual_words = set(re.findall(r"[a-z]+", actual.lower())) - {
            "the",
            "a",
            "an",
            "is",
            "in",
            "of",
            "colored",
            "color",
            "object",
            "and",
        }
        claim_words = set(re.findall(r"[a-z]+", claim.lower()))
        if actual_words and actual_words <= claim_words:
            matched = True
    return matched, actual[:80]


def track_window(registry_dir, frame_paths: list, box_xyxy_norm: list) -> tuple[dict, str]:
    """Box-seeded tracking over an ordered frame window: {frame_position: box}.

    The serving path propagates FORWARD-ONLY from the prompt frame (phase-1 probe,
    measured via reversed-order diagnostic), so callers order ``frame_paths`` with the
    SEED FRAME FIRST (backward sweep from a window-end binding). Returns the seeded
    track's boxes keyed by position in ``frame_paths`` plus the serving model identity.
    Endpoint failures wrap as RuntimeError so consumers degrade.
    """
    import base64
    from pathlib import Path

    body, endpoint = _post_failing_over(
        registry_dir,
        "/v1/track",
        {
            "frames_b64": [
                base64.b64encode(Path(path).read_bytes()).decode() for path in frame_paths
            ],
            "box_xyxy_norm": box_xyxy_norm,
            "prompt_frame_index": 0,
            "text": None,
        },
        timeout=600,
        kind="tracking",
    )

    def iou(a, b):
        ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
        ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        if inter == 0.0:
            return 0.0
        return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)

    seeded, best = None, 0.0
    for track in body.get("tracks", []):
        at_seed = next(
            (o["box_xyxy_norm"] for o in track["observations"] if o["frame_index"] == 0), None
        )
        if at_seed is not None and iou(at_seed, box_xyxy_norm) > best:
            best, seeded = iou(at_seed, box_xyxy_norm), track
    boxes = {o["frame_index"]: o["box_xyxy_norm"] for o in seeded["observations"]} if seeded else {}
    return boxes, endpoint.get("model", "unknown")


def da3_geometry(registry_dir, image_paths: list) -> tuple[dict, str]:
    """One DA3 inference over a frame set: depth/conf/intrinsics/extrinsics arrays.

    Async job API: submit server-visible paths, poll, load the npz from the shared
    filesystem. Returns ({depth, conf, intrinsics, extrinsics} as numpy arrays in
    input order, model identity). Endpoint failures wrap as RuntimeError.
    """
    import json as json_module
    import time
    import urllib.error
    import urllib.request
    import uuid
    import zipfile
    from pathlib import Path

    import numpy as np

    # the task id lives on ONE server, so a live endpoint is chosen up front and kept;
    # a stale registry entry is skipped here rather than losing the whole pass
    record = None
    for candidate in _ready_endpoints(registry_dir):
        try:
            # /v1/models is the readiness route the registry contract defines for every
            # service here (DA3 has no /health -- probing it would reject every endpoint)
            urllib.request.urlopen(f"{candidate['url']}/v1/models", timeout=10).read()
        except (urllib.error.URLError, OSError):
            continue
        record = candidate
        break
    if record is None:
        raise RuntimeError(f"no reachable geometry endpoint in {registry_dir}")
    export = f"gallery/alparsg_geom/{uuid.uuid4().hex[:12]}"

    def call(route, body=None):
        request = urllib.request.Request(
            f"{record['url']}{route}",
            data=json_module.dumps(body).encode() if body is not None else None,
            headers={
                "content-type": "application/json",
                record.get("api_key_header", "X-API-Key"): record.get("api_key", ""),
            },
        )
        return json_module.load(urllib.request.urlopen(request, timeout=120))

    try:
        task = call(
            "/inference",
            {
                "image_paths": [str(Path(p).resolve()) for p in image_paths],
                "export_dir": export,
                "export_format": "npz",
            },
        )
        deadline = time.time() + 600
        while time.time() < deadline:
            status = call(f"/task/{task['task_id']}")
            if status["status"] == "completed":
                break
            if status["status"] in ("failed", "error"):
                raise RuntimeError(f"geometry task failed: {status.get('message')}")
            time.sleep(3)
        else:
            raise RuntimeError("geometry task timed out")
    except (urllib.error.URLError, OSError) as error:
        raise RuntimeError(f"geometry endpoint unreachable: {error}") from error
    npz_path = Path(record["export_root"]) / export / "exports" / "npz" / "results.npz"
    data = None
    for _ in range(10):  # Lustre visibility lags task completion; wait out partial writes
        try:
            if npz_path.is_file():
                data = np.load(npz_path)
                break
        except (OSError, ValueError, EOFError, zipfile.BadZipFile):
            # partially-written npz surfaces as any of these depending on how far
            # the writer got; all mean "not ready yet"
            pass
        time.sleep(2)
    if data is None:
        raise RuntimeError(f"geometry result unreadable at {npz_path}")
    return (
        {key: data[key] for key in ("depth", "conf", "intrinsics", "extrinsics")},
        record.get("model", "unknown"),
    )
