"""Run configuration: the typed schema and its ownership scopes.

Three scopes by who owns the values (docs/infra/env_config_plan.md):

- **playground** -- user-defined; the task. Env contract inputs and the evaluation
  references. Hashed into ``playground_id``; the loop must never mutate it, and
  comparisons are valid only within one playground_id.
- **policy** -- the alpaloop-evolvable surface: program knobs, experimental thresholds,
  generation parameters. Hashed into ``policy_id``; a candidate is one policy overlay
  over a frozen playground.
- **infra** -- endpoint/registry paths; no semantic content, excluded from both hashes
  (the same run on a different server is the same run).

Defaults reproduce the pre-config behavior exactly, so ``run --clip X`` without a config
file is unchanged. The fully resolved config is written into every run dir
(``config.yaml``) with both hashes recorded in the trace; the validator checks the stored
config against the artifact's env contract, so config drift is a validation error.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_serializer, model_validator

# Stream-name -> reliability-class names the manifest list resolves through
# (mirrors env.default_sources; cameras are appended from playground.env.cameras).
KNOWN_STREAMS = ("ego_pose", "object_labels", "map")


class BudgetConfig(BaseModel):
    """User-set execution bounds (env contract)."""

    max_ops: int = 64
    wall_clock_s: float = 900.0


class EnvConfig(BaseModel):
    """Inputs to ``build_env_native`` -- resolves to the hashed env contract."""

    anchor_mode: str = "keyframe"  # keyframe | discovery
    sources: list[str] = Field(default_factory=lambda: list(KNOWN_STREAMS))
    cameras: list[str] = Field(default_factory=lambda: ["camera_front_wide_120fov"])
    history_s: float = 2.0
    future_s: float = 6.0
    max_fps: float = 1.0
    max_frames: int = 3
    image_max_height_px: int = 720
    image_overview_height_px: int | None = None
    # evidence availability: labels only (default), model-derived reference beside the
    # labels (hybrid), or model reference only. Env contract input -> hashed.
    reference: str = "labels"  # labels | hybrid | model
    # which concepts the reference pass grounds (None = the op's default trio). What
    # perception can even DENOTE is a property of the playground, so a changed set is a
    # different playground; the key stays out of the contract at None so old hashes hold.
    reference_concepts: list[str] | None = None
    # relevance scope: agents enter the graph when distance <= scope_distance_m OR
    # ego TTC <= scope_ttc_s (the union -- the larger envelope wins). None = no scope
    # (everything enters; the keys stay out of the contract, so old hashes stand).
    scope_distance_m: float | None = None
    scope_ttc_s: float | None = None
    budgets: BudgetConfig = Field(default_factory=BudgetConfig)
    navigation_manifest: str | None = None
    meta_action_manifest: str | None = None
    video_manifest: str | None = None
    sensor_manifest: str | None = None

    @model_serializer(mode="wrap")
    def _serialize_inputs(self, handler):
        """Keep an absent optional input out of persisted native/video configurations."""
        result: dict = handler(self)
        if self.sensor_manifest is None:
            result.pop("sensor_manifest", None)
        return result

    @model_validator(mode="after")
    def _metadata_contract(self) -> EnvConfig:
        """Require explicit opt-in and pinned sources for optional teacher metadata."""
        if self.image_overview_height_px is not None and not (
            0 < self.image_overview_height_px < self.image_max_height_px
        ):
            raise ValueError("overview height must be positive and below the image ceiling")
        if self.video_manifest is not None and (
            self.sources
            or self.anchor_mode != "discovery"
            or self.cameras != ["camera_front_wide_120fov"]
        ):
            raise ValueError("video manifest requires camera-only discovery inputs")
        if self.sensor_manifest is not None and (
            self.video_manifest is not None
            or self.anchor_mode != "discovery"
            or self.cameras != ["camera_front_wide_120fov"]
            or set(self.sources) - {"ego_pose", "navigation", "production_meta_actions"}
            or "ego_pose" not in self.sources
        ):
            raise ValueError("sensor manifest requires declared ego/front-wide discovery inputs")
        if ("navigation" in self.sources) != (self.navigation_manifest is not None):
            raise ValueError(
                "navigation stream and navigation_manifest must be configured together"
            )
        if ("production_meta_actions" in self.sources) != (self.meta_action_manifest is not None):
            raise ValueError(
                "production_meta_actions and meta_action_manifest must be set together"
            )
        if self.meta_action_manifest is not None and self.anchor_mode != "discovery":
            raise ValueError("production meta-action context is restricted to discovery teachers")
        return self


class EvalConfig(BaseModel):
    """Evaluation references. Deliberately inside the user scope: the loop cannot mutate
    what it is scored by (the anti-reward-hacking rule that governs tier-3).
    """

    meta_action_db: str | None = None


class PlaygroundConfig(BaseModel):
    """USER-DEFINED. The task; alpaloop must never mutate anything under here."""

    env: EnvConfig = Field(default_factory=EnvConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    # The interpreter over the compiled graph: task definition is user scope (the keyframe
    # task's moments are role-3 external labels). "none" = compile only (discovery).
    task: str = "keyframe_decision"


def _one_of(field: str, allowed: tuple[str, ...]):
    """Reject a value outside a knob's domain.

    These knobs are the alpaloop candidate space, and an optimizer proposing a value no op
    reads would be scored as a no-op edit rather than rejected -- so the domain has to be
    enforced where the config is built, not where it is consumed.
    """

    def check(value: str) -> str:
        if value not in allowed:
            raise ValueError(f"{field} must be one of {allowed}, got {value!r}")
        return value

    return check


class ObserveConfig(BaseModel):
    """Observation policy."""

    windows: str = "maneuver"  # single | tile:<N>s | maneuver
    media: str = "frames"  # frames | video
    max_frames: int | None = Field(default=None, ge=1)
    max_fps: float | None = Field(default=None, gt=0)
    # each round's candidate table lists only segment-overlapping dynamic agents
    # (clutter fix: window-global tables ran 23-108 rows; first step of the
    # orchestrator-managed observe context)
    round_local_context: bool = False
    # overlay marker style: "track_id" (green corner text) or "numbered_pin" (the
    # decluttered form ported from reasoning_agent's rendered_marker_declutter_plan:
    # distinct colors, collision-avoided pins; roster maps pin -> component)
    render_style: str = "track_id"
    # per_tracklet (DEFAULT since 2026-09-02) = a claim citing a tracklet binds to the
    # component that tracklet already has. per_round = mint a component per observation
    # round and let merge:tracklets fold them afterwards, which is what produced 2.3
    # components per grounded object.
    # Measured on the 10-clip screen set, paired per clip against three repeats of
    # per_round: single_component_recall +0.373/+0.412/+0.425 against a between-draw floor
    # of 0.052, components per object 2.3 -> 1.04, with existence and distinctness
    # unchanged. The conflation guard (motion_agreement) moved -0.037/+0.068/+0.078 -- it
    # changes SIGN by baseline, so it is inside its own 0.115 between-draw spread.
    # Recovery is what makes this safe: revise:tracklets splits a component whose claims
    # straddle a break the agent rejected.
    bind: str = "per_tracklet"

    _check_bind = field_validator("bind")(_one_of("bind", ("per_round", "per_tracklet")))
    _check_render_style = field_validator("render_style")(
        _one_of("render_style", ("track_id", "numbered_pin"))
    )
    _check_media = field_validator("media")(_one_of("media", ("frames", "video")))

    @model_validator(mode="after")
    def _check_media_windowing(self) -> ObserveConfig:
        """Video binds claims to the whole span, so it only pairs with ``windows=single``.

        The op enforces this too, but by then a candidate has cost a generation: measured
        2026-08-30, a proposal set media=video under maneuver windows and all ten clips
        died. A cross-knob constraint has to live where the candidate is built.
        """
        if self.media == "video" and self.windows != "single":
            raise ValueError(
                f"media=video binds claims to the whole span, so it requires windows=single, "
                f"got windows={self.windows!r}"
            )
        return self

    @field_validator("windows")
    @classmethod
    def _check_windows(cls, value: str) -> str:
        """``single``, ``maneuver``, or a ``tile:<N>s`` window."""
        if value in ("single", "maneuver") or re.fullmatch(r"tile:\d+(\.\d+)?s", value):
            return value
        raise ValueError(f"windows must be single | maneuver | tile:<N>s, got {value!r}")


class FocusConfig(BaseModel):
    """Adaptive focus policy: frozen round-robin baseline or the agentic planner."""

    mode: Literal["round_robin", "agentic"] = "round_robin"
    max_requests: int = 4  # round_robin request cap
    max_rounds: int = 6  # agentic loop cap (planner turns are budget-exempt middleware)
    planner_prompt: str = "v2"  # pinned prompt version (part of the policy genome)

    @field_validator("mode")
    @classmethod
    def planner_available(cls, value: str) -> str:
        """Reject a planner policy before dispatch when that optional module is absent."""
        if (
            value == "agentic"
            and not (Path(__file__).resolve().parents[2] / "infra/middleware/planner.py").is_file()
        ):
            raise ValueError("agentic focus is not included in this package")
        return value


class DedupConfig(BaseModel):
    """EXPERIMENTAL claim-dedup thresholds (docs/arxiv/claim_dedup_plan.md)."""

    jaccard: float = 0.5
    gap_s: float = 2.0
    # WHAT decides that two claims are the same claim (agent_tracks_plan phase 5). `text`
    # is the validated rule: full-text Jaccard within a component. `identity` drops the
    # SIMILARITY term for claims bound to a tracklet -- the referent is already settled by
    # binding, so two adjacent claims in the same state about the SAME tracklet are the
    # same claim however differently they are worded. The salient-token veto survives in
    # both, because "turning left" and "turning right" share a claim_state.
    # Default `text`: over-merge (lost temporal structure) is the failure mode here and no
    # guard on this repo measures it at ten clips, so identity stays an opt-in the
    # optimizer may select rather than a default nobody validated.
    key: str = "text"

    _key = field_validator("key")(_one_of("dedup.key", ("text", "identity")))


class GenerationConfig(BaseModel):
    """Model generation parameters."""

    max_tokens: int = 8192
    temperature: float = 0.7


class IdentityConfig(BaseModel):
    """Identity-pass behavior knobs."""

    # WHAT the pass adjudicates, as one knob rather than an on/off flag beside a scope:
    #   "claim"     -- every claim event in its own window (Exp A: 34% of claims were
    #                  never verified under one-verdict-per-component; cost 1.53x)
    #   "component" -- one verdict per component
    #   "skip"      -- no identity pass. It is a SELF-CHECK: skipping it costs
    #                  verification, not grounding, so a run can trade it for wall-clock
    #                  (measured 2026-08-25: identity was 996s of a 1150s run, 87%).
    # NOT "off": the --set override parser reads YAML scalars, where off/no/yes are
    # BOOLEANS, so `policy.identity.mode=off` arrives as False and fails validation.
    mode: str = "component"

    @property
    def per_claim(self) -> bool:
        """Whether every claim is adjudicated in its own window."""
        return self.mode == "claim"


class ElementsConfig(BaseModel):
    """Map-element anchoring knobs (tracked_reference_plan phase 6)."""

    # multiplies the range-banded cluster tolerance. That band was derived from 3 clips
    # of one depth substrate; the scale states it is an estimate, not a constant, and
    # lets a different substrate or concept set move it without editing the op.
    tolerance_scale: float = 1.0
    # what to do when perception's track and geometry's position disagree:
    #   "split" -- one component per position-coherent run (geometry decides)
    #   "keep"  -- ONE component holding every sighting of the track (perception decides),
    #              with per-group anchors recorded so the bimodality stays visible
    # Neither is neutral, which is why it is a knob rather than a default buried in code.
    # CAVEAT measured 2026-08-24: 18% of crosswalk sighting pairs have a mask that rides
    # with ego instead of being approached (|dworld|/|dego| = 0.98) -- tracker drift, not
    # anchor noise. "keep" fuses those into one 50m-wide component, so it is unsafe until
    # a drift detector exists. Hence "split" stays the default.
    track_disagreement: str = "split"

    _check_track_disagreement = field_validator("track_disagreement")(
        _one_of("track_disagreement", ("split", "keep"))
    )


class ReviseConfig(BaseModel):
    """The agent's reflect step: whether a geometry-flagged tracklet gets adjudicated.

    ALPALOOP-SEARCHABLE, and deliberately so even though the measurement is unflattering.
    Splitting is what makes bind-at-creation safe, but the judge that adjudicates it
    refuses 27% of genuine same-object pairs (task 3, 2026-09-03), so the op over-splits by
    an amount nobody has bounded. A capability whose right setting depends on a model that
    keeps changing belongs in the candidate space, not in a constant.
    """

    # judged = adjudicate every flagged tracklet and split what the judge rejects;
    # off = record the geometry flag and split nothing
    mode: str = "judged"

    _check_mode = field_validator("mode")(_one_of("revise.mode", ("judged", "off")))


class MergeConfig(BaseModel):
    """Cross-round component merging: how much warrant one tracklet id carries.

    ALPALOOP-SEARCHABLE. `observe:frames` mints a component per round, so one object
    routinely arrives as several; `merge:tracklets` folds them by cited tracklet. The gate
    is the question: the tracklet id is a merge KEY, and `judged` refuses to treat it as a
    WARRANT because a drifting tracklet would fuse claims about different objects.
    Measured 2026-09-01: with `judged`, 41% of tracklets still carry several components
    (mean 1.9 per grounded object, worst case 12) -- so the gate's cost is now visible in
    the benchmark's single_component_rate, and its benefit in distinct_recall, which is
    what makes this searchable rather than a matter of taste.
    """

    # judged = fold only claims the appearance judge matches to the tracklet's crop
    # (abstention merges nothing); trusted = the tracklet id alone warrants the merge;
    # off = no cross-round merging at all
    mode: str = "judged"

    _check_mode = field_validator("mode")(_one_of("merge.mode", ("judged", "trusted", "off")))


class PolicyConfig(BaseModel):
    """ALPALOOP-EVOLVABLE. interface.py exposes exactly this subtree as the candidate
    space -- the config file is the genome boundary.
    """

    observe: ObserveConfig = Field(default_factory=ObserveConfig)
    identity: IdentityConfig = Field(default_factory=IdentityConfig)
    focus: FocusConfig = Field(default_factory=FocusConfig)
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    revise: ReviseConfig = Field(default_factory=ReviseConfig)
    merge: MergeConfig = Field(default_factory=MergeConfig)
    elements: ElementsConfig = Field(default_factory=ElementsConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)


class InfraConfig(BaseModel):
    """Where things run; excluded from both hashes."""

    endpoint_registry: str | None = None
    grounding_registry: str | None = None
    metadb: str | None = None
    data_dir: str | None = None
    # identity-pass bundle concurrency: pure performance (results apply in component
    # order, so the artifact is unchanged) -- hence infra, not policy or playground
    identity_concurrency: int = 1

    # The OPTIMIZER's proposer model, which is infra rather than policy: it writes the
    # candidate edits, never the graph, so no artifact depends on it and the two hashes
    # must not move when it changes. Unset = the local endpoint registry answers, which is
    # the same model the agent itself uses. Set both to point the proposer at an
    # OpenAI-compatible service instead; the key comes from the environment (one of
    # `interface._PROPOSER_KEY_VARS`), never the config, so a campaign YAML stays
    # committable.
    proposer_base_url: str | None = None
    proposer_model: str | None = None


class RunConfig(BaseModel):
    """One run's full configuration across the three ownership scopes."""

    playground: PlaygroundConfig = Field(default_factory=PlaygroundConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    infra: InfraConfig = Field(default_factory=InfraConfig)

    @model_validator(mode="after")
    def optional_passes_available(self) -> RunConfig:
        """Reject requested passes absent from a scoped distribution before paid work."""
        root = Path(__file__).parents[2]
        required = []
        if self.playground.task != "none":
            required.append(("task interpreters", root / "tasks/registry.py"))
        if self.playground.env.reference != "labels" and self.infra.grounding_registry:
            required.append(("reference grounding", root / "streaming_rsg/ops/reference.py"))
            if (Path(self.infra.grounding_registry).parent / "da3_local").is_dir():
                required.append(("geometry grounding", root / "streaming_rsg/ops/geometry.py"))
            if self.policy.identity.mode != "skip":
                required.append(("identity grounding", root / "streaming_rsg/ops/identity.py"))
        for feature, path in required:
            if not path.is_file():
                raise ValueError(f"{feature} is not included in this package")
        return self

    @property
    def playground_id(self) -> str:
        """Hash of the user scope; comparisons are valid only within one."""
        return _scope_hash(self.playground.model_dump())

    @property
    def policy_id(self) -> str:
        """Hash of the evolvable scope; one alpaloop candidate = one policy_id."""
        return _scope_hash(self.policy.model_dump())

    def resolved_dict(self) -> dict:
        """The full resolved config plus both hashes (what run dirs serialize)."""
        return {
            **self.model_dump(),
            "playground_id": self.playground_id,
            "policy_id": self.policy_id,
        }


def _scope_hash(scope: dict) -> str:
    return hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()[:12]


def load_config(path: Path | None = None, overrides: list[str] = ()) -> RunConfig:
    """Load a config file (or pure defaults) and apply dotted ``key=value`` overrides.

    Overrides mutate the raw mapping before validation, so every override lands in the
    resolved (and serialized) config -- reproducibility survives one-off tweaks.
    """
    raw: dict = {}
    if path is not None:
        with Path(path).open() as config_file:
            raw = yaml.safe_load(config_file) or {}
    for override in overrides:
        key, _, value = override.partition("=")
        if not _:
            raise ValueError(f"override {override!r} is not of the form key.path=value")
        node = raw
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = parse_override_value(value)
    return RunConfig.model_validate(raw)


# Words YAML 1.1 reads as booleans that are NOT the literals true/false. A mode named `off`
# (merge.mode, revise.mode) written as `policy.merge.mode=off` arrived here as False and the
# schema rejected it as "not a string" -- twice in the 2026-09-10 optimize runs, each time
# costing a step. The proposer, like a human, means the word.
_YAML_BOOL_WORDS = {"y", "yes", "n", "no", "on", "off"}


def parse_override_value(value: str) -> Any:
    """A dotted override's value as YAML, except that bool-words other than true/false stay text."""
    parsed = yaml.safe_load(value)
    if isinstance(parsed, bool) and value.strip().lower() in _YAML_BOOL_WORDS:
        return value.strip()
    return parsed


def write_resolved(config: RunConfig, out_dir: Path) -> Path:
    """Serialize the fully resolved config beside the run's artifacts."""
    path = out_dir / "config.yaml"
    with path.open("w") as out_file:
        yaml.safe_dump(config.resolved_dict(), out_file, sort_keys=False)
    return path
