#!/usr/bin/env python3
"""Stage 2: validate ego behavior described by a CoC label."""

from __future__ import annotations

import json
import sys
from dataclasses import replace

from common import NormalizedRow, StageSpec, run_stage

SPEC = StageSpec(
    stage=2,
    name="stage2_coc_behavior",
    default_jsonl="coc_eval_stage2_behavior.jsonl",
    default_json="coc_eval_stage2_behavior.json",
    video_segment="6s_future",
    start_offset_sec=2.0,
    duration_sec=6.0,
)

EVAL_KEYS = (
    "behavior_match_score",
    "longitudinal_match_score",
    "lateral_match_score",
    "behavior_hallucination_score",
    "explanation",
)

BEHAVIOR_TAXONOMY = """## Ego behavior taxonomy
LONGITUDINAL, choose the dominant one:
- Stop: decelerate to and hold at a stop/yield line or control point.
- Yield: give way to pedestrian, cross-traffic, cyclist, or emergency vehicle.
- Pass/Overtake: accelerate to pass a moving vehicle in the traffic stream.
- Gap-search: modulate speed to create or wait for a gap for merge, lane change, or nudge.
- Keep Distance: follow a lead agent and maintain safe headway.
- Adapt Speed: slow for curvature, ramp, roundabout, intersection, or speed bump.
- Resume Speed: return toward cruise speed when no stronger behavior dominates.

LATERAL, choose the dominant one and direction when identifiable:
- Lane Change: sustained transition into an adjacent lane.
- Nudge: temporary small departure for clearance, then remain on the same route lane.
- Turn: route-following turn at an intersection, roundabout, or U-turn.
- Merge: enter mainline from ramp or weaving segment.
- Split: leave mainline toward ramp or weaving segment.
- Keep Lane: stay centered in the current lane with no meaningful lateral maneuver."""

PROMPT_TEMPLATE = """You are an expert driving-scene evaluator. You are watching the last about 6 seconds of an 8-second front-camera clip. This segment starts at the key decision moment and shows what the ego vehicle does afterward.

## What you are judging
Judge only whether this CoC label accurately describes what the ego vehicle does in the video. Do not compare to a human annotator or another model.

## Task
1. From the video alone, identify the dominant observed longitudinal and lateral behaviors using the taxonomy below.
2. Parse the label into its dominant longitudinal claim and lateral claim.
3. Decide whether the label's primary combined maneuver matches the observed maneuver.
4. Assign dimension scores and an overall behavior_match_score.

{behavior_taxonomy}

## Primary maneuver rule
The label has a primary maneuver: its main longitudinal plus lateral combination. behavior_match_score is not an average. It is low when the label's primary maneuver does not match the video.

Lower behavior_match_score when:
- The video's dominant maneuver type differs from the label's.
- The video shows keep lane but the label claims an executed lane change, merge, nudge, or turn without visible support.
- The label emphasizes stop/yield but the ego clearly proceeds, or the label proceeds but the ego clearly stops.
- The label's lateral direction conflicts with the observed direction.

## Dimension scoring
Longitudinal:
- 0.90-1.0: same specific behavior clearly observed.
- 0.70-0.85: compatible but less specific.
- 0.45-0.65: related but different.
- 0.10-0.35: contradicted.
- about 0.60: uncertain.

Lateral:
- 0.90-1.0: clear lane change, nudge, turn, merge, or split in the stated direction.
- 0.65-0.85: partial preparation or early maneuver onset visible.
- 0.40-0.65: label asserts lateral intent or plan, but video mostly shows keep lane.
- 0.10-0.35: label asserts completed lateral maneuver, video clearly contradicts.
- about 0.60: ambiguous.

Preparation-only lane-change wording without visible preparation should receive moderate, not extreme, lateral and behavior scores.

## CoC label to evaluate
{label_json}

## Optional metadata
{metadata_json}

## Output
Respond with ONLY one JSON object, no markdown fences, with keys:
- "behavior_match_score": number 0-1
- "longitudinal_match_score": number 0-1
- "lateral_match_score": number 0-1
- "behavior_hallucination_score": number 0-1, higher means label behavior is contradicted or unsupported
- "explanation": "Observed longitudinal: <taxonomy>. Observed lateral: <taxonomy>. Label primary maneuver: <...>. Comparison: <...>"

Use double-quoted keys and valid JSON."""

PROMPT_TEMPLATE_FULL_8S = """You are an expert driving-scene evaluator. You are watching the full about 8-second front-camera clip centered on the key decision moment.

Judge whether the CoC label accurately describes what the ego vehicle does in the video. Use the same taxonomy and scoring anchors as the 6-second evaluator.

{behavior_taxonomy}

Rules:
- behavior_match_score is low when the label's primary combined maneuver does not match the observed maneuver.
- Executed lateral claims require visible support.
- If the video shows a turn, lane change, merge, split, or nudge, do not score high lateral_match for keep-lane-only labels.
- Consider the full 8-second clip, including late-onset maneuvers.

## CoC label to evaluate
{label_json}

## Optional metadata
{metadata_json}

## Output
Respond with ONLY one JSON object, no markdown fences, with keys:
- "behavior_match_score": number 0-1
- "longitudinal_match_score": number 0-1
- "lateral_match_score": number 0-1
- "behavior_hallucination_score": number 0-1
- "explanation": "Observed longitudinal: <taxonomy>. Observed lateral: <taxonomy>. Label primary maneuver: <...>. Comparison: <...>"

Use double-quoted keys and valid JSON."""


def build_prompt(row: NormalizedRow, *, template: str = PROMPT_TEMPLATE) -> str:
    """Build the stage 2 behavior-validation prompt for a record."""
    metadata = {
        "clip_id": row.clip_id,
        "event_timestamp": row.event_timestamp,
        "label_class_identifier": row.label_class_identifier,
    }
    return template.format(
        behavior_taxonomy=BEHAVIOR_TAXONOMY,
        label_json=json.dumps(row.coc_label, ensure_ascii=False),
        metadata_json=json.dumps(metadata, ensure_ascii=False),
    )


def main() -> int:
    """Run the stage 2 behavior validation."""
    argv = sys.argv[1:]
    use_full_8s = "--full-8s-window" in argv
    if use_full_8s:
        argv = [arg for arg in argv if arg != "--full-8s-window"]

    spec = SPEC
    template = PROMPT_TEMPLATE
    if use_full_8s:
        spec = replace(SPEC, video_segment="8s_full", start_offset_sec=0.0, duration_sec=8.0)
        template = PROMPT_TEMPLATE_FULL_8S

    return run_stage(spec, lambda row: build_prompt(row, template=template), EVAL_KEYS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
