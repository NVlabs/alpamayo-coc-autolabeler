#!/usr/bin/env python3
"""Stage 1: validate visual evidence mentioned by a CoC label."""

from __future__ import annotations

import json

from common import NormalizedRow, StageSpec, run_stage

SPEC = StageSpec(
    stage=1,
    name="stage1_coc_evidence",
    default_jsonl="coc_eval_stage1_evidence.jsonl",
    default_json="coc_eval_stage1_evidence.json",
    video_segment="2s_history",
    start_offset_sec=0.0,
    duration_sec=2.0,
)

EVAL_KEYS = (
    "visual_evidence_score",
    "scene_factor_hallucination_score",
    "explanation",
)

TRAFFIC_LIGHT_RULES = """## Traffic-light evaluation rules
Traffic lights can change inside short clips. If the label cites a specific ego-relevant signal color, treat it as supported when that color appears in at least one plausible frame. If signal state is distant, occluded, glare-affected, or outside this two-second window, use an intermediate score instead of an extreme penalty."""

SCENE_FACTOR_RULES = """## Scene-factor evaluation rules
Judge only scene factors that the label relies on for the ego decision.
- Lead or adjacent vehicles: supported if visible in the cited relation.
- Lane, intersection, and road geometry: supported at a coarse level.
- Static obstacles, construction, or parked blockers: supported if visible when cited.
- Do not penalize factors that are not cited by the label.

Score anchors:
- 0.85-1.0: cited critical factors are visible and well supported.
- 0.65-0.80: factors are mostly visible with minor uncertainty.
- 0.45-0.65: mixed visibility or behavior-only label with thin scene evidence.
- <0.45: absent or contradicted critical factors."""

PROMPT_TEMPLATE = """You are an expert driving-scene evaluator. You are watching the first about 2 seconds of an 8-second front-camera driving scene clip. This segment is the context immediately before or around the key decision moment.

## Task
The CoC label below is a free-form description of the ego vehicle's intended behavior and the scene factors that caused it.

Validate the visual grounding of the scene factors mentioned in the label:
1. Identify concrete scene factors mentioned by the label, such as traffic lights, lead vehicles, adjacent vehicles, pedestrians, lanes, intersections, obstacles, speed bumps, or road conditions.
2. Decide whether those factors are visible or strongly supported by this video segment.
3. Penalize hallucinated scene factors that are absent, irrelevant, or contradicted by the video.

## General rules
- Focus on scene evidence, not whether the full ego behavior has completed yet.
- If the label only describes behavior and gives little scene evidence, assign an intermediate visual evidence score and explain the limitation.
- Do not penalize minor wording differences if the referenced scene factor is visible and relevant.
- For non-signal scene factors, penalize only when absent or clearly contradicted.

{scene_factor_rules}

{traffic_light_rules}

## CoC label to evaluate
{label_json}

## Optional metadata
{metadata_json}

## Output
Respond with ONLY one JSON object, no markdown fences, with keys:
- "visual_evidence_score": number 0-1
- "scene_factor_hallucination_score": number 0-1, higher means more absent or contradicted scene factors
- "explanation": short string comparing the visible scene evidence with the label

Use double-quoted keys and valid JSON."""


def build_prompt(row: NormalizedRow) -> str:
    """Build the stage 1 evidence-validation prompt for a record."""
    metadata = {
        "clip_id": row.clip_id,
        "event_timestamp": row.event_timestamp,
        "label_class_identifier": row.label_class_identifier,
    }
    return PROMPT_TEMPLATE.format(
        traffic_light_rules=TRAFFIC_LIGHT_RULES,
        scene_factor_rules=SCENE_FACTOR_RULES,
        label_json=json.dumps(row.coc_label, ensure_ascii=False),
        metadata_json=json.dumps(metadata, ensure_ascii=False),
    )


if __name__ == "__main__":
    raise SystemExit(run_stage(SPEC, build_prompt, EVAL_KEYS))
