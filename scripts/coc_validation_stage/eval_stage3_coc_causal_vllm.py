#!/usr/bin/env python3
"""Stage 3: validate causal consistency of a CoC label."""

from __future__ import annotations

import json

from common import NormalizedRow, StageSpec, run_stage

SPEC = StageSpec(
    stage=3,
    name="stage3_coc_causal",
    default_jsonl="coc_eval_stage3_causal.jsonl",
    default_json="coc_eval_stage3_causal.json",
    video_segment="8s_full",
    start_offset_sec=0.0,
    duration_sec=8.0,
)

EVAL_KEYS = (
    "decision_supported_by_video",
    "causal_evidence_score",
    "causal_consistency_score",
    "causal_hallucination_score",
    "explanation",
)

TRAFFIC_LIGHT_RULES = """## Traffic-light causal rules
- Lights may switch during the 8-second clip; do not require one color to persist for the whole video.
- If the label cites red, yellow, or green for the ego-relevant signal, treat it as supported when that color appears on the relevant signal in at least one plausible frame.
- Do not treat one conflicting transition frame as a full contradiction when another frame supports the labeled state.
- If signal color is uncertain, use intermediate scores around 0.6 rather than harsh penalties."""

PROMPT_TEMPLATE = """You are an expert driving-scene evaluator. You are watching a full about 8-second front-camera driving scene clip.

## What you are judging
Judge whether the label's causal story matches the video: behavior support, visible causes, and whether those causes explain the behavior.

## Task
1. Identify the ego's dominant observed behavior across the clip.
2. Identify cited scene factors and whether they are visible and relevant.
3. Decide whether the causal link is plausible.

## Score each field independently
Use the full 0-1 range. Do not cluster every sample above 0.75.

### decision_supported_by_video
Does the labeled ego behavior match what happens in the video?
- 0.85-1.0: clear match.
- 0.60-0.80: mostly matches with minor specificity differences.
- 0.40-0.55: partial overlap or over-specific wording without support.
- 0.10-0.35: major behavior contradiction.

### causal_evidence_score
Are the cited scene factors visible and relevant? Score the factors themselves, not whether they justify the behavior.

### causal_consistency_score
Do the cited factors plausibly explain the labeled behavior?
- 0.85-1.0: strong causal link.
- 0.55-0.75: plausible but somewhat generic or over-specific.
- 0.35-0.50: weak link, extra causal claims, or behavior-scene mismatch.
- <0.35: wrong, absent, or irrelevant cause.

### causal_hallucination_score
High when the explanation depends on absent, wrong, or irrelevant causes.
- Wrong traffic-light color, wrong object, or wrong lane/intersection context: 0.55-0.85.
- Mild over-specificity with visible causes: 0.20-0.40.
- Good support: 0.05-0.20.

## Hard rules
1. If the labeled behavior is clearly contradicted by the video, set decision_supported_by_video <= 0.40 and causal_consistency_score <= 0.45.
2. If cited scene factors are visible but do not explain the labeled maneuver, causal_consistency_score <= 0.55.
3. Gap-search or planned lane-change wording without visible preparation should receive moderate causal_consistency_score, not >= 0.85.
4. If the label cites a maneuver-causing object that is absent or wrong, causal_hallucination_score >= 0.50.

{traffic_light_rules}

## CoC label to evaluate
{label_json}

## Optional metadata
{metadata_json}

## Output
Respond with ONLY one JSON object, no markdown fences, with keys:
- "decision_supported_by_video": number 0-1
- "causal_evidence_score": number 0-1
- "causal_consistency_score": number 0-1
- "causal_hallucination_score": number 0-1, higher means the causal explanation depends on absent, irrelevant, or contradicted evidence
- "explanation": short string explaining the ego behavior, the cited evidence, and the causal link

Use double-quoted keys and valid JSON."""


def build_prompt(row: NormalizedRow) -> str:
    """Build the stage 3 causal-consistency prompt for a record."""
    metadata = {
        "clip_id": row.clip_id,
        "event_timestamp": row.event_timestamp,
        "label_class_identifier": row.label_class_identifier,
    }
    return PROMPT_TEMPLATE.format(
        traffic_light_rules=TRAFFIC_LIGHT_RULES,
        label_json=json.dumps(row.coc_label, ensure_ascii=False),
        metadata_json=json.dumps(metadata, ensure_ascii=False),
    )


if __name__ == "__main__":
    raise SystemExit(run_stage(SPEC, build_prompt, EVAL_KEYS))
