# CoC Video Validation Stages

This directory provides three independent VLM-based validation stages for
existing Chain-of-Causation (CoC) labels produced by the `coc_label_oss`
autolabeler. After a normal CoC labeling run, point the scripts at the
experiment output directory and run stage 1/2/3 to score each sample.

The scripts do not prescribe a filtering policy. Downstream users can choose
their own thresholds or combine the per-stage scores for their sample set.

## Stages

| Stage | Script                             | Video window              | Main scores                                                                                                      |
| ----- | ---------------------------------- | ------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 1     | `eval_stage1_coc_evidence_vllm.py` | first 2s of an 8s context | `visual_evidence_score`, `scene_factor_hallucination_score`                                                      |
| 2     | `eval_stage2_coc_behavior_vllm.py` | last 6s of an 8s context  | `behavior_match_score`, `longitudinal_match_score`, `lateral_match_score`, `behavior_hallucination_score`        |
| 3     | `eval_stage3_coc_causal_vllm.py`   | full 8s context           | `decision_supported_by_video`, `causal_evidence_score`, `causal_consistency_score`, `causal_hallucination_score` |

### Stage 1 scope (cited-factor grounding, not cause coverage)

Stage 1 is intentionally a **precision / grounding check** for scene factors
**cited by the label**, not an overall label-quality or cause-coverage audit.

- It watches only the **first ~2 seconds** of the 8-second context (pre-decision
  window), so the visible evidence is limited compared with the full clip.
- The evaluator asks whether factors the label **mentions** are visible or
  supported, and whether any **cited** factors are absent or contradicted
  (hallucination). It does **not** score omissions of scene factors the label
  failed to mention.
- In practice, scoring every object or factor visible in the clip tends to
  **over-penalize** labels that focus on the ego-relevant subset; restricting
  Stage 1 to cited factors gives more stable evidence scores.
- **Cause coverage and causal plausibility** are handled later: Stage 3 uses the
  full 8-second clip to judge whether cited evidence supports the labeled
  behavior and whether the causal story is consistent.

If you need explicit omission / completeness metrics, treat them as a separate
concern (e.g. a future stage or an extension to Stage 3), not mixed into this
2-second cited-evidence check.

## Primary Input: CoC Autolabeler Output

The recommended input is the autolabeler experiment directory:

```text
experiments/<run_id>_<exp_name>/
├── <clip_id>/
│   ├── cot_<event_start_timestamp>.yaml
│   └── ...
└── ...
```

Each `cot_<timestamp>.yaml` file is read directly. No merge step, parquet
conversion, or human-golden preprocessing is required.

The loader extracts:

- `clip_id` from the parent directory name
- `event_start_timestamp` from the YAML payload or `cot_<timestamp>.yaml` filename
- `effect_on_ego_behavior` from `final_content.ego_behavior_schema.effect_on_ego_behavior`
- optional `label_class_identifier`
- optional camera name from the saved prompt payload

## Video Sources

The scripts support two video layouts.

### 1. Saved 8-second segment videos from autolabeling (recommended)

If CoC labeling was run with:

```yaml
data_loader.video.save_segment_videos: true
data_loader.video.segment_video_output_dir: ./experiments/video_segment
```

the validation scripts auto-detect segment videos at:

```text
experiments/video_segment/<clip_id>/<clip_id>_<event_start_timestamp>.mp4
```

In this mode the scripts default to `--video-mode segment_8s`.

### 2. Raw full camera clips

If segment videos were not saved, pass `--video-root` pointing to the PAI camera
root used during autolabeling. The scripts use the relative
`event_start_timestamp` from each `cot_*.yaml` file and cut the 8-second
decision window from the raw clip.

## Model Backends

The evaluator uses the same `model_name` aliases as CoC autolabeling:

- `qwen3_vl_235b_awq`
- `qwen3.5_35b`
- `qwen3.5_397b_fp8`
- `gpt5`
- `gpt5.5`

Pass the alias with `--model-name`. The scripts auto-detect credentials and API
endpoints using the same precedence as the autolabeler:

1. NVIDIA-hosted Azure OpenAI (`NVHOST_OAI_CLIENT_ID` +
   `NVHOST_OAI_CLIENT_SECRET`)
2. NVIDIA inference (`NVIDIA_API_KEY`)
3. Standard OpenAI (`OPENAI_API_KEY`)
4. Local Qwen via an OpenAI-compatible vLLM server (`LOCAL_OPENAI_BASE_URL` +
   `LOCAL_OPENAI_API_KEY`)

For label quality, the recommended evaluator models are `gpt5.5` and
`qwen3.5_397b_fp8`. For local-only runs, start with `qwen3.5_397b_fp8` when
the machine has enough GPU memory; otherwise use `qwen3.5_35b`.

### Cloud GPT (`gpt5`, `gpt5.5`)

NVIDIA-hosted Azure OpenAI:

```bash
export NVHOST_OAI_CLIENT_ID="your_client_id"
export NVHOST_OAI_CLIENT_SECRET="your_client_secret"
```

NVIDIA inference:

```bash
export NVIDIA_API_KEY="nvapi-your-nvidia-api-key"
```

Standard OpenAI:

```bash
export OPENAI_API_KEY="sk-your-openai-api-key"
```

Hosted GPT endpoints may reject explicit sampling parameters. The scripts omit
`temperature` by default for `gpt5` and `gpt5.5`.

### Local Qwen

Start an OpenAI-compatible vLLM server for the selected Qwen model, then point
the validation scripts at it:

```bash
export HF_TOKEN="hf_yourtoken"
hf auth login --token "$HF_TOKEN"

export LOCAL_OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export LOCAL_OPENAI_API_KEY="EMPTY"
```

For Qwen models, the scripts default to `--temperature 0.0` and disable
thinking mode in the chat template.

Optional overrides:

- `--provider-model`: provider-side model ID (similar to `NV_INFERENCE_MODEL` or
  `LOCAL_OPENAI_MODEL`)
- `--api-base`: force a specific API base URL instead of auto-detection
- `--api-key-env`: read the API key from a custom environment variable name

## Examples

### Cloud GPT-5.5 on NVIDIA inference

```bash
cd projects/coc_label_oss/scripts/coc_validation_stage

export NVIDIA_API_KEY="nvapi-your-nvidia-api-key"
COC_RESULT=/path/to/experiments/20260609_105441_pai_smoke_gpt55
VIDEO_ROOT=/path/to/extracted_pai_videos

COMMON=(
  --coc-result-dir "$COC_RESULT"
  --video-root "$VIDEO_ROOT"
  --model-name gpt5.5
  --resume
)

python3 eval_stage1_coc_evidence_vllm.py "${COMMON[@]}" \
  --jsonl-out stage1_evidence.jsonl --json-out stage1_evidence.json

python3 eval_stage2_coc_behavior_vllm.py "${COMMON[@]}" \
  --jsonl-out stage2_behavior.jsonl --json-out stage2_behavior.json

python3 eval_stage3_coc_causal_vllm.py "${COMMON[@]}" \
  --jsonl-out stage3_causal.jsonl --json-out stage3_causal.json
```

If segment videos already exist under `experiments/video_segment`, you can omit
`--video-root`. To point to a non-default segment directory, pass
`--segment-video-root`.

### Local Qwen3.5-35B via vLLM

```bash
export LOCAL_OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export LOCAL_OPENAI_API_KEY="EMPTY"

COMMON=(
  --coc-result-dir "$COC_RESULT"
  --model-name qwen3.5_35b
  --resume
)

python3 eval_stage1_coc_evidence_vllm.py "${COMMON[@]}"
```

### Dry run (no model calls)

```bash
python3 eval_stage1_coc_evidence_vllm.py \
  --coc-result-dir "$COC_RESULT" \
  --video-root "$VIDEO_ROOT" \
  --dry-run \
  --limit 2
```

## Optional Legacy Table Input

`--input-table` is still supported for CSV/JSON/JSONL/parquet tables, but the
recommended workflow is `--coc-result-dir`.

## Output

Each stage writes append-only JSONL plus a finalized JSON array:

```text
stage1_evidence.jsonl / stage1_evidence.json
stage2_behavior.jsonl / stage2_behavior.json
stage3_causal.jsonl / stage3_causal.json
```

Each row includes:

- `clip_id`
- `event_timestamp`
- `yaml_path`
- `video_segment`
- `source_video_path`
- `coc_label`
- `evaluator_model_name`
- `evaluator_backend`
- `evaluation`
- `model_response_raw`
- `error`, if processing failed

Use `--resume` to skip rows already present in the JSONL file.

## Troubleshooting

**Cloud GPT request fails with temperature rejection**

Use `--model-name gpt5.5` (or `gpt5`) and let the scripts omit temperature by
default. Do not pass `--temperature` unless you know the endpoint accepts it.

**Local Qwen connection error**

Confirm `LOCAL_OPENAI_BASE_URL` points to a running vLLM OpenAI-compatible
server and that the served model ID matches `--provider-model` or the default
Hugging Face model ID for the selected `--model-name`.
