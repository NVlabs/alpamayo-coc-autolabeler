# Streaming RSG and CoC labeling

Label a full driving clip with a reasoning scene graph (RSG) and Chain-of-Causation (CoC)
descriptions over time intervals. The graph retains entities, events, relationships and
evidence behind the descriptions.

Start with the sensor workflow below: public Physical AI AV (PAI) video, measured ego motion
and precomputed meta-actions. For video without motion data, use
[camera-only streaming](#camera-only-streaming).

## Table of Contents

- [Installation](#installation)
- [Prepare PAI inputs](#prepare-pai-inputs)
- [Configure credentials and spending](#configure-credentials-and-spending)
- [Generate, resume and export](#generate-resume-and-export)
- [Outputs and troubleshooting](#outputs-and-troubleshooting)
- [Camera-only streaming](#camera-only-streaming)

## Installation

Run commands from the repository root. For PAI preparation and streaming generation:

```bash
python -m pip install '.[pipeline,streaming]'
```

Use the [keyframe CoC guide](keyframe_coc.md#dependencies-build-image-from-dockerfile) for
the dataset/Docker environment. If inputs are already prepared, `.[streaming]` is sufficient.
Install FFmpeg/ffprobe for real video preparation. Hosted inference uses your own API key;
these workflows do not require a separate perception-model service or checkpoint.

Copy the packaged configurations into a writable local directory:

```bash
python - <<'PYTHON'
from importlib.resources import files
from shutil import copytree

copytree(files("rsg_labeling") / "tasks/streaming_rsg/configs", "rsg-config")
PYTHON
```

Use a new destination. The sensor recipe is `wl_readable_context.yaml`; the camera-only
recipe is `video_wl_readable.generation.yaml`. Each has a matching compiler configuration
and registry directory.

## Prepare PAI inputs

Reuse the existing [PAI preparation workflow](keyframe_coc.md#prepare-the-data-physical-ai-av-dataset):

1. Download PAI, build its trajdata cache and generate meta-actions.
2. [Extract front-wide videos](keyframe_coc.md#step-3-generate-coc-labels) with timestamp sidecars.
3. Create a text file containing one selected clip UUID per line.

Existing data/cache/action outputs can be reused directly. The extracted directory must
contain `camera/<clip>.camera_front_wide_120fov.mp4` and the matching
`.camera_front_wide_120fov.timestamps.parquet` sidecar.

```bash
python -m rsg_labeling.preparation.pai \
  --data-dir ./data/physical_ai \
  --cache-dir ./data/trajdata_cache \
  --meta-action-dir ./data/meta_actions/final_outputs \
  --video-dir ./data/extracted_videos \
  --clip-list ./data/clip_ids.txt \
  --endpoint-registry ./rsg-config/wl_readable_context_registry \
  --out ./prepared-pai
```

Preparation makes no model calls and does not regenerate caches, actions or videos. It
verifies clocks and measured motion coverage, then writes `sensor/generation.json` and
`video/generation.json`. The sensor workflow uses video, cached ego motion and existing
actions; it needs no maps, navigation, object labels or human CoC references. Motion is
not padded outside measured coverage. The camera-only configuration uses only the MP4s
and their presentation clocks.

For another dataset, prepare the [sensor input contract](rsg_reference.md#custom-sensor-inputs)
instead of using the PAI adapter. Dataset archives are not read during generation.

## Configure credentials and spending

The public registry uses the OpenAI API. Set your key and a persistent private state
directory; reuse an existing state directory if already configured:

```bash
umask 077
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
export ALPAMAYO_INFERENCE_STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/coc-autolabeling/inference"
mkdir -p "$ALPAMAYO_INFERENCE_STATE_DIR"
chmod 700 "$ALPAMAYO_INFERENCE_STATE_DIR"
```

Edit `rsg-config/wl_readable_context_registry/r0.json`. Its model is `gpt-6.1-sol` with
`request_profile: openai_reasoning` and medium reasoning effort. Replace both `null`
`pricing` rates with positive USD-per-million input/output token rates for your chosen
model. Check [current pricing](https://developers.openai.com/api/docs/models/gpt-6.1-sol);
model access depends on your API account. Keep the registry and key configured for resume,
even if the run has no remaining model requests. Startup validation itself makes no HTTP calls.

In `prepared-pai/sensor/generation.json`, fill these settings before a paid run:

| Setting                           | Value to supply                                                                                                      |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `budget.owner`, `budget.campaign` | Your accounting owner and campaign identifiers.                                                                      |
| `budget.ledger`                   | Absolute path to a dedicated generation ledger shared by this campaign's controllers.                                |
| `budget.accounting_verified_at`   | Timezone-qualified timestamp of verified provider accounting evidence, no more than one hour old at dispatch/resume. |
| `budget.dispatch_envelope_usd`    | The approved maximum allocation for this run.                                                                        |
| `budget.policy`                   | Approved monthly, daily and campaign limits, reserve and large-job threshold; replace the disabled template values.  |
| `allow_paid_dispatch`             | `true` after approving the allocation.                                                                               |

Create output/ledger directories privately. Keep data, credentials, configuration and ledger
outside tracked source. A timestamp is your accounting attestation; the CLI does not query
provider balances. Never invent fresh accounting evidence or retry a recorded exhausted key.

The registry profile uses `max_completion_tokens`, reasoning effort and standard service
tier, and omits temperature. Token-based costs are estimates until confirmed by provider
billing. Missing/unpriced charges stop dispatch unless covered by the configured reserve.
Qualify a small approved pilot, including the 256-token appearance-judge request, before a
large batch; changing provider/model settings can change quality. See the
[cost and recovery reference](rsg_reference.md#cost-records) for receipt fields and reconciliation.

## Generate, resume and export

Run the prepared sensor configuration:

```bash
python -m rsg_labeling.tasks.streaming_rsg.generation run \
  --config ./prepared-pai/sensor/generation.json --out ./rsg-run
```

Resume the same run with unchanged inputs, settings, code and output path:

```bash
python -m rsg_labeling.tasks.streaming_rsg.generation run \
  --config ./prepared-pai/sensor/generation.json --out ./rsg-run --resume
```

For parallel execution, use a separate output directory:

```bash
python -m rsg_labeling.tasks.streaming_rsg.generation.parallel \
  --config ./prepared-pai/sensor/generation.json --out ./rsg-parallel --workers 4
```

Resume a parallel run by repeating its identical parent command, including the worker count;
it has no separate `--resume` flag. Preserve successful records, receipts and the full run
directory. Do not independently resume parent-owned worker shards. For changed code/settings,
larger allocations or unresolved charges, use the [recovery reference](rsg_reference.md#interruption-and-reconciliation).

Export one successfully generated clip using a new destination for each format:

```bash
python -m rsg_labeling.tasks.streaming_rsg.generation export-annotation \
  --publication ./rsg-run/clips --clip CLIP_ID --out ./annotation-CLIP_ID
python -m rsg_labeling.tasks.streaming_rsg.generation export-public \
  --publication ./rsg-run/clips --clip CLIP_ID --out ./public-CLIP_ID
```

Use `./rsg-parallel/clips` for a parallel publication. Run either generation entry point
with `--help` for flags and supported overrides.

## Outputs and troubleshooting

`annotation.json` contains the RSG, interval boundaries, saved CoCs, actions, causes and
evidence. Normal driving has descriptive text by default; numeric interval times remain
in structured fields rather than the CoC sentence. This JSON includes saved compiler/model
text. `--diagnostics` additionally exports prompts, configuration, paths and accounting
traces for internal inspection. A completed run indicates execution success, not verified
annotation accuracy.

The public exporter writes `labels_cot_intervals.parquet` and CSV for annotation-tool
interchange. Both formats require successful generation; preserve the full run for recovery.

| Symptom                                 | Action                                                                                          |
| --------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Unfilled settings or invalid registry   | Fill the named paths, budget fields, credential and token rates.                                |
| Stale accounting evidence               | Obtain current provider evidence for an eligible credential before resume.                      |
| Identity or worker-assignment mismatch  | Restore the original configuration/inputs/worker count, or use explicit new-directory recovery. |
| Unknown charges or exhausted allocation | Retain all receipts and reservations; reconcile or approve recovery before dispatch.            |

Do not clear exhaustion blocks, unknown charges or active locks to force a restart. See the
[technical reference](rsg_reference.md) for custom inputs, accounting and recovery details;
contributor test/build commands are in [packaging and validation](packaging.md#validation).

## Camera-only streaming

Use this workflow for front-facing MP4s without measured ego motion or meta-actions. If
you prepared PAI inputs above, configure the budget in `prepared-pai/video/generation.json`
and use it with the same run/resume/export commands.

For your own videos, edit `rsg-config/video_wl_readable.generation.yaml` so `compiler_config`
and `endpoint_registry` resolve to the absolute paths of its local compiler YAML and
`video_registry` directory. Fill that registry's credential/rates as above, then prepare:

```bash
python -m rsg_labeling.tasks.streaming_rsg.generation prepare-video \
  --config ./rsg-config/video_wl_readable.generation.yaml \
  --videos ./front-wide-mp4s --out ./prepared-video
```

This local command makes no model calls. It writes manifests, `compiler.json` and
`generation.json`; fill the generated configuration's budget and use it for labeling.
Videos need constant frame spacing, at least two captures, gaps no larger than 600 ms,
unambiguous presentation times and a positive final-frame duration. Variable-frame-rate
inputs and clips exceeding configured frame limits are rejected rather than truncated.

Camera-only labeling uses video presentation times and visibly supported motion. It reads
no ego/action/navigation sidecars and does not synthesize measured trajectories or
policy-training targets. Resume and export follow the same workflow as sensor labeling.
