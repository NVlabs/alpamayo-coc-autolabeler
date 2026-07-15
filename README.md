# Chain-of-Causation (CoC) Autolabeling Pipeline

This repository provides an autolabeling pipeline for autonomous-driving
scenarios. It derives meta-actions, which are high-level categorical
descriptions of ego motion, and chain-of-causation labels, which connect causal
factors in the scene to the ego vehicle's intended behavior.

## Support

📣 **Usage questions and discussion**: please join us on the
[Alpamayo NV Developer Forum](https://forums.developer.nvidia.com/c/autonomous-vehicles/alpamayo/766).

🐛 **Code-level bugs, documentation issues, and feature requests**: file a
[GitHub issue](../../issues/new/choose) using the appropriate template (Bug
report, Documentation request, or Feature request). The relevant NVIDIA
responder is auto-assigned via the `assignees:` field on the template.

🚨 **Security vulnerabilities**: please use
[NVIDIA's Vulnerability Disclosure Program](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail).
Do not file security issues publicly here.

## Table of Contents

- [Support](#support)
- [Workflow Overview](#workflow-overview)
- [Paper](#paper)
- [Runtime Requirements](#runtime-requirements)
  - [Validated Configurations](#validated-configurations)
- [Dependencies (Build Image from Dockerfile)](#dependencies-build-image-from-dockerfile)
- [Prepare the Data (Physical-AI AV Dataset)](#prepare-the-data-physical-ai-av-dataset)
  - [Step 1: Generate Meta-Actions](#step-1-generate-meta-actions)
  - [Step 2: Identify Keyframes](#step-2-identify-keyframes)
- [Run CoC Autolabeling](#run-coc-autolabeling)
  - [Step 3: Generate CoC Labels](#step-3-generate-coc-labels)
  - [Output Structure](#output-structure)
- [Step 4: Validate CoC Labels](#step-4-validate-coc-labels)
- [Troubleshooting](#troubleshooting)
  - [1. vLLM Import Fails With `libtorch_cuda.so`](#1-vllm-import-fails-with-libtorch_cudaso)
  - [2. Local Qwen Fails With Unsupported CUDA/PTX](#2-local-qwen-fails-with-unsupported-cudaptx)
  - [3. Cloud GPT Request Fails With `invalid_request_error`](#3-cloud-gpt-request-fails-with-invalid_request_error)
- [Extend with Other Model Clients](#extend-with-other-model-clients)
  - [1) Add or reuse a wrapper class](#1-add-or-reuse-a-wrapper-class)
  - [2) Register the model key in the wrapper factory](#2-register-the-model-key-in-the-wrapper-factory)
  - [3) (Optional) Export the wrapper in package init](#3-optional-export-the-wrapper-in-package-init)
  - [4) Add or update runtime configs](#4-add-or-update-runtime-configs)
  - [5) Validate with a smoke run](#5-validate-with-a-smoke-run)
- [Disclaimer](#disclaimer)
- [License](#license)
- [Citation](#citation)

## Workflow Overview

1. **Step 1: Generate Meta-Actions**: produce per-clip high-level motion labels
   from trajectory data.
2. **Step 2: Identify Keyframes**: select frames where ego meta-actions change,
   since these transitions are likely to contain decision-making context.
3. **Step 3: Generate CoC Labels**: run the VLM pipeline on selected keyframes
   to produce chain-of-causation labels.
4. **Step 4: Validate CoC Labels**: run the optional VLM-based validation
   stages to identify potentially lower-quality labels prior to downstream use.

## Paper

This autolabeling pipeline is related to the Chain-of-Causation reasoning
pipeline described in
[Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable
Autonomous Driving in the Long Tail](https://arxiv.org/abs/2511.00088).

## Runtime Requirements

The minimum hardware requirement depends on the model backend and dataset size.
GPU is not required when CoC labels are generated with a hosted model API such
as `gpt5` or `gpt5.5`; in that setup, local compute is mainly used for data
loading, meta-action generation, keyframe selection, and video extraction.

Local Qwen inference requires GPU capacity sufficient for the selected Qwen
model. The released Qwen model examples have been tested on NVIDIA A100 and
H100 GPUs with the CUDA 12.8 Docker image; other GPU setups may work depending
on the selected Qwen model, available VRAM, batch size, driver compatibility,
and worker settings. CPU and host-memory usage also scale with the number of
workers used for trajectory-data caching and meta-action labeling.

### Validated Configurations

Use `nvidia-smi` on the host to confirm the NVIDIA driver before running local
Qwen inference. The Docker image uses CUDA 12.8 and vLLM 0.17.1.

| GPU generation | Example GPUs | Minimum host NVIDIA driver | Validation status |
| -------------- | ------------ | -------------------------- | ----------------- |
| Ampere         | A100         | `>=535`                    | Tested            |
| Hopper         | H100         | `>=545`                    | Tested            |

For a small smoke test of about 100 clips with hosted model API for CoC
generation, the following setup is sufficient:

- 8 CPU cores
- 8 GB memory
- no GPU, unless running a local Qwen model

## Dependencies (Build Image from Dockerfile)

A standalone image can be built from this repository's `Dockerfile`.

Build:

```bash
docker build -t coc_auto_labeling:latest .
```

Run:

```bash
docker run --gpus all -it --ipc=host \
  -v path/to/coc_label_oss:/workspace/coc_auto_labeling \
  coc_auto_labeling:latest
```

The default in-container project path is `/workspace/coc_auto_labeling`.

For local Qwen inference, verify the vLLM import before starting a labeling
run:

```bash
docker run --gpus all coc_auto_labeling:latest \
  python -c 'from vllm import LLM; print("ok")'
```

## Prepare the Data (Physical-AI AV Dataset)

To standardize trajectory data formats and support reindexing and
interpolation, this pipeline leverages
[trajdata](https://github.com/NVlabs/trajdata) for data formatting under the
hood.

Download the Physical-AI AV dataset once from
`https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles/tree/main`.
The same dataset root is used for meta-action labeling, keyframe selection,
video extraction, and CoC labeling.

### Step 1: Generate Meta-Actions

Run meta-action autolabeling to produce per-clip ego-motion labels:

```bash
meta-action-autolabel \
  --dataset_name pai \
  --meta_action_names all_ego \
  --data_dir /path/to/physical_ai_data \
  --cache_dir /path/to/trajdata_cache \
  --save_dir /path/to/meta_action/resultdir \
  --num_workers 8
```

Common options:

- `--meta_action_names`: meta-action types to generate. The CLI default is
  `go_straight`; the example uses `all_ego` to generate the default ego-action
  set.
- `--scene_list`: optional path to a text file with one clip ID per line. Use
  this for small subset or sample-eval runs.
- `--num_workers`: worker count for data loading and clip processing. The CLI
  default is `32`; the example uses `8` as a conservative sample-run setting.

Key outputs:

- `--cache_dir`: formatted trajectory data cache.
- `--save_dir`: final per-clip meta-action text outputs. Use
  this path as `meta_action_dir` in later steps, for example
  `/path/to/meta_action/resultdir/final_outputs`.

For details on running meta-action autolabeling, including dataset input
layout, output format, worker guidance, and visualization commands, see
[`docs/meta_action_autolabel.md`](docs/meta_action_autolabel.md).

### Step 2: Identify Keyframes

Use meta-action transitions to generate relative keyframe timestamps:

```bash
python -m coc_labeling.keyframe_auto_select \
  --meta_action_dir /path/to/meta_action/resultdir/final_outputs \
  --output_dir ./experiments/keyframes
```

Arguments:

- `--meta_action_dir`: folder containing meta-action text outputs from Step 1.
  This should point to the `final_outputs` directory.
- `--min_duration`: minimum action span duration in frames. The default is
  `10`, which keeps brief maneuvers such as short strong-deceleration events in
  sample-eval runs.
- `--target_count`: maximum number of segments to keep per action type after
  balancing. The default is `500000` for large-scale experiments, so small
  samples usually keep every matching segment.
- `--output_dir`: folder where the generated keyframe index files are written.

This script automatically generates relative keyframe timestamps from
meta-action outputs by selecting frames where ego meta-actions change, as these
transitions are more likely to indicate ego decision-making moments.

The output keyframes will be stored at:

`./experiments/keyframes/segments_relative_timestamp_sampled.json`

The structure is:

```json
{
  "<meta_action_1>": [
    {
      "meta_action": "<meta_action_1>",
      "clip_id": "<clip_uuid>",
      "event_start_frame": <start_frame_index>,
      "event_end_frame": <end_frame_index>,
      "duration": <num_frames>
    },
    "... additional entries ..."
  ],
  "<meta_action_2>": [
    {
      "meta_action": "<meta_action_2>",
      "clip_id": "<clip_uuid>",
      "event_start_frame": <start_frame_index>,
      "event_end_frame": <end_frame_index>,
      "duration": <num_frames>
    }
  ],
  "... additional meta_action keys ...": [
    "... additional entries ..."
  ]
}
```

## Run CoC Autolabeling

### Step 3: Generate CoC Labels

Before running CoC labeling, confirm the following inputs. Configure dataset
paths in `src/coc_labeling/config/data/base.yaml`.

1. `data_dir`: root folder that contains parquet clip data, for example, `/path/to/physical_ai_data`
2. `cache_dir`: formatted trajectory data cache.
3. `meta_action_dir`: meta-action outputs, for example:
   `/path/to/meta_action/final_outputs`
4. Keyframe input: the `segment_config_path` can be set in the config `src/coc_labeling/config/data/keyframe_rel_ts.yaml`,
   the `segment_generator_type` and `meta_action_filter` can be set in the config `src/coc_labeling/config/base_config_vlm_rel_ts.yaml`.
5. `video_dir`: root folder that contains a `camera` subfolder for raw AV videos, for example:
   `/path/to/extracted_pai_videos`

If `video_dir` is not already populated, you can extract only the clips
referenced by your keyframe/index JSON. Set `EXTRACTED_VIDEO_ROOT` to the path
you want to use for `video_dir`; the extracted MP4 files will be saved under
its `camera` subfolder. `VIDEO_ZIP_DIR` should point to the folder containing
the PAI camera chunk zips, for example `camera_front_wide_120fov.chunk_0000.zip`,
`camera_front_wide_120fov.chunk_0001.zip`, and so on.

```bash
export SEGMENT_INDEX=./experiments/keyframes/segments_relative_timestamp_sampled.json
export VIDEO_ZIP_DIR=/path/to/physical_ai_data/camera/camera_front_wide_120fov
export EXTRACTED_VIDEO_ROOT=/path/to/extracted_pai_videos

python scripts/extract_pai_videos_from_index.py \
  --index-file "${SEGMENT_INDEX}" \
  --video-zip-dir "${VIDEO_ZIP_DIR}" \
  --output-video-root "${EXTRACTED_VIDEO_ROOT}"
```

Variables and arguments:

- `SEGMENT_INDEX`: keyframe/index JSON generated in the previous step.
- `VIDEO_ZIP_DIR`: folder containing the PAI camera chunk zip files.
- `EXTRACTED_VIDEO_ROOT`: output root used later as `video_dir`.
- `--index-file`: path to the keyframe/index JSON that lists clips to extract.
- `--video-zip-dir`: path to the source camera zip folder.
- `--output-video-root`: root directory where extracted videos are written under
  a `camera` subfolder.
- `--meta-action-filter`: optional filter that extracts only clips matching the
  requested meta-action type.

Without `--meta-action-filter`, the script extracts all clips referenced by the
index file. If your run uses a meta-action filter, pass the same filter to the
extraction script, for example `--meta-action-filter go_straight`.

Example video structure:

```text
/path/to/extracted_pai_videos/camera
├── 01d55181-c15d-49f2-8b52-0ddf141375d0.camera_front_wide_120fov.mp4
├── 5b530101-f63b-4c61-aeac-178ad1626774.camera_front_wide_120fov.mp4
└── ...
```

The released framework currently supports a VLM labeling agent.

Supported `model_name` values in the current release:

- `qwen3_vl_235b_awq`
- `qwen3.5_35b`
- `qwen3.5_397b_fp8`
- `gpt5`
- `gpt5.5`

For label quality, the recommended `model_name` values are `gpt5.5` and
`qwen3.5_397b_fp8`. For local-only runs, start with `qwen3.5_397b_fp8` when
the machine has enough GPU memory; otherwise use `qwen3.5_35b`. Use
`qwen3.5_35b` for local smoke tests because it is faster to download and load
than the larger local Qwen variants.

For Qwen models, authenticate with Hugging Face:

```bash
export HF_TOKEN="hf_yourtoken"
hf auth login --token "$HF_TOKEN"
```

This is required because the Qwen models are hosted on Hugging Face. `HF_TOKEN`
is your Hugging Face access token, and `--token` passes it to the login command.

For `gpt5` or `gpt5.5`, configure one of the following credential sets.

Standard OpenAI API credentials:

```bash
export OPENAI_API_KEY="sk-your-openai-api-key"
```

NVIDIA inference credentials from inference.nvidia.com/build.nvidia.com:

```bash
export NVIDIA_API_KEY="nvapi-your-nvidia-api-key"
```

NVIDIA-hosted Azure OpenAI credentials:

```bash
export NVHOST_OAI_CLIENT_ID="your_client_id"
export NVHOST_OAI_CLIENT_SECRET="your_client_secret"
```

Credential precedence is NVIDIA-hosted Azure OpenAI first, then NVIDIA
inference, then standard OpenAI.

For Hugging Face local cache, optionally set:

```bash
export MODEL_CACHE_DIR=/path/to/hf-cache   #examples, /workspace/hf-cache
# or
export HF_HOME=/path/to/hf-cache
```

Run CoC autolabeling. For example:

```bash
export MODEL_CACHE_DIR=/workspace/hf-cache
python -m coc_labeling.data_labeling \
  --config-name=base_config_vlm_rel_ts \
  model_name=qwen3.5_35b \
  resume_exp_dir=null \
  exp_name=qwen3.5_35b_test \
  data_loader.keyframe.meta_action_filter='[go_straight]' \
  data_loader.video.save_segment_videos=false
```

Variables and arguments:

- `MODEL_CACHE_DIR`: local Hugging Face model cache used by local Qwen model
  loading.
- `--config-name`: Hydra config preset. `base_config_vlm_rel_ts` runs VLM
  labeling with relative keyframe timestamps.
- `model_name`: model backend to use for CoC generation. Use one of the
  supported `model_name` values listed above.
- `vlm_agent.temperature`, `vlm_agent.top_p`, `vlm_agent.repetition_penalty`:
  local Qwen sampling parameters. Defaults are `0.0`, `1.0`, and `1.0`.
- `resume_exp_dir`: existing experiment directory to resume from. Use `null` to
  create a new experiment directory.
- `exp_name`: readable suffix for the output experiment folder.
- `data_loader.keyframe.meta_action_filter`: list of meta-action types to label.
  Use `null` to include all available meta-action types.
- `data_loader.video.save_segment_videos`: whether to save extracted segment
  videos alongside CoC outputs.

You can configure `meta_action_filter`, `save_segment_videos`, `model_name`, and related settings in:
`src/coc_labeling/config/base_config_vlm_rel_ts.yaml`

More examples for running CoC autolabeling.

```bash
python -m coc_labeling.data_labeling \
  --config-name=base_config_vlm_rel_ts \
  model_name=qwen3.5_397b_fp8 \
  resume_exp_dir=null \
  exp_name=qwen3.5_397b_fp8_test \
  data_loader.keyframe.meta_action_filter=null \
  data_loader.video.save_segment_videos=false
```

This example uses the same arguments as above. It switches to
`qwen3.5_397b_fp8` and sets `data_loader.keyframe.meta_action_filter=null` to
process all available meta-action types.

For local model inference, if you run into out-of-memory errors, use a GPU or
machine with more available GPU memory, or reduce parallelism.

### Output Structure

Experiment outputs are written under:

```text
path/to/coc_label_oss/experiments/<run_id>_<exp_name>/
```

Example:

```text
path/to/coc_label_oss/experiments/20260308_002355_qwen3.5_35b_test
├── b6700354-ab89-45ec-8b47-7d6dbfe16b1a/
│   ├── cot_12600000.yaml
│   ├── ...
└── ...
```

`cot_<keyframe_timestamp>.yaml` uses the keyframe timestamp in the filename.
In this example, `12600000` is the keyframe's relative timestamp.

Inside each YAML file, the `results` structure looks like:

```yaml
results:
  event_start_frame: 126
  event_start_timestamp: 12600000
  final_content:
    ego_behavior_schema:
      effect_on_ego_behavior: "Keep distance to the lead vehicle by decelerating."
  # prompt: [...]  # full model input (system/user text + sampled video frame references)
```

Field meanings:

- `event_start_timestamp`: relative keyframe timestamp in microseconds (also used in `cot_<timestamp>.yaml` filename).
- `event_start_frame`: relative frame index in the sampled clip timeline. With default `10` FPS, `126` means `12.6s`.
- `final_content`: model output payload.
- `final_content.ego_behavior_schema.effect_on_ego_behavior`: free-form CoC text.

If `data_loader.video.save_segment_videos` is set to true, the segment videos are saved under:
`experiments/video_segment` folder by default.

## Step 4: Validate CoC Labels

Use the optional three-stage VLM validation pipeline under
[`scripts/coc_validation_stage/`](scripts/coc_validation_stage/) to generate
quality signals from `cot_*.yaml` files. The stages check the grounding of scene
factors cited by a label, agreement between the labeled and observed ego
behavior, and whether the causal explanation is supported and consistent with
the video.

These signals are intended to help identify potentially lower-quality CoC
labels for downstream use or data filtering. The scripts do not require
human-provided labels. The validators accept the same `model_name` aliases and
credential setup as CoC autolabeling (e.g., `gpt5.5`, `qwen3.5_35b`). See the
validation-stage README for examples.

## Troubleshooting

### 1. vLLM Import Fails With `libtorch_cuda.so`

PyTorch likely resolved to a CPU wheel, or the image was built without the CUDA
12.8 wheel index. Rebuild from this Dockerfile and run:

```bash
docker run --gpus all coc_auto_labeling:latest \
  python -c 'from vllm import LLM; print("ok")'
```

### 2. Local Qwen Fails With Unsupported CUDA/PTX

If local Qwen loading fails with `cudaErrorUnsupportedPtxVersion` or another
CUDA/PTX compatibility error, check the host NVIDIA driver with `nvidia-smi`.
The Docker image uses CUDA 12.8 and vLLM 0.17.1, so the host driver must satisfy
the validated configuration table above.

### 3. Cloud GPT Request Fails With `invalid_request_error`

The hosted GPT endpoint may reject explicit sampling parameters. Use the
current wrapper defaults for `gpt5` or `gpt5.5`; sampling overrides documented
in this README are for local Qwen. Non-retryable 4xx errors fail fast.

## Extend with Other Model Clients

Use this section to add new model clients beyond built-in options (for example,
future Qwen variants or other VLM providers).

### 1) Add or reuse a wrapper class

If the new model uses a provider flow that is already implemented, reuse the
existing wrapper. For example, new Qwen aliases should use `QwenWrapper`, and
new OpenAI-compatible aliases should use `OpenAIWrapper`.

If the new provider needs different request or response handling, create a
wrapper under:

- `src/coc_labeling/model_clients/vlm_wrappers/`

Your wrapper should implement the same interface used by existing wrappers:

- `infer(...)`
- `add_message(...)`

See `qwen.py`, `cloud.py`, and `dummy.py` as reference implementations.

### 2) Register the model key in the wrapper factory

Edit:

- `src/coc_labeling/model_clients/vlm_wrappers/factory.py`

Import the wrapper class if it is new, then add your model key to
`MODEL_WRAPPER_REGISTRY`, mapping to the wrapper class. This key is the value
passed via `model_name=...` when launching
`coc_labeling.data_labeling`.

For a new Qwen model alias, also add its cache folder, Hugging Face model ID,
and quantization setting to `QWEN_MODEL_SPECS` in `qwen.py`.

For a new OpenAI-compatible alias, also update `model_name_map` in `cloud.py`
when the public `model_name` key should map to a different provider model ID.

### 3) (Optional) Export the wrapper in package init

If you want public import coverage from the wrapper package, update:

- `src/coc_labeling/model_clients/vlm_wrappers/__init__.py`

### 4) Add or update runtime configs

Update default configs if you want the new model as a runnable preset:

- `src/coc_labeling/config/base_config_vlm.yaml`
- `src/coc_labeling/config/base_config_vlm_rel_ts.yaml`

### 5) Validate with a smoke run

Run a short test on a small clip subset:

```bash
python -m coc_labeling.data_labeling \
  --config-name=base_config_vlm_rel_ts \
  model_name=<your_model_key> \
  resume_exp_dir=null \
  exp_name=smoke_<your_model_key>
```

Use the same CoC labeling arguments described above. Replace
`<your_model_key>` with the model registry key you added, and set `exp_name` to
a short name that identifies the smoke run.

If your wrapper loads local model weights, make sure cache folder names and Hugging
Face model IDs are both correctly mapped in the wrapper implementation.

## Disclaimer

This autolabeling tool is provided for research and development in the autonomous vehicle (AV) domain. It is intended as a foundation and a starting point for building custom VLA applications and is not a production-ready system.

Because this pipeline relies on VLMs, generated CoC outputs may contain errors, including incorrect maneuver attribution (for example, right vs. left lane change), hallucinated objects, or inaccurate temporal-causal reasoning about surrounding agents and ego behavior.

To improve CoC quality, use human auditing and/or a hybrid post-processing
workflow that combines review with heuristic checks. Example safeguards include
validating outputs against object-detection signals (for example, lead-vehicle
or pedestrian presence), planner or behavior signals (for example, nudging or
yielding), and human correction loops.

To facilitate this process, this release also includes VLM-based validation
stages that provide signals for identifying potentially lower-quality CoC
labels for downstream review or filtering. Additional application-specific
safeguards, including human correction loops, are still recommended and remain
outside the scope of the current release.

By using this tool, you acknowledge that it is intended to support scientific inquiry, benchmarking, and exploration, and is not a substitute for a validated or certified AV stack. Developers and contributors disclaim responsibility and liability for use of the model and its outputs.

## License

This project is licensed under the [Apache-2.0](./LICENSE) License.

## Citation

If you use this autolabeling pipeline in your research, please cite:

```bibtex
@article{nvidia2025alpamayo,
      title={{Alpamayo 1}: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail},
      author={NVIDIA and Yan Wang and Wenjie Luo and Junjie Bai and Yulong Cao and Tong Che and Ke Chen and Yuxiao Chen and Jenna Diamond and Yifan Ding and Wenhao Ding and Liang Feng and Greg Heinrich and Jack Huang and Peter Karkus and Boyi Li and Pinyi Li and Tsung-Yi Lin and Dongran Liu and Ming-Yu Liu and Langechuan Liu and Zhijian Liu and Jason Lu and Yunxiang Mao and Pavlo Molchanov and Lindsey Pavao and Zhenghao Peng and Mike Ranzinger and Ed Schmerling and Shida Shen and Yunfei Shi and Sarah Tariq and Ran Tian and Tilman Wekel and Xinshuo Weng and Tianjun Xiao and Eric Yang and Xiaodong Yang and Yurong You and Xiaohui Zeng and Wenyuan Zhang and Boris Ivanovic and Marco Pavone},
      year={2025},
      journal={arXiv preprint arXiv:2511.00088},
}
```
