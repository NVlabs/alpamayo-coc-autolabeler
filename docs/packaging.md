# Packaging and validation

The package contains keyframe-based VLM CoC labeling and full-clip streaming reasoning
scene graph (RSG) labeling. The lightweight planning API lets applications integrate the
keyframe workflow with their own media, inference and storage.

## Table of Contents

- [Install and build](#install-and-build)
- [Export a source tree](#export-a-source-tree)
- [Source boundaries and build identity](#source-boundaries-and-build-identity)
- [Validation](#validation)
  - [Offline smoke test](#offline-smoke-test)
- [Supported scope](#supported-scope)

## Install and build

Python 3.11 or newer is required. Install the lightweight planning API with `pip install .`,
`.[pipeline]` for standalone labeling, or `.[streaming]` for generation from prepared RSG
inputs. PAI input preparation also uses the standalone dataset dependencies. Add `.[qwen]`
to a workflow installation for local Qwen inference. See the
[installation guide](../README.md#installation-and-packaging) for the dependency contract
and standalone Docker lock.

To build a wheel and source archive from the repository root:

```bash
python -m pip install '.[dev]'
python -m build
```

The source archive and wheel contain self-contained `rsg_labeling` files. No parent
repository, private source revision or code-review identifier is required.

## Export a source tree

From a clean, committed Git checkout:

```bash
python tools/materialize_export.py /path/to/new-coc-release
```

The tool copies tracked package files into a self-contained source tree and verifies the
RSG file selection. It rejects uncommitted changes, undeclared files and unsupported links.
The result contains regular files and can be transferred or used as a Docker context.

Build the image from this self-contained tree:

```bash
cd /path/to/new-coc-release
docker build -t coc-autolabeling:2.0.0 .
```

Build from the verified source tree so every package file is included in the Docker
context. Preserve the applicable licenses and notices of the dependencies included in any
image you distribute.

## Source boundaries and build identity

`rsg_labeling/releases/files.json` lists the package-relative RSG files. Builds reject
missing or extra files and nested symlinks. Distribution artifacts contain regular files.

The component digest hashes the CoC, meta-action and RSG sources plus package metadata
using relative paths. The wheel records that digest and the package version in its build
identity. Rebuilding the same source archive preserves the identity. Editing selected code
is supported and produces a different digest. Adding a source file also requires updating
the declared file list.

Prompt identity separately records the static prompt, schema and rendering policy used by
the keyframe planning API. See the [identity contract](planning_api.md#identity-and-conformance).

## Validation

Run the lightweight API and artifact checks with:

```bash
python -m pytest \
  tests/test_planning_api.py \
  tests/test_planning_conformance.py \
  tests/test_prompt_timing.py \
  tests/test_distribution.py
```

The tests cover motion spans, stop suppression, heading wrap, signed zero, context coverage,
ordered messages, import boundaries, wheel/source-archive parity, archive rebuilds,
editable installation and source identity. Synthetic baseline fixtures are provided in
`tests/fixtures/planning_conformance.json`. Their public CoC baseline is
`4dda80943b8a88b186f0fe8b0340789380c0ce39`; the standalone trajdata tag is
`coc_autolabeling-1.0.0`.

With streaming dependencies installed, run the offline generation/resume/export checks:

```bash
python -m pytest --pyargs rsg_labeling.tasks.streaming_rsg.tests.test_release_pipeline -q
```

These use generated inputs and scripted responses; they do not measure live-provider
quality, billing or GPU compatibility.

The Python 3.12 Docker image passes the public CoC and streaming tests and a one-event
public PAI smoke test with local Qwen3.5-35B-A3B on two H100 80 GB GPUs. Use the
[keyframe guide's startup settings and GPU scope](keyframe_coc.md#validated-configurations).
Other model, GPU and dependency environments require their own compatibility checks.

### Offline smoke test

This optional integration check uses generated inputs and scripted model replies. It
exercises preparation, generation, resume and exports without running a VLM or downloading
PAI. With streaming dependencies and pytest installed:

```bash
python -m rsg_labeling.tasks.streaming_rsg.tests.release_smoke native --out ./offline-sensor-example
```

Use a new destination. `offline_example.json` identifies the synthetic inputs, requests and
saved annotations. HTTP and ffprobe calls are stubbed; network connections are forbidden.
The captions are predetermined, so the result demonstrates mechanics and output structure,
not labeling quality or real provider billing. For actual labeling, use the
[PAI workflow](rsg.md#prepare-pai-inputs) with your own API credentials.

## Supported scope

Keyframe planning creates requests from prepared motion before media is loaded. Scene-cause
labeling uses camera evidence and model inference supplied by the standalone workflow or
the integrating application. Lane-dependent processing remains in the standalone workflow.

Full-clip streaming RSG labeling has its own incremental graph execution and interval
outputs. It is not exposed through the keyframe planning API. Model, prompt, motion-rule
and dependency changes require their own compatibility and quality checks.
