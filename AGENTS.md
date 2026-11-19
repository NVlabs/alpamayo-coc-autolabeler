# CoC autolabeling contributor instructions

Read [README.md](README.md) to choose the workflow and [docs/packaging.md](docs/packaging.md)
for installation, source export and supported environments.

- Keyframe VLM CoC labeling selects anchors from ego meta-actions and labels their video
  context. Its model-free `coc_labeling.api` plans requests; callers still bind video and
  perform inference. Preserve the standalone motion rules, sampling, prompts and schema.
- Streaming `rsg_labeling` labels a full clip using sensor or camera-only inputs. Read its
  [component instructions](src/rsg_labeling/AGENTS.md) and [PAI guide](docs/rsg.md).
  For an optional offline check, use the [smoke test](docs/packaging.md#offline-smoke-test).
- The lightweight core must import and plan without dataset, GPU or model clients.
  Optional `pipeline`, `streaming` and `qwen` extras own their workflow dependencies.
- Keep credentials, data, output artifacts and local budget configuration outside tracked
  files. Honor existing owner-approved envelopes; do not invent approval, increase limits,
  clear unknown charges or retry an exhausted credential.
