# Streaming RSG contributor instructions

Read the [RSG workflow guide](../../docs/rsg.md) for inputs, generation and exports.

- Compiler and graph operations live in `streaming_rsg/`; interval interpretation and
  CoC publication live in `tasks/streaming_rsg/generation/`. Tasks consume the graph;
  graph changes pass through compiler operations.
- Preserve prompts, motion rules, sampling, graph evidence, output schemas and resume
  identity unless the owner requests a behavior change.
- Sensor and camera-only recipes have different evidence contracts. Missing motion
  stays unavailable; do not fill it with invented measurements.
