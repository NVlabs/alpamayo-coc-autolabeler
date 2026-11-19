# Model-free CoC request planning

`coc_labeling.api.prepare_coc_events()` shares the standalone ego motion rules, cleaning,
candidate selection, sampling, text rendering and request construction. It prepares requests
before video loading. The caller owns motion preparation, camera decoding, model inference and storage.

## Table of Contents

- [Use the planner](#use-the-planner)
- [Input and selection contract](#input-and-selection-contract)
- [Timing and returned records](#timing-and-returned-records)
- [Baseline and prompt policy](#baseline-and-prompt-policy)
- [Caller-owned inference and response handling](#caller-owned-inference-and-response-handling)
- [Identity and conformance](#identity-and-conformance)
- [Standalone and streaming boundaries](#standalone-and-streaming-boundaries)

## Use the planner

```python
import coc_labeling
from coc_labeling.api import OutputSchema, prepare_coc_events

plan = prepare_coc_events(
    xy, heading, velocity_xy,
    selected_actions=None,
    min_duration_ticks=10,
    min_start_tick=0,
    max_start_tick=len(xy) - 1,
)
```

Each `plan.events` entry contains `frame_ticks`, `speed_ticks`, motion text and eight ordered
upstream messages. Exactly one message has `m_type="video"` and `content=None`. Load those camera
ticks, bind that slot, convert the preserved content/order to the chosen model chat format, and
validate the model response with `OutputSchema`. The response has one explanation at
`ego_behavior_schema.effect_on_ego_behavior`. Media binding and inference are caller operations.
Two action records at the same anchor can request identical media and messages.

Record `coc_labeling.__version__`, `coc_labeling.component_source_sha256`, the plan's prompt
identity, effective selection settings, input provenance and media provenance. Installed wheels
provide a deterministic component source digest. A source checkout reports a null component
digest; record its Git revision and working changes instead.

## Input and selection contract

Use finite real NumPy arrays with matching lengths `N >= 2`, uniformly sampled at 10 Hz:

| Input         | Meaning                                                                         |
| ------------- | ------------------------------------------------------------------------------- |
| `xy`          | `(N, 2)` world-frame positions in metres                                        |
| `heading`     | `(N,)` radians, counterclockwise from world +X, representing vehicle forward    |
| `velocity_xy` | `(N, 2)` world-frame metres/second at the same physical reference point as `xy` |

Use one right-handed planar frame. The caller owns reference-point selection, coordinate
conversion, resampling and capture-time origin. The direct API evaluates the whole supplied
trajectory. Wrapped or continuous headings are supported for consecutive changes smaller than pi.
No dataset handles, camera/model clients or GPU initialization occur in the planning API. Calls
hold no episode state, perform no data I/O and are independently usable by parallel workers.
Normal Python dependency imports may load local module files.

Selection arguments are required. `selected_actions=None` selects the fourteen names in
`EGO_META_ACTIONS`; an empty sequence selects none. Accept only canonical ego names; unknown,
legacy-alias and lane/map names raise `ValueError`. `ALL_META_ACTIONS` retains the standalone
vocabulary, including lane actions. Require integer bounds, excluding booleans:
`min_duration_ticks >= 1` and `0 <= min_start_tick <= max_start_tick`.

A span qualifies when its action is selected, its duration is at least the requested minimum,
and its start is within the inclusive start bounds. Malformed motion, timing or selection raises
`ValueError`. The API does not assume a fixed episode duration or the standalone 20–135 range.

## Timing and returned records

V1 supports only `DEFAULT_EVENT_TIMING`, equal to
`EventTiming(history_s=2.0, future_s=6.0, sample_interval_s=0.5)`. Other timing is rejected.

| Context around span start `e` | Requested ticks            | Count               |
| ----------------------------- | -------------------------- | ------------------- |
| Camera and meta-actions       | `e + [-20, -15, ..., +55]` | 16                  |
| Speed                         | `e + [-20, -15, ..., +60]` | 17                  |
| Complete motion coverage      | `e >= 20` and `e + 60 < N` | No padding/clamping |

Timing exposes `history_ticks`, `future_ticks`, `sample_interval_ticks`, `sampled_fps`,
`num_frames`, and `num_speed_samples`. The 2 Hz camera request rate differs from the 10 Hz motion grid.

`CocEpisodePlan` contains `num_ticks`, full `longitudinal_spans` and `lateral_spans`, `events`,
`excluded_candidates`, timing and prompt identity. Spans expose canonical `action`, `start_tick`,
`end_tick` and `duration_ticks`. Events include the span bounds, both sampling tick tuples,
`meta_action_text`, `ego_text` and messages. Records are dataclasses with tuple collections;
message dictionaries are fresh per request and may be copied when binding media.

Both events and excluded candidates are sorted by `(start_tick, end_tick, action)`.
`excluded_candidates` contains selected spans missing complete motion context, its only exclusion
reason. Together these distinguish no qualifying action from incomplete context.

## Baseline and prompt policy

The initial upstream baseline is `4dda80943b8a88b186f0fe8b0340789380c0ce39` from
[NVlabs/alpamayo-coc-autolabeler](https://github.com/NVlabs/alpamayo-coc-autolabeler).
Preserve its terminal `N-1` span endpoint and half-open filling; tick `N-1` remains unfilled
for such a terminal span. Longitudinal stop intervals are subtracted from lateral spans before
candidate selection. Lateral spans select anchors, while rendered metadata fills longitudinal
actions and leaves lateral/lane as `Not Available`.

Prompt wording, including the baseline's lateral-sign sentences and ego-only lane guidance,
is preserved. Those sentences require a separate reviewed prompt-policy correction: the motion
projection is left-positive (`-vx*sin(heading) + vy*cos(heading)`), while the baseline text says
positive is rightward. This integration does not claim that changing those words has been quality
validated. A correction must change the prompt version and identity and receive conformance and
label-quality review. Custom prompts also need their own prompt/schema identity.

## Caller-owned inference and response handling

For caller-hosted local Qwen, the standalone local wrapper's starting profile is
`temperature=0`, `top_p=1`, `repetition_penalty=1`, `max_tokens=4096`, with
`OutputSchema`-constrained generation. This describes its decoding configuration, not
quality qualification for another model or serving stack. Use the documented model choices
as a starting point; qualify new models separately. The Qwen3.5 adapter disables thinking
when the processor supports that setting. Hosted GPT callers use their documented wrapper
settings; do not copy local Qwen sampling parameters into hosted GPT requests.

The planning API does not parse or normalize model responses. A caller may require strict
JSON and use `OutputSchema.model_validate_json(raw_response)`, or apply a documented
normalization policy before schema validation. Standalone local Qwen removes thinking text
and can extract JSON from fenced or surrounding text, with a Python-literal fallback. Those
permissive parsing steps are not part of the planning API contract and need not be copied.
For output comparisons, use the same response policy and retain raw model output alongside
the parsed value.

`OutputSchema` permits an empty or whitespace-only `effect_on_ego_behavior`. Requiring
nonblank text is a consumer acceptance policy in this release. For example, after schema
validation:

```python
explanation = label.ego_behavior_schema.effect_on_ego_behavior
if not explanation.strip():
    raise ValueError("The generated CoC explanation is blank")
```

Preserve the returned frame ticks, order, 16-frame context at 2 fps, forward-facing camera,
17 speed samples, motion conventions, rendered text, message order and output schema for
V1 prompt compatibility. Bind exactly the returned video slot; avoid a second sampling pass
that chooses different frames. A different model, image resolution, JPEG frames versus an
encoded video, compression policy, or model chat adapter is a consumer-qualified inference
choice, not a planner-parity guarantee. The standalone local Qwen wrapper binds ordered
frames as video input with 2 fps metadata; its hosted OpenAI wrapper binds the ordered frames
as JPEG image blocks. Record the effective model/revision, media binding, decoding and
parsing/nonblank policies in result identity, in addition to the plan's prompt identity.

## Identity and conformance

`prompt_identity(timing=DEFAULT_EVENT_TIMING)` returns `(prompt_version, prompt_sha256)` without
episode motion. Every plan has that same identity for its timing. Hash input is canonical UTF-8
JSON: sorted keys, compact separators, unescaped Unicode and no non-finite values. It includes
static rendered messages, output JSON schema, action display wording and executable motion-text
examples that cover headings, timestamps, precision and signed-zero formatting.

The component source digest identifies motion rules, endpoint semantics and packaging inputs;
the prompt digest identifies prompt/schema content. The build digest hashes canonical inventories
of CoC, meta-action and RSG sources plus `pyproject.toml`. Rebuilding the
same sources through the source archive preserves it. It does not embed machine-specific paths.

Identical prepared inputs, effective settings and component builds produce identical plans
within a qualified dependency environment. Exact rendered text is the conformance boundary;
floating motion intermediates are compared with a stated tolerance. Signed zero is preserved by using the standalone axis rotation and wrapped-heading projection
for prompt speed values, including its arithmetic order. The supplied world-frame convention
remains unchanged.
Contract tests cover heading wraparound, full/insufficient context, invalid inputs, ordering,
parallel calls, message shape and import/runtime boundaries. Reusable native-baseline fixtures are in `tests/fixtures/planning_conformance.json` in the
source archive; they cover float32/float64 motion, stop suppression, reverse, acceleration, turns
and heading wraparound. Native adapter and artifact checks are recorded in the [validation guide](packaging.md#validation).

## Standalone and streaming boundaries

Standalone CLI/dataset support, lane derivation, ranking, balancing and repair prompts stay in
their existing workflows. Their adapters call the same extracted helpers. Scene clipping and
rebasing remain separate from full-trajectory API inputs.

The full-clip streaming RSG workflow is available through the `streaming` extra and its existing
entrypoints. Its later requests depend on prior graph/model responses; the one-shot CoC planner
is not its external-hosted inference interface. See [streaming RSG labeling](rsg.md).
