# Streaming RSG technical reference

The [PAI workflow guide](rsg.md) is sufficient for normal labeling. Use this reference only
when preparing another dataset or recovering an interrupted paid run.

## Table of Contents

- [Custom sensor inputs](#custom-sensor-inputs)
- [Cost records](#cost-records)
- [Interruption and reconciliation](#interruption-and-reconciliation)
- [Runtime and recovery files](#runtime-and-recovery-files)

## Custom sensor inputs

Generation uses two manifests: `input_manifest` selects clips and sampling anchors;
`playground.env.sensor_manifest` supplies their measured data. The PAI preparation command
writes both. Other adapters must preserve physical clocks and measured coverage.

`input_manifest` is a JSON object with a nonempty `clips` list. Each entry contains
`clip_id`, `clock` (`origin_us`, `end_us`), `clock_verified` and `sample_anchors_us`.
For integer camera capture times, derive the clock and anchors exactly:

```python
origin_us = captures[0]
end_us = captures[-1] + max(1, round(median(diff(captures))))
sample_anchors_us = list(range(origin_us + 2_000_000, end_us - 6_000_000 + 1, policy.anchor_stride_us))
```

Each anchor needs 2 s history and 6 s future. Only mark a physical clock verified after
checking the data. Camera-only preparation uses a presentation clock and
`clock_verified: false`.

The sensor manifest uses schema `alparsg_sensor_inputs_v1`. Each `clips[clip_id]` entry
contains exactly:

| Field    | Required contents                                                                                                                                   |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `clock`  | The clip's physical microsecond clock.                                                                                                              |
| `ego`    | Parquet `path` and `sha256`; integer `t_us` and finite `x,y,z,qw,qx,qy,qz` columns.                                                                 |
| `camera` | `video`, `video_sha256`, `timestamps` and `timestamps_sha256`. The text sidecar pairs consecutive frame indices with physical integer microseconds. |
| `source` | Source/provenance metadata.                                                                                                                         |

The compiler verifies hashes, capture counts, clocks and anchor bounds before model calls.
Do not pad motion beyond its measured coverage. Generation reads prepared manifests rather
than dataset archives. For MP4-only inputs, use the documented `prepare-video` command
instead of inventing ego motion or verified sensor timestamps.

## Cost records

Provider-confirmed charges and token-based estimates are separate. `budget_cost_usd`
includes admission liabilities; `total_cost_usd` remains null until all charges are
provider-confirmed. Valid estimates permit dispatch under the configured policy, while
unreserved unknown charges block further requests. Estimates and error reserves are not
provider invoices or guaranteed price caps.

Use a dedicated generation ledger and retain its billing evidence. The runtime does not
query provider balances or automatically resolve missing charges. Never treat missing
usage as zero cost, clear an exhausted-key block or delete liabilities to force a restart.

## Interruption and reconciliation

For an unchanged configuration, inputs, code and output path, use the workflow guide's
resume command. Resume parallel work through its original parent with the same worker
assignment. Successful records and receipts are reused.

For changed code/settings or a larger parallel allocation, use a new output directory and
`--recovery-source /path/to/prior-run`. Preserve the recovery source for later resumes.
An increased allocation needs the budget owner's approval and does not erase earlier spend.

If a durable receipt lacks a price, stop affected controllers and obtain provider evidence.
The public `CostLedger.reconcile` method records a replacement price while retaining the
original receipt. Use the verified amount; verified zero is valid, missing evidence is not.
Missing receipts or a lost run require accounting reconciliation before recovery; the CLI
cannot infer those charges. Preserve the run and evidence rather than manually editing
journals or releasing an unexplained reservation.

## Runtime and recovery files

Keep the complete output directory, dedicated ledger and persistent inference state
together. Receipts, identity files and reservations are recovery inputs; do not remove
active locks or truncate accounting files. Keep prepared sensor inputs immutable.

Memory use grows with the batch and retained evidence, so choose an appropriate batch size.
The compiler wall-clock limit is checked between operations, not a hard HTTP deadline.
Request caps, retries and budget admission remain separate limits.
