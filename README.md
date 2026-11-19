# Chain-of-Causation and Reasoning Scene Graph Autolabeling

Label autonomous-driving scenarios with Chain-of-Causation (CoC) explanations at selected
keyframes, or reasoning scene graphs (RSGs) and CoC descriptions across full driving clips.
Meta-actions describe ego motion; CoC explanations connect scene causes to driving behavior.
RSGs retain the entities, events, relationships and evidence behind those explanations.

## Table of Contents

- [Choose a workflow](#choose-a-workflow)
- [Installation and packaging](#installation-and-packaging)
- [Disclaimer](#disclaimer)
- [Citation](#citation)
- [License](#license)

## Choose a workflow

| Workflow                                        | Coverage and outputs                                                                                                                                                                      | Instructions                           |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| Keyframe-based VLM CoC labeling                 | Meta-actions select keyframes; a VLM labels the surrounding camera context with a CoC explanation at each selected keyframe.                                                              | [Keyframe guide](docs/keyframe_coc.md) |
| **Full-clip streaming RSG/CoC labeling (new!)** | Builds **reasoning scene graphs (RSGs)** incrementally and produces CoC descriptions over time intervals covering the clip. Uses camera video, with optional ego motion and meta-actions. | [Streaming RSG guide](docs/rsg.md)     |

An RSG is a richer representation than its compact exported CoC descriptions. Both
workflows can reuse the public Physical AI AV dataset and prepared motion/meta-action data.

Applications that own media, inference and storage can use the [CoC planning API](docs/planning_api.md)
as a library entry point for keyframe labeling. It prepares sampling ticks and prompt
requests; camera evidence and model inference still provide scene causes.

## Installation and packaging

Use Python **3.11 or newer** and run installation commands from this repository's root:

```bash
python -m pip install .
```

The lightweight planning core installs NumPy, pandas and Pydantic. Add the dependencies
for the workflow you need:

| Extra          | Purpose                                                                                             |
| -------------- | --------------------------------------------------------------------------------------------------- |
| `.[pipeline]`  | Standalone dataset preparation and keyframe CoC labeling.                                           |
| `.[streaming]` | Streaming RSG generation from prepared inputs. PAI preparation also needs the dataset dependencies. |
| `.[qwen]`      | Local Qwen inference; supplements a workflow installation.                                          |

The base install does not include Torch or trajdata. The planning core is tested on
Python 3.13; the standalone dataset/VLM environment uses Python 3.12 and its documented
Docker dependency lock. See the [keyframe guide](docs/keyframe_coc.md#dependencies-build-image-from-dockerfile)
for that setup and the [packaging guide](docs/packaging.md) for source exports, builds and
validated environments. Optional extras should be installed using the selected workflow's
dependency instructions.

Start streaming with the [PAI sensor workflow](docs/rsg.md#prepare-pai-inputs),
using prepared data and your own model API credentials.
See [meta-action preparation](docs/meta_action_autolabel.md) and the
[changelog](CHANGELOG.md) for more detail.
Contributors and coding agents should read [AGENTS.md](AGENTS.md).

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

## License

This project is licensed under the [Apache-2.0](./LICENSE) License.
