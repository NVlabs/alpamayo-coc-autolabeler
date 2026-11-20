# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import copy
import json
import logging
import re
from typing import Any, Dict

import coc_labeling.prompts.vlm.coc as prompt
import coc_labeling.prompts.vlm.coc_repair_lateral as repair_prompt
from coc_labeling.request import build_coc_messages

from coc_labeling.model_clients.timeout import TIMEOUT_MAX
from coc_labeling.model_clients.vlm_wrapper import VLMWrapper
from coc_labeling.prompts.common import OutputSchema
from coc_labeling.utils.general_helpers import get_vlm_yaml_path, save_yaml

Message = Dict[str, Any]
DataDict = Dict[str, Any]
OutputDict = Dict[str, Any]
logger = logging.getLogger(__name__)
MODEL_USAGE_KEYS = ("prompt_tokens", "response_tokens", "reasoning_tokens")


class VLMCoCAgent:
    """VLM agent that performs CoC-style structured reasoning."""

    def __init__(
        self,
        model_name: str,
        agent_config: Any,
        vector_config: Any,
        data_loader_config: Any | None = None,
        ip_addr: str | None = None,
        *,
        init_model: bool = True,
        timeout_sec: int = TIMEOUT_MAX,
    ) -> None:
        """Initialize CoC VLM agent and helper function agent.

        Args:
            model_name: Model identifier for the VLM wrapper.
            agent_config: Agent config object (kept for constructor compatibility).
            vector_config: Vector configuration passed to functional helper agent.
            data_loader_config: Resolved data-loader config used for prompt timing.
            init_model: Whether to initialize local model weights immediately.
            timeout_sec: Request timeout in seconds for remote model calls.
        """
        self.vlm = VLMWrapper(
            model_name,
            init_model=init_model,
            ip_addr=ip_addr,
            timeout_sec=timeout_sec,
        )
        self.model_name = model_name
        self.data_loader_config = data_loader_config
        self.temperature = float(getattr(agent_config, "temperature", 0.0))
        self.top_p = float(getattr(agent_config, "top_p", 1.0))
        self.repetition_penalty = float(getattr(agent_config, "repetition_penalty", 1.0))

    @staticmethod
    def _get_prompt_version() -> str:
        """Return prompt version string for logging and output metadata."""
        return getattr(prompt, "PROMPT_VERSION", prompt.__name__.split(".")[-1])

    @staticmethod
    def _get_prompt_version_from_module(prompt_module: Any) -> str:
        """Return prompt version string from an arbitrary prompt module."""
        return getattr(
            prompt_module,
            "PROMPT_VERSION",
            prompt_module.__name__.split(".")[-1],
        )

    def process_results(
        self,
        data: DataDict,
        messages: list[Message],
        response: OutputDict,
        save_root: str,
        *,
        prompt_version: str | None = None,
        extra_output: OutputDict | None = None,
    ) -> tuple[list[Message], OutputDict]:
        """Persist model output and redact heavy video payloads in prompt logs."""
        output_messages = copy.deepcopy(messages)
        for message in output_messages:
            if message["m_type"] == "video":
                video_info = []
                for frame_info in data["all_fpv_frames_info"]:
                    video_info.append(
                        f"[{frame_info.video_filename}, index={frame_info.index_in_video}, "
                        f"ts={frame_info.timestamp_micros}]"
                    )
                message["content"] = (
                    "\n[VIDEO CONTENT PLACEHOLDER INFO]\n" + "\n".join(video_info) + "\n"
                )

        output_dict: OutputDict = {
            "event_start_frame": data["event_start_frame"],
            "event_start_timestamp": data["event_start_timestamp"],
            "coc_prompt_version": prompt_version or self._get_prompt_version(),
            "final_content": response["content"],
            "prompt": output_messages,
        }
        model_usage = {
            key: response[key]
            for key in MODEL_USAGE_KEYS
            if key in response and response[key] is not None
        }
        if model_usage:
            output_dict["model_usage"] = model_usage
        if "event_start_frame_unrounded" in data:
            output_dict["event_start_frame_unrounded"] = data["event_start_frame_unrounded"]
        if extra_output:
            output_dict.update(extra_output)

        save_path = get_vlm_yaml_path(save_root, data)
        save_yaml(save_path, output_dict)
        logger.info(
            "[VLMCoCAgent] Saved result (prompt=%s) to %s",
            output_dict["coc_prompt_version"],
            save_path,
        )

        return messages, output_dict

    def _build_input_message_with_prompt(self, data: DataDict, prompt_module: Any) -> list[Message]:
        """Construct messages for a given prompt module."""
        return build_coc_messages(
            prompt_module,
            self.data_loader_config,
            data["all_meta_action_str"],
            data["ego_text"],
            data["hist_fpv_frames"] + data["fut_fpv_frames"],
        )

    def _build_input_message(self, data: DataDict) -> list[Message]:
        """Construct the common system/user messages for a single-pass CoC run."""
        return self._build_input_message_with_prompt(
            data=data,
            prompt_module=prompt,
        )

    def _infer(self, messages: list[Message], json_schema: Any) -> OutputDict:
        """Run VLM inference with configured sampling parameters."""
        return self.vlm.infer(
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            json_schema=json_schema,
        )

    @staticmethod
    def _extract_effect_on_ego_behavior(content: Any) -> str:
        """Extract effect_on_ego_behavior text from model output payload."""
        if isinstance(content, dict):
            schema_obj = content.get("ego_behavior_schema", {})
            if isinstance(schema_obj, dict):
                effect_text = schema_obj.get("effect_on_ego_behavior", "")
                if isinstance(effect_text, str):
                    return effect_text.strip()
            effect_text = content.get("effect_on_ego_behavior", "")
            if isinstance(effect_text, str):
                return effect_text.strip()
        return ""

    @staticmethod
    def _attach_original_effect(final_content: Any, original_effect_text: str) -> Any:
        """Attach first-pass effect text as effect_on_ego_behavior_original."""
        if not isinstance(final_content, dict):
            return final_content
        if not original_effect_text:
            return final_content
        schema_obj = final_content.get("ego_behavior_schema")
        if isinstance(schema_obj, dict):
            schema_obj["effect_on_ego_behavior_original"] = original_effect_text
        else:
            final_content["effect_on_ego_behavior_original"] = original_effect_text
        return final_content

    @staticmethod
    def _normalize_lane_label(label: str) -> str:
        normalized = re.sub(r"[^a-z_ ]", "", label.strip().lower()).replace(" ", "_")
        mapping = {
            "left_lane_change": "left_lane_change",
            "right_lane_change": "right_lane_change",
            "slightly_shift_left": "slightly_shift_left",
            "slightly_shift_right": "slightly_shift_right",
        }
        return mapping.get(normalized, "")

    @staticmethod
    def _parse_lateral_from_effect_text(effect_text: str) -> tuple[str, str]:
        """Return predicted lateral decision type and direction from output text."""
        text = effect_text.lower()
        has_left = " left" in f" {text}" or "left " in text
        has_right = " right" in f" {text}" or "right " in text
        direction = ""
        if has_left and not has_right:
            direction = "left"
        elif has_right and not has_left:
            direction = "right"

        lane_change_patterns = ("lane change", "change lane", "change lanes")
        nudge_patterns = (
            "nudge",
            "leftward",
            "rightward",
        )
        if any(pattern in text for pattern in lane_change_patterns):
            return "lane_change", direction
        if any(pattern in text for pattern in nudge_patterns):
            return "nudge", direction
        if "keep lane" in text:
            return "keep_lane", direction
        if "turn" in text:
            return "turn", direction
        return "unknown", direction

    def _extract_lateral_prior(self, data: DataDict) -> OutputDict:
        """Build onset-based lane-change/nudge prior from meta-action timeline."""
        all_meta_actions = data.get("all_meta_actions")
        all_ts = data.get("all_ts")
        if not isinstance(all_meta_actions, list) or not isinstance(all_ts, list):
            return {
                "has_prior": False,
                "expected_type": "",
                "expected_direction": "",
                "summary": "No structured all_meta_actions/all_ts provided.",
            }
        if len(all_meta_actions) != len(all_ts) or not all_meta_actions:
            return {
                "has_prior": False,
                "expected_type": "",
                "expected_direction": "",
                "summary": "Meta-action timeline is empty or length-mismatched.",
            }

        def lane_token_at(idx: int) -> tuple[int, str] | None:
            """Return normalized lane token for one original sampled meta-action."""
            action = all_meta_actions[idx]
            if not isinstance(action, dict):
                return None
            lane_label = action.get("Lane")
            if not isinstance(lane_label, str):
                return None
            lane_token = self._normalize_lane_label(lane_label)
            if not lane_token:
                return None
            ts = all_ts[idx]
            if not isinstance(ts, int):
                return None
            return ts, lane_token

        original_ts = sorted({ts for ts in all_ts if isinstance(ts, int)})
        min_step = 1
        if len(original_ts) >= 2:
            deltas = [b - a for a, b in zip(original_ts, original_ts[1:]) if b - a > 0]
            if deltas:
                min_step = min(deltas)

        timeline: list[tuple[int, str]] = []
        for idx in range(len(all_meta_actions)):
            token_info = lane_token_at(idx)
            if token_info is not None:
                timeline.append(token_info)

        if not timeline:
            return {
                "has_prior": False,
                "expected_type": "",
                "expected_direction": "",
                "summary": "No lane-change/nudge token found in meta actions.",
            }

        near_window = 4 * min_step

        lane_change_onset: dict[str, int | None] = {"left": None, "right": None}
        for direction, token in (
            ("left", "left_lane_change"),
            ("right", "right_lane_change"),
        ):
            for idx in range(len(all_meta_actions) - 1):
                token_info0 = lane_token_at(idx)
                token_info1 = lane_token_at(idx + 1)
                if token_info0 is None or token_info1 is None:
                    continue
                ts0, tok0 = token_info0
                ts1, tok1 = token_info1
                if tok0 == token and tok1 == token and 0 < (ts1 - ts0) <= (2 * min_step):
                    lane_change_onset[direction] = ts0
                    break

        nudge_onset: dict[str, int | None] = {"left": None, "right": None}
        for ts, token in timeline:
            if token == "slightly_shift_left" and nudge_onset["left"] is None:
                nudge_onset["left"] = ts
            if token == "slightly_shift_right" and nudge_onset["right"] is None:
                nudge_onset["right"] = ts

        def pick_direction(
            onset_by_dir: dict[str, int | None],
        ) -> tuple[str, int | None]:
            candidates: list[tuple[str, int]] = [
                (direction, onset) for direction, onset in onset_by_dir.items() if onset is not None
            ]
            if not candidates:
                return "", None
            direction, onset = min(candidates, key=lambda item: item[1])
            return direction, onset

        lane_change_dir, lane_change_ts = pick_direction(lane_change_onset)
        nudge_dir, nudge_ts = pick_direction(nudge_onset)

        expected_type = ""
        expected_direction = ""
        expected_onset: int | None = None
        if lane_change_ts is not None and nudge_ts is None:
            expected_type = "lane_change"
            expected_direction = lane_change_dir
            expected_onset = lane_change_ts
        elif nudge_ts is not None and lane_change_ts is None:
            expected_type = "nudge"
            expected_direction = nudge_dir
            expected_onset = nudge_ts
        elif lane_change_ts is not None and nudge_ts is not None:
            lane_near = abs(lane_change_ts) <= near_window
            nudge_near = abs(nudge_ts) <= near_window
            if lane_near != nudge_near:
                if lane_near:
                    expected_type = "lane_change"
                    expected_direction = lane_change_dir
                    expected_onset = lane_change_ts
                else:
                    expected_type = "nudge"
                    expected_direction = nudge_dir
                    expected_onset = nudge_ts
            elif lane_change_ts <= nudge_ts:
                expected_type = "lane_change"
                expected_direction = lane_change_dir
                expected_onset = lane_change_ts
            else:
                expected_type = "nudge"
                expected_direction = nudge_dir
                expected_onset = nudge_ts

        lane_change_count = sum(
            1 for _, token in timeline if token in {"left_lane_change", "right_lane_change"}
        )
        nudge_count = sum(
            1 for _, token in timeline if token in {"slightly_shift_left", "slightly_shift_right"}
        )
        summary = (
            "Lane-prior summary: "
            f"lane_change_count={lane_change_count}, "
            f"lane_change_onset_left={lane_change_onset['left']}, "
            f"lane_change_onset_right={lane_change_onset['right']}, "
            f"nudge_count={nudge_count}, "
            f"nudge_onset_left={nudge_onset['left']}, "
            f"nudge_onset_right={nudge_onset['right']}, "
            f"near_window=abs(ts)<={near_window}. "
            f"Expected lateral decision={expected_type or 'none'} "
            f"direction={expected_direction or 'none'} "
            f"onset_ts={expected_onset if expected_onset is not None else 'none'}."
        )
        return {
            "has_prior": bool(expected_type),
            "expected_type": expected_type,
            "expected_direction": expected_direction,
            "expected_onset_ts": expected_onset,
            "summary": summary,
        }

    def _should_trigger_meta_repair(
        self, effect_text: str, lateral_prior: OutputDict
    ) -> tuple[bool, str]:
        if not lateral_prior.get("has_prior"):
            return False, "No lane-change/nudge prior found."

        expected_type = str(lateral_prior.get("expected_type", ""))
        expected_direction = str(lateral_prior.get("expected_direction", ""))
        predicted_type, predicted_direction = self._parse_lateral_from_effect_text(effect_text)

        if predicted_type == "keep_lane":
            return (
                True,
                "Output collapsed to Keep Lane despite detected lane-change/nudge prior.",
            )
        if predicted_type != expected_type:
            return (
                True,
                f"Output lateral type mismatch: predicted={predicted_type}, "
                f"expected={expected_type}.",
            )
        if expected_direction and predicted_direction != expected_direction:
            return (
                True,
                f"Output direction mismatch: predicted={predicted_direction or 'none'}, "
                f"expected={expected_direction}.",
            )
        return False, "Initial output is consistent with onset-based lateral prior."

    def _build_repair_messages(
        self,
        data: DataDict,
        initial_content: Any,
        lateral_prior: OutputDict,
        trigger_reason: str,
        base_prompt_version: str,
    ) -> list[Message]:
        """Build second-pass repair messages using full video + focused constraints."""
        all_fpv_frames = data["hist_fpv_frames"] + data["fut_fpv_frames"]
        initial_json = json.dumps(initial_content, ensure_ascii=True, indent=2)
        repair_context = repair_prompt.repair_context_template.format(
            base_prompt_version=base_prompt_version,
            trigger_reason=trigger_reason,
            lateral_prior_summary=lateral_prior.get("summary", "N/A"),
            initial_json=initial_json,
        )
        return [
            {"role": "system", "m_type": "text", "content": repair_prompt.system},
            {"role": "user", "m_type": "text", "content": repair_prompt.images_prompt},
            {"role": "user", "m_type": "video", "content": all_fpv_frames},
            {
                "role": "user",
                "m_type": "text",
                "content": repair_prompt.meta_action_prompt,
            },
            {"role": "user", "m_type": "text", "content": data["all_meta_action_str"]},
            {"role": "user", "m_type": "text", "content": repair_context},
            {"role": "user", "m_type": "text", "content": repair_prompt.output_prompts},
        ]

    def run(self, data: DataDict, save_root: str) -> tuple[list[Message], OutputDict]:
        """Run single-pass CoC inference and persist structured output.

        Args:
            data: Segment data dictionary including video/meta/vector fields.
            save_root: Output root directory for YAML artifacts.

        Returns:
            Tuple of prompt messages and saved output payload.
        """
        logger.info(
            "[VLMCoCAgent] Running inference with CoC prompt version: %s",
            self._get_prompt_version(),
        )
        messages = self._build_input_message(data)
        combined_message = self.vlm.add_message_seq(messages)
        response = self._infer(messages=combined_message, json_schema=OutputSchema)
        return self.process_results(data, messages, response, save_root)

    def run_metaaction_check(
        self, data: DataDict, save_root: str
    ) -> tuple[list[Message], OutputDict]:
        """Run base CoC pass, then conditionally run lane/nudge contradiction repair."""
        base_prompt_module = prompt
        base_prompt_ver = self._get_prompt_version_from_module(prompt)
        logger.info(
            "[VLMCoCAgent] Running meta-action check pipeline with base prompt: %s",
            base_prompt_ver,
        )

        base_messages = self._build_input_message_with_prompt(
            data=data,
            prompt_module=base_prompt_module,
        )
        base_combined = self.vlm.add_message_seq(base_messages)
        base_response = self._infer(messages=base_combined, json_schema=OutputSchema)
        initial_content = base_response.get("content", {})
        initial_effect = self._extract_effect_on_ego_behavior(initial_content)

        lateral_prior = self._extract_lateral_prior(data)
        trigger_repair, trigger_reason = self._should_trigger_meta_repair(
            initial_effect,
            lateral_prior,
        )
        logger.info(
            "[VLMCoCAgent] Meta-action repair decision: trigger=%s, reason=%s",
            trigger_repair,
            trigger_reason,
        )

        final_messages = base_messages
        final_response = base_response
        repair_content: Any = None
        if trigger_repair:
            second_messages = self._build_repair_messages(
                data=data,
                initial_content=initial_content,
                lateral_prior=lateral_prior,
                trigger_reason=trigger_reason,
                base_prompt_version=base_prompt_ver,
            )
            second_combined = self.vlm.add_message_seq(second_messages)
            second_response = self._infer(
                messages=second_combined,
                json_schema=OutputSchema,
            )
            second_response["content"] = self._attach_original_effect(
                second_response.get("content", {}),
                initial_effect,
            )
            final_messages = base_messages + second_messages
            final_response = second_response
            repair_content = second_response.get("content", {})

        pipeline_version = (
            f"{base_prompt_ver}+{repair_prompt.PROMPT_VERSION}"
            if trigger_repair
            else base_prompt_ver
        )
        extra_output: OutputDict = {
            "metaaction_check": {
                "base_prompt_version": base_prompt_ver,
                "repair_prompt_version": repair_prompt.PROMPT_VERSION,
                "repair_triggered": trigger_repair,
                "repair_trigger_reason": trigger_reason,
                "lateral_prior_summary": lateral_prior,
            },
            "initial_content": initial_content,
        }
        if repair_content is not None:
            extra_output["repair_content"] = repair_content

        return self.process_results(
            data=data,
            messages=final_messages,
            response=final_response,
            save_root=save_root,
            prompt_version=pipeline_version,
            extra_output=extra_output,
        )


class VLMCoCRemoteAgent(VLMCoCAgent):
    """Remote variant that assumes model weights are served externally."""

    def __init__(
        self,
        model_name: str,
        agent_config: Any,
        vector_config: Any,
        data_loader_config: Any | None = None,
        ip_addr: str | None = None,
        timeout_sec: int = TIMEOUT_MAX,
    ) -> None:
        """Initialize remote-serving variant without local model initialization.

        Args:
            model_name: Model identifier for remote VLM serving.
            agent_config: Agent config object.
            vector_config: Vector configuration for helper functions.
            data_loader_config: Resolved data-loader config used for prompt timing.
            timeout_sec: Request timeout in seconds for remote model calls.
        """
        super().__init__(
            model_name=model_name,
            agent_config=agent_config,
            vector_config=vector_config,
            data_loader_config=data_loader_config,
            ip_addr=ip_addr,
            init_model=False,
            timeout_sec=timeout_sec,
        )
