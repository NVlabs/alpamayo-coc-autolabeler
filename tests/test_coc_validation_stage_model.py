"""Unit tests for coc_validation_stage model resolution."""

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

STAGE_DIR = Path(__file__).resolve().parents[1] / "scripts" / "coc_validation_stage"
sys.path.insert(0, str(STAGE_DIR))

import common  # noqa: E402


class TestModelResolution(unittest.TestCase):
    """Model name resolution, defaults, and runtime backend selection."""

    def test_supported_model_names_match_autolabeler(self) -> None:
        """Supported model names stay in sync with the autolabeler set."""
        expected = {
            "qwen3_vl_235b_awq",
            "qwen3.5_35b",
            "qwen3.5_397b_fp8",
            "gpt5",
            "gpt5.5",
        }
        self.assertEqual(set(common.SUPPORTED_MODEL_NAMES), expected)

    def test_resolve_provider_model_name_qwen(self) -> None:
        """Qwen models map to their local HF checkpoint name."""
        self.assertEqual(
            common.resolve_provider_model_name("qwen3.5_35b"),
            common.QWEN_HF_MODEL_MAP["qwen3.5_35b"],
        )

    def test_apply_model_defaults_gpt_omits_temperature(self) -> None:
        """GPT defaults omit temperature and keep Qwen thinking-disable off."""
        args = mock.Mock()
        args.model_name = "gpt5.5"
        args.omit_temperature = None
        args.qwen_disable_thinking = None
        args.dry_run = True
        args.api_base = "auto"
        args.api_key_env = None
        args.provider_model = None
        runtime = common.apply_model_defaults(args)
        self.assertIsNone(runtime)
        self.assertTrue(args.omit_temperature)
        self.assertFalse(args.qwen_disable_thinking)

    def test_apply_model_defaults_qwen_enables_thinking_disable(self) -> None:
        """Qwen defaults keep temperature and enable thinking-disable."""
        args = mock.Mock()
        args.model_name = "qwen3.5_35b"
        args.omit_temperature = None
        args.qwen_disable_thinking = None
        args.dry_run = True
        args.api_base = "auto"
        args.api_key_env = None
        args.provider_model = None
        runtime = common.apply_model_defaults(args)
        self.assertIsNone(runtime)
        self.assertFalse(args.omit_temperature)
        self.assertTrue(args.qwen_disable_thinking)

    def test_create_model_runtime_local_qwen(self) -> None:
        """A local Qwen runtime picks the local backend and defaults."""
        env = {
            "LOCAL_OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
            "LOCAL_OPENAI_API_KEY": "EMPTY",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            runtime = common.create_model_runtime("qwen3.5_35b")
        self.assertEqual(runtime.backend, "local")
        self.assertEqual(runtime.model_name, common.QWEN_HF_MODEL_MAP["qwen3.5_35b"])
        self.assertTrue(runtime.qwen_disable_thinking)
        self.assertEqual(runtime.default_temperature, 0.0)

    def test_explicit_api_base_plus_custom_key_selects_openai_compatible_route(self) -> None:
        """Explicit base plus a resolvable custom key selects OpenAI-compatible."""
        env = {"CUSTOM_KEY": "sk-custom"}
        with mock.patch.dict(os.environ, env, clear=True):
            runtime = common.create_model_runtime(
                "gpt5.5",
                api_base="https://my-endpoint.test/v1",
                api_key_env="CUSTOM_KEY",
            )
        self.assertEqual(runtime.backend, "openai_compatible")
        self.assertEqual(str(runtime.client.base_url).rstrip("/"), "https://my-endpoint.test/v1")

    def test_explicit_api_base_without_resolvable_key_falls_through(self) -> None:
        """An explicit base with no resolvable key raises ValueError."""
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                common.create_model_runtime(
                    "gpt5.5",
                    api_base="https://my-endpoint.test/v1",
                    api_key_env="CUSTOM_KEY",
                )

    def test_is_frame_based_vision_model(self) -> None:
        """Frame-based vision detection is true for GPT, false for Qwen."""
        self.assertTrue(common.is_frame_based_vision_model("provider/gpt-5.5"))
        self.assertFalse(
            common.is_frame_based_vision_model(common.QWEN_HF_MODEL_MAP["qwen3.5_35b"])
        )

    def test_gpt5_runtime_uses_max_completion_tokens(self) -> None:
        """GPT-5 family runtimes flag max_completion_tokens usage."""
        env = {"OPENAI_API_KEY": "synthetic-public-key"}
        with mock.patch.dict(os.environ, env, clear=True):
            runtime = common.create_model_runtime("gpt5.5")
        self.assertTrue(runtime.uses_max_completion_tokens)

    def test_qwen_runtime_uses_legacy_max_tokens(self) -> None:
        """Qwen runtimes keep the legacy max_tokens parameter."""
        env = {
            "LOCAL_OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
            "LOCAL_OPENAI_API_KEY": "EMPTY",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            runtime = common.create_model_runtime("qwen3.5_35b")
        self.assertFalse(runtime.uses_max_completion_tokens)

    def test_is_gpt5_family_model_name(self) -> None:
        """GPT-5 family detection covers aliased and provider-prefixed names."""
        self.assertTrue(common.is_gpt5_family_model_name("gpt5.5"))
        self.assertTrue(common.is_gpt5_family_model_name("gpt-5"))
        self.assertTrue(common.is_gpt5_family_model_name("provider/gpt-5.5"))
        self.assertFalse(common.is_gpt5_family_model_name("qwen3.5-35b"))


class TestPublicRuntimeCredentials(unittest.TestCase):
    """Explicit endpoint keys and public model aliases are never privately remapped."""

    def test_public_api_key_and_base_select_standard_runtime(self):
        """Use the caller's public credentials and configured compatible endpoint."""
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "synthetic-key"}, clear=True):
            runtime = common.create_model_runtime("gpt5.5", api_base="https://example.test/v1")
        self.assertEqual(runtime.backend, "openai")
        self.assertEqual(runtime.model_name, "gpt-5.5")
        self.assertEqual(str(runtime.client.base_url).rstrip("/"), "https://example.test/v1")

    def test_missing_explicit_key_does_not_fall_back_to_another_key(self):
        """A named custom credential must exist, even when another public key is set."""
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "different-key"}, clear=True):
            with self.assertRaises(ValueError):
                common.create_model_runtime(
                    "gpt5.5", api_base="https://example.test/v1", api_key_env="MISSING_KEY"
                )

    def test_explicit_provider_model_is_preserved(self):
        """Caller overrides take precedence over the public alias mapping."""
        self.assertEqual(
            common.resolve_provider_model_name("gpt5.5", "custom-model"), "custom-model"
        )


class TestChatCompletionTokenParameter(unittest.TestCase):
    """Chat completion sends the correct token-budget parameter per model."""

    def _make_runtime(self, *, uses_max_completion_tokens: bool) -> common.ModelRuntime:
        """Build a runtime whose fake client records the kwargs it receives."""
        recorded: dict[str, object] = {}

        class _FakeCompletions:
            """Stub completions endpoint that records call kwargs."""

            def create(self, **kwargs: object) -> object:
                """Record kwargs and return a canned completion response."""
                recorded.update(kwargs)
                message = types.SimpleNamespace(content="ok")
                choice = types.SimpleNamespace(message=message, finish_reason="stop")
                return types.SimpleNamespace(choices=[choice])

        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=_FakeCompletions())
        )
        runtime = common.ModelRuntime(
            client=fake_client,
            model_name="gpt-5.5",
            backend="openai",
            omit_temperature=True,
            qwen_disable_thinking=False,
            default_temperature=None,
            uses_max_completion_tokens=uses_max_completion_tokens,
        )
        self._recorded = recorded
        return runtime

    def test_gpt5_family_sends_max_completion_tokens(self) -> None:
        """GPT-5 family calls pass max_completion_tokens, not max_tokens."""
        runtime = self._make_runtime(uses_max_completion_tokens=True)
        common.call_chat_completion(
            runtime,
            user_content=[{"type": "text", "text": "hi"}],
            max_tokens=123,
            temperature=None,
            qwen_disable_thinking=False,
        )
        self.assertEqual(self._recorded["max_completion_tokens"], 123)
        self.assertNotIn("max_tokens", self._recorded)

    def test_legacy_model_sends_max_tokens(self) -> None:
        """Legacy models pass max_tokens, not max_completion_tokens."""
        runtime = self._make_runtime(uses_max_completion_tokens=False)
        common.call_chat_completion(
            runtime,
            user_content=[{"type": "text", "text": "hi"}],
            max_tokens=123,
            temperature=0.0,
            qwen_disable_thinking=False,
        )
        self.assertEqual(self._recorded["max_tokens"], 123)
        self.assertNotIn("max_completion_tokens", self._recorded)


class TestChatCompletionTruncationRetry(unittest.TestCase):
    """Truncated (finish_reason == 'length') responses trigger budget retries."""

    def _make_runtime(self, finish_reasons: list[str]) -> tuple[common.ModelRuntime, list[int]]:
        """Build a runtime whose fake client replays scripted finish reasons."""
        budgets: list[int] = []
        pending = list(finish_reasons)

        class _FakeCompletions:
            """Stub completions endpoint that records token budgets per call."""

            def create(self, **kwargs: object) -> object:
                """Record the max_tokens budget and return the next scripted result."""
                budgets.append(int(kwargs["max_tokens"]))
                reason = pending.pop(0)
                message = types.SimpleNamespace(content="partial text")
                choice = types.SimpleNamespace(message=message, finish_reason=reason)
                return types.SimpleNamespace(choices=[choice])

        fake_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=_FakeCompletions())
        )
        runtime = common.ModelRuntime(
            client=fake_client,
            model_name="qwen-x",
            backend="local",
            omit_temperature=False,
            qwen_disable_thinking=False,
            default_temperature=None,
            uses_max_completion_tokens=False,
        )
        return runtime, budgets

    def test_retries_on_length_even_with_partial_text(self) -> None:
        """A length finish retries with a doubled budget despite partial text."""
        runtime, budgets = self._make_runtime(["length", "stop"])
        common.call_chat_completion(
            runtime,
            user_content=[{"type": "text", "text": "hi"}],
            max_tokens=100,
            temperature=None,
            qwen_disable_thinking=False,
        )
        self.assertEqual(budgets, [100, 200])

    def test_no_retry_when_not_truncated(self) -> None:
        """A non-truncated response makes exactly one request."""
        runtime, budgets = self._make_runtime(["stop"])
        common.call_chat_completion(
            runtime,
            user_content=[{"type": "text", "text": "hi"}],
            max_tokens=100,
            temperature=None,
            qwen_disable_thinking=False,
        )
        self.assertEqual(budgets, [100])


class TestEvaluationValidation(unittest.TestCase):
    """Score validation gates parsed evaluations before they are accepted."""

    STAGE1_KEYS = (
        "visual_evidence_score",
        "scene_factor_hallucination_score",
        "explanation",
    )

    def test_valid_scores_accept_numeric_string_and_float(self) -> None:
        """Numeric strings and floats are both accepted and normalized."""
        parsed = {
            "visual_evidence_score": "0.75",
            "scene_factor_hallucination_score": 0.2,
            "explanation": "visible lead vehicle",
        }
        evaluation, error = common.validate_parsed_evaluation(parsed, self.STAGE1_KEYS)
        self.assertIsNone(error)
        assert evaluation is not None
        self.assertEqual(evaluation["visual_evidence_score"], 0.75)
        self.assertEqual(evaluation["scene_factor_hallucination_score"], 0.2)
        self.assertEqual(evaluation["explanation"], "visible lead vehicle")

    def test_missing_score_is_retryable_validation_error(self) -> None:
        """A missing required score yields a retryable validation error."""
        parsed = {"scene_factor_hallucination_score": 0.2}
        evaluation, error = common.validate_parsed_evaluation(parsed, self.STAGE1_KEYS)
        self.assertIsNone(evaluation)
        self.assertEqual(error, "validation_error:missing_score:visual_evidence_score")

    def test_invalid_score_rejects_out_of_range_and_non_numeric(self) -> None:
        """Out-of-range, non-finite, boolean, and non-numeric scores are rejected."""
        cases = [
            {"visual_evidence_score": 1.5, "scene_factor_hallucination_score": 0.2},
            {"visual_evidence_score": -0.1, "scene_factor_hallucination_score": 0.2},
            {"visual_evidence_score": float("inf"), "scene_factor_hallucination_score": 0.2},
            {"visual_evidence_score": True, "scene_factor_hallucination_score": 0.2},
            {"visual_evidence_score": "not-a-number", "scene_factor_hallucination_score": 0.2},
        ]
        for parsed in cases:
            with self.subTest(parsed=parsed):
                evaluation, error = common.validate_parsed_evaluation(parsed, self.STAGE1_KEYS)
                self.assertIsNone(evaluation)
                self.assertEqual(error, "validation_error:invalid_score:visual_evidence_score")

    def test_explanation_optional_for_ok_gate(self) -> None:
        """A missing optional explanation still passes validation."""
        parsed = {
            "visual_evidence_score": 0.8,
            "scene_factor_hallucination_score": 0.1,
        }
        evaluation, error = common.validate_parsed_evaluation(parsed, self.STAGE1_KEYS)
        self.assertIsNone(error)
        assert evaluation is not None
        self.assertNotIn("explanation", evaluation)

    def test_stage3_decision_supported_by_video_validated_as_score(self) -> None:
        """Stage 3 decision-support field is validated like a numeric score."""
        keys = (
            "decision_supported_by_video",
            "causal_evidence_score",
            "causal_consistency_score",
            "causal_hallucination_score",
            "explanation",
        )
        parsed = {
            "decision_supported_by_video": 0.7,
            "causal_evidence_score": 0.6,
            "causal_consistency_score": 0.5,
            "causal_hallucination_score": 0.2,
        }
        evaluation, error = common.validate_parsed_evaluation(parsed, keys)
        self.assertIsNone(error)
        assert evaluation is not None
        self.assertEqual(evaluation["decision_supported_by_video"], 0.7)


if __name__ == "__main__":
    unittest.main()
