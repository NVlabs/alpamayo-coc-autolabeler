"""Copyedit interval CoC presentation without changing its labeling contract."""

from copy import deepcopy

from .....common.coc import clean_causal_factor, render_causal_sentence
from .....common.coc_copyedit import (
    clock_time_spans,
    copyedit_causal_sentence,
    copyedit_causal_texts,
)


def copyedit_interval_cocs(
    result: dict, style="exact_action_composition_v1", complete=None, *, omit_clock_times=False
) -> dict:
    """Copy finished interval labels and retain every original field and edit receipt.

    Anchor-conditioned policy samples are deliberately untouched: a teacher interval's
    wording is not permission to substitute teacher evidence into a causal sample.
    """
    if type(omit_clock_times) is not bool or (
        omit_clock_times
        and style not in {"verified_readable_account_v1", "verified_readable_decision_account_v1"}
    ):
        raise ValueError("clock-free prose requires a readable account")
    if style not in {
        "exact_action_composition_v1",
        "verified_fluent_v1",
        "verified_fluent_exact_v1",
        "verified_cause_first_v1",
        "verified_source_components_v1",
        "verified_event_account_v1",
        "verified_readable_account_v1",
        "verified_readable_decision_account_v1",
        "restore_selected_original_v1",
        "verified_local_deletion_v1",
        "verified_local_reorder_v1",
    }:
        raise ValueError("unknown CoC copyedit style")
    if style in {
        "restore_selected_original_v1",
        "verified_local_deletion_v1",
        "verified_local_reorder_v1",
    }:
        edited = _restore_selected_originals(result)
        if style == "restore_selected_original_v1":
            return edited
        if complete is None:
            raise ValueError("local copyediting requires a metered completion")
        texts = {
            b["interval_id"]: b["coc"]
            for b in edited["bundles"]
            if isinstance(b["coc"], str) and b["coc"].strip() and not b["coc"].startswith("<")
        }
        components = {
            b["interval_id"]: {
                "selected_action": b["coc_export"]["public_action"]["rendered"],
                "selected_explanations": [f["text"] for f in b["coc_export"]["factors"]],
            }
            for b in edited["bundles"]
            if b["interval_id"] in texts
        }
        edits = copyedit_causal_texts(
            texts,
            complete,
            components=components,
            edit_scope=style.removeprefix("verified_").removesuffix("_v1"),
        )
        for bundle in edited["bundles"]:
            if bundle["interval_id"] in edits:
                receipt = edits[bundle["interval_id"]]
                bundle["coc_copyedit"] = {"version": style, **receipt}
                bundle["coc"] = receipt["text"]
        return edited
    if style == "verified_fluent_exact_v1":
        if not any("coc_copyedit" in bundle for bundle in result["bundles"]):
            result = copyedit_interval_cocs(result, "verified_fluent_v1", complete)
        return finish_rejected_interval_cocs(result)
    if any("coc_copyedit" in bundle for bundle in result["bundles"]):
        raise ValueError("copyediting requires the original publication, not an edited replay")
    edited = deepcopy(result)
    if style in {
        "verified_fluent_v1",
        "verified_cause_first_v1",
        "verified_source_components_v1",
        "verified_event_account_v1",
        "verified_readable_account_v1",
        "verified_readable_decision_account_v1",
    }:
        if complete is None:
            raise ValueError("fluent copyediting requires a metered completion")
        texts = {
            bundle["interval_id"]: bundle["coc"]
            for bundle in edited["bundles"]
            if isinstance(bundle["coc"], str)
            and bundle["coc"].strip()
            and not bundle["coc"].startswith("<")
        }
        if len({bundle["interval_id"] for bundle in edited["bundles"]}) != len(edited["bundles"]):
            raise ValueError("interval IDs must be unique")
        clause_order = "cause_first" if style == "verified_cause_first_v1" else "action_first"
        components = None
        if style in {
            "verified_source_components_v1",
            "verified_event_account_v1",
            "verified_readable_account_v1",
            "verified_readable_decision_account_v1",
        }:
            components = {
                bundle["interval_id"]: {
                    "selected_action": bundle["coc_export"]["public_action"]["rendered"],
                    "selected_explanations": [
                        factor["text"] for factor in bundle["coc_export"]["factors"]
                    ]
                    + (
                        [bundle["coc_support_context"]["text"]]
                        if bundle.get("coc_support_context", {}).get("text")
                        else []
                    ),
                }
                for bundle in edited["bundles"]
                if bundle["interval_id"] in texts
            }
        pending = texts
        edits: dict[str, dict] = {}
        rejected: dict[str, list[dict]] = {}
        retry_feedback = None
        for _ in range(3 if omit_clock_times else 1):
            attempt = copyedit_causal_texts(
                pending,
                complete,
                clause_order=clause_order,
                components=(
                    {key: components[key] for key in pending} if components is not None else None
                ),
                edit_scope=(
                    style.removeprefix("verified_").removesuffix("_v1")
                    if style
                    in {
                        "verified_event_account_v1",
                        "verified_readable_account_v1",
                        "verified_readable_decision_account_v1",
                    }
                    else "sentence"
                ),
                omit_clock_times=omit_clock_times,
                retry_feedback=retry_feedback,
            )
            edits.update(attempt)
            pending = {
                key: texts[key]
                for key, receipt in attempt.items()
                if omit_clock_times and clock_time_spans(receipt["text"])
            }
            for key in pending:
                rejected.setdefault(key, []).append(attempt[key])
            if not pending:
                break
            retry_feedback = {
                key: {
                    "proposed": attempt[key]["proposed"],
                    "issues": attempt[key]["issues"],
                    "remaining_clock_references": [
                        attempt[key]["proposed"][start:end]
                        for start, end in clock_time_spans(attempt[key]["proposed"])
                    ],
                }
                for key in pending
            }
        if pending:
            raise ValueError(
                f"clock-free copyediting exhausted presentation repairs: {sorted(pending)}"
            )
        for key, attempts in rejected.items():
            edits[key]["clock_presentation_rejected_attempts"] = attempts
        for bundle in edited["bundles"]:
            if bundle["interval_id"] in edits:
                receipt = edits[bundle["interval_id"]]
                bundle["coc_copyedit"] = {"version": style, **receipt}
                bundle["coc"] = receipt["text"]
        return (
            finish_rejected_interval_cocs(edited, expected_style=style)
            if style
            not in {
                "verified_fluent_v1",
                "verified_event_account_v1",
                "verified_readable_account_v1",
                "verified_readable_decision_account_v1",
            }
            else edited
        )
    for bundle in edited["bundles"]:
        export = bundle.get("coc_export", {})
        action = export.get("public_action", {}).get("rendered")
        if not action or not isinstance(bundle["coc"], str):
            continue
        wording = copyedit_causal_sentence(
            bundle["coc"], action, [factor["text"] for factor in export.get("factors", [])]
        )
        if wording["edits"]:
            bundle["coc_copyedit"] = {
                "version": "exact_action_composition_v1",
                "original": bundle["coc"],
                "edits": wording["edits"],
            }
            bundle["coc"] = wording["text"]
    return edited


def _restore_selected_originals(result):
    """Recover renderer-verified originals with append-only prior-publication custody."""
    edited = deepcopy(result)
    if len({b["interval_id"] for b in edited["bundles"]}) != len(edited["bundles"]):
        raise ValueError("interval IDs must be unique")
    for bundle in edited["bundles"]:
        text = bundle["coc"]
        if not isinstance(text, str) or not text.strip() or text.startswith("<"):
            continue
        export = bundle["coc_export"]
        expected = render_causal_sentence(
            export["public_action"]["rendered"],
            [clean_causal_factor(f["text"]) for f in export["factors"]],
        )
        receipt = bundle.get("coc_copyedit")
        if receipt:
            allowed = {
                "verified_fluent_v1",
                "verified_fluent_exact_v1",
                "verified_cause_first_v1",
                "verified_cause_first_exact_v1",
                "verified_source_components_v1",
                "verified_source_components_exact_v1",
                "verified_local_deletion_v1",
                "verified_local_reorder_v1",
                "restore_selected_original_v1",
            }
            if receipt.get("version") not in allowed or receipt.get("text") != text:
                raise ValueError("unexpected or modified copyedit publication")
            if receipt.get("original") != expected:
                raise ValueError("original differs from selected-content renderer")
            bundle.setdefault("coc_copyedit_history", []).append(
                {"receipt": receipt, "published": text}
            )
            bundle["coc"] = expected
            bundle["coc_copyedit"] = {
                "version": "restore_selected_original_v1",
                "original": expected,
                "text": expected,
                "accepted": True,
                "changed": False,
                "fidelity_status": "original_selected_content_restored_not_fluency_checked",
            }
        elif text != expected:
            raise ValueError("publication differs from selected-content renderer")
    return edited


def finish_rejected_interval_cocs(result: dict, expected_style="verified_fluent_v1") -> dict:
    """Keep checked fluent prose and exact-compose only rejected original wrappers.

    Rejections remain in the receipt. Unrecognized paraphrases are unchanged; this
    operation does not infer causal links or claim that every grammar issue is repaired.
    """
    if expected_style not in {
        "verified_fluent_v1",
        "verified_cause_first_v1",
        "verified_source_components_v1",
    }:
        raise ValueError("unknown upstream copyedit style")
    edited = deepcopy(result)
    for bundle in edited["bundles"]:
        receipt = bundle.get("coc_copyedit")
        if receipt is None:
            if (
                isinstance(bundle["coc"], str)
                and bundle["coc"].strip()
                and not bundle["coc"].startswith("<")
            ):
                raise ValueError("fluent cleanup requires checked upstream prose")
            continue
        if receipt["version"] != expected_style or bundle["coc"] != receipt["text"]:
            raise ValueError("unexpected or modified fluent publication")
        if receipt["accepted"]:
            continue
        if bundle["coc"] != receipt["original"]:
            raise ValueError("a rejected fluent edit must retain the original")
        export = bundle.get("coc_export", {})
        action = export.get("public_action", {}).get("rendered")
        if not action:
            continue
        wording = copyedit_causal_sentence(
            bundle["coc"], action, [factor["text"] for factor in export.get("factors", [])]
        )
        if not wording["edits"]:
            continue
        bundle["coc_copyedit"] = {
            "version": receipt["version"].removesuffix("_v1") + "_exact_v1",
            "original": receipt["original"],
            "text": wording["text"],
            "changed": True,
            "accepted": True,
            "acceptance_kind": "deterministic_exact_action_composition",
            "fidelity_status": "original_explanation_words_unchanged",
            "upstream_copyedit": receipt,
            "edits": wording["edits"],
        }
        bundle["coc"] = wording["text"]
    return edited
