"""Schema-neutral CoC rendering and action checks shared by both exporters."""

import re


def clean_causal_factor(text: str) -> str | None:
    """Remove graph identifiers and provenance syntax from one causal factor."""
    cause = re.split(r"\s+\|\s+", " ".join(text.split()), maxsplit=1)[0]
    cause = re.sub(
        r"\b(?:track|lane|map|actor|crosswalk|signal)_[a-z0-9_-]+\b\s*:?",
        "",
        cause,
        flags=re.IGNORECASE,
    )
    cause = re.sub(r"\s+([,.;:])", r"\1", cause)
    cause = re.sub(r"(?:\s+(?:and|or))+\s*$", "", cause, flags=re.IGNORECASE)
    cause = re.sub(r"^(?:and|or)\s+", "", cause, flags=re.IGNORECASE)
    return " ".join(cause.split()).strip(" .;,:-") or None


def render_causal_sentence(action: str, causes: list[str]) -> str:
    """Render the full committed action and up to two already-selected causal factors.

    Temporal eligibility and causal relevance belong to the packet adapter. Rendering
    preserves operational detail; it does not select or infer another driving decision.
    """
    action = " ".join(action.split()).strip(" .;,")
    if not action or len(causes) > 2 or any(not cause for cause in causes):
        raise ValueError("rendering requires an action and at most two nonempty causes")
    sentence_causes = [
        cause if cause.split(maxsplit=1)[0].isupper() else cause[0].lower() + cause[1:]
        for cause in causes
    ]
    if not sentence_causes:
        return f"{action}."
    return f"{action} because {' and '.join(sentence_causes)}."


def apply_causal_span_patch(
    narrative: str, old_span: str, new_span: str, locked_prefix: str | None = None
) -> str:
    """Replace one unique span of at most twelve words without changing a locked prefix."""
    old_span, new_span = " ".join(old_span.split()), " ".join(new_span.split())
    if not old_span or not new_span:
        raise ValueError("a required edit must contain both spans")
    if len(old_span.split()) > 12 or len(new_span.split()) > 12:
        raise ValueError("minimal span edits are limited to 12 words per span")
    if narrative.count(old_span) != 1:
        raise ValueError("old_span must occur exactly once in the prior draft")
    if locked_prefix is not None and (
        not narrative.startswith(locked_prefix) or narrative.index(old_span) < len(locked_prefix)
    ):
        raise ValueError("causal patch overlaps the locked action prefix")
    return narrative.replace(old_span, new_span, 1)


def action_lock_issues(action: str, narrative: str) -> list[str]:
    """Return lexical action-family/direction mismatches with a committed action.

    This preserves the keyframe exporter's established guard exactly. Passing this
    guard is not proof of semantic equivalence or evidence entailment.
    """
    patterns = (
        ("lane_change", r"\b(lane change|change lanes?|changing lanes?)\b"),
        ("nudge", r"\b(nudge|shift(?:ing)? (?:slightly )?(?:left|right))\b"),
        ("turn", r"\b(turn|turning|u-turn)\b"),
        ("merge", r"\b(merge|merging)\b"),
        ("split", r"\b(split|diverge|exit(?:ing)?)\b"),
        ("stop", r"\b(stop|stopping|hold at)\b"),
        ("yield", r"\b(yield|give way)\b"),
        ("gap_search", r"\b(gap.search|find (?:a )?(?:safe )?gap|wait for (?:a )?gap)\b"),
    )
    action, output = action.lower(), narrative.lower()
    required = {family for family, pattern in patterns if re.search(pattern, action)}
    produced = {family for family, pattern in patterns if re.search(pattern, output)}
    issues = [f"missing committed {family}" for family in sorted(required - produced)]
    left = bool(re.search(r"\b(left|leftward)\b", action))
    right = bool(re.search(r"\b(right|rightward)\b", action))
    if left != right and required & {"lane_change", "nudge", "turn"}:
        direction = "left" if left else "right"
        opposite = "right" if left else "left"
        if not re.search(rf"\b{direction}\b", output):
            issues.append(f"missing committed {direction} direction")
        if re.search(rf"\b{opposite}\b", output):
            issues.append(f"contains opposite {opposite} direction")
    return issues
