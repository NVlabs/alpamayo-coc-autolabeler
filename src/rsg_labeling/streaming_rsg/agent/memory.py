"""The agent's generic memory block: distilled notes carried across ops within a run.

Memory is agent working state, not epistemic state — it lives in the context layer, is
rendered into model contexts by the orchestrator, and is persisted alongside the RSG so the
optimization loop can mine it. Scope and eviction are orchestrator policy, pinned per run via
the env.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MemoryBlock:
    """Append-only note store for one run."""

    notes: list[str] = field(default_factory=list)

    def write(self, note: str) -> None:
        """Append one distilled note."""
        self.notes.append(note)

    def render(self, budget_chars: int = 2000) -> str:
        """Render the newest notes into a context block, newest last, within the budget."""
        rendered: list[str] = []
        used = 0
        for note in reversed(self.notes):
            if used + len(note) > budget_chars:
                break
            rendered.append(note)
            used += len(note)
        return "\n".join(reversed(rendered))
