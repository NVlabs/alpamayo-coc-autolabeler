"""Ground external route intent through the same operation gateway as other evidence."""

from ...common.navigation import maneuver_family, maneuver_text
from ..data_structure.rsg import Component, Event, Provenance, Tier
from .base import OpResult


class GroundNavigationOp:
    """Represent route instructions, never infer executed turns or governing permission."""

    name = "ground:navigation"

    def run(self, rsg, env, anchor, orchestrator) -> OpResult:
        """Keep each route sample's own provenance, without consulting later route states."""
        records = env.navigation_records("agent", anchor)
        for record in records:
            sample = record.payload
            maneuver = sample["maneuver"]
            family = maneuver_family(maneuver)
            direction = "left" if "LEFT" in maneuver else "right" if "RIGHT" in maneuver else "none"
            target = maneuver_text(maneuver, family, direction) if family else maneuver.lower()
            label = f"Recorded route intent: {target}; remaining {sample['remaining_m']:.0f} m"
            timestamp = record.timestamp_us
            rsg.add_component(
                Component(
                    id=f"route_{timestamp}",
                    category="ego_goal",
                    label=label,
                    tier=Tier.OBSERVABLE,
                    provenance=Provenance(op=self.name, read_ids=(record.id,)),
                    sources=(record.id,),
                    timestamp_us=timestamp,
                    event_window_us=(timestamp, timestamp),
                    events=(
                        Event(
                            label=label,
                            window_us=(timestamp, timestamp),
                            sources=(record.id,),
                            attributes={**sample, "intent_only": True},
                        ),
                    ),
                )
            )
        orchestrator.memory.write(f"grounded {len(records)} external route-intent samples")
        return OpResult()
