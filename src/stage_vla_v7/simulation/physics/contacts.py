"""Named contact requirements for the Franka stack task."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContactSensorBinding:
    name: str
    parent_link: str
    target_role: str

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.parent_link.strip() or not self.target_role.strip():
            raise ValueError("contact sensor binding fields must be non-empty")


FRANKA_FINGER_CONTACTS = (
    ContactSensorBinding("left_finger_contact", "panda_leftfinger", "manipulated_object"),
    ContactSensorBinding("right_finger_contact", "panda_rightfinger", "manipulated_object"),
)
