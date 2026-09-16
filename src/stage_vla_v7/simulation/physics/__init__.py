"""Physics descriptions independent of Isaac Lab classes."""

from .contacts import FRANKA_FINGER_CONTACTS, ContactSensorBinding
from .limits import ControlLimits, RigidBodyLimits
from .materials import PhysicsMaterial

__all__ = [
    "ContactSensorBinding",
    "ControlLimits",
    "FRANKA_FINGER_CONTACTS",
    "PhysicsMaterial",
    "RigidBodyLimits",
]
