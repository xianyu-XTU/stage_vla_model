"""Franka stack action conventions used by diagnostics.

For Isaac Lab's BinaryJointAction float inputs:
- positive -> open command
- negative -> close command

The local M3.2-net probe also verified this convention on the user's runtime.
"""

GRIPPER_OPEN_ACTION = 1.0
GRIPPER_CLOSE_ACTION = -1.0
