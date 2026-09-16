"""Independent definitions for the eight canonical V7 skills."""

from .align import DEFINITION as ALIGN
from .descend import DEFINITION as DESCEND
from .grasp import DEFINITION as GRASP
from .lift import DEFINITION as LIFT
from .reach import DEFINITION as REACH
from .release_stabilize import DEFINITION as RELEASE_STABILIZE
from .retreat import DEFINITION as RETREAT
from .transport import DEFINITION as TRANSPORT

ALL_DEFINITIONS = (REACH, GRASP, LIFT, TRANSPORT, ALIGN, DESCEND, RELEASE_STABILIZE, RETREAT)

__all__ = [
    "ALIGN",
    "ALL_DEFINITIONS",
    "DESCEND",
    "GRASP",
    "LIFT",
    "REACH",
    "RELEASE_STABILIZE",
    "RETREAT",
    "TRANSPORT",
]
