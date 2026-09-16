"""M14 Semantic Action Representation.

Stage Token + Continuous Action Token. Lossless encode/decode of the 7-D
continuous expert action while carrying the stage label as a discrete semantic
token, so the fine-grained control information is preserved for the Student VLA.
"""

from .tokenizer import (
    ACTION_DIM,
    GRIP_INDEX,
    ID_TO_STAGE,
    NUM_STAGES,
    POS_SLICE,
    ROT_SLICE,
    STAGE_NAMES,
    STAGE_TO_ID,
    VECTOR_DIM,
    SemanticActionToken,
    SemanticActionTokenizer,
)
from .dataset_converter import convert_m13_to_m14

__all__ = [
    "ACTION_DIM",
    "GRIP_INDEX",
    "ID_TO_STAGE",
    "NUM_STAGES",
    "POS_SLICE",
    "ROT_SLICE",
    "STAGE_NAMES",
    "STAGE_TO_ID",
    "VECTOR_DIM",
    "SemanticActionToken",
    "SemanticActionTokenizer",
    "convert_m13_to_m14",
]
