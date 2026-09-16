"""OpenVLA teacher pipeline (frozen teacher, inference/data generation only)."""
from .openvla_runner import OpenVLATeacher, DEFAULT_MODEL_PATH, PROMPT_TEMPLATE
from .dataset import TeacherSample, TeacherDatasetBuilder

__all__ = [
    "OpenVLATeacher",
    "DEFAULT_MODEL_PATH",
    "PROMPT_TEMPLATE",
    "TeacherSample",
    "TeacherDatasetBuilder",
]
