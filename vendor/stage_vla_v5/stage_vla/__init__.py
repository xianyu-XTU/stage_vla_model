"""stage_vla_v5: action-first skill policies with replaceable vision/VLM adapters."""

__version__ = "0.5.0"

from .action_runtime import (
    ActionModelRouter, SkillModelSpec,
    V5ActionModelManifest,
    V5ActionModelRuntime,
    default_action_model_manifest,
    generalized_cube_action_model_manifest,
    irregular_cylinder_action_model_manifest,
)
from .action_output import (
    ActionOutput, ActionOutputModule, ActionSource, MappingActionSource,
    ObjectActionOutput, ObjectActionSource,
)
from .objects import (
    ACTION_CONTEXT_ORDER, ACTION_CONTEXT_VERSION, ObjectDomain, ObjectModule,
    ObjectModuleOutput, ObjectSpec, get_object_spec, object_labels,
    register_object_spec, rigid_cube_domain,
)
from .vision.recognition import ObjectRecognitionModule, RecognitionBackend, RecognitionResult, VisionFrame
from .pipeline import GeometryPlan, LoadPlan, ManipulationPlan, V5ManipulationPipeline
from .language_interface import DeterministicInstructionAdapter, InstructionRequest
from .task_dsl import (
    CatalogObjectNameResolver,
    InstructionParser,
    InstructionPlan,
    StackChainSpec,
    TaskSpec,
)

__all__ = [
    "SkillModelSpec",
    "ActionModelRouter",
    "V5ActionModelManifest",
    "V5ActionModelRuntime",
    "default_action_model_manifest",
    "generalized_cube_action_model_manifest",
    "irregular_cylinder_action_model_manifest",
    "ActionOutput",
    "ActionOutputModule",
    "ActionSource",
    "MappingActionSource",
    "ObjectActionOutput",
    "ObjectActionSource",
    "ACTION_CONTEXT_ORDER",
    "ACTION_CONTEXT_VERSION",
    "ObjectDomain",
    "ObjectModule",
    "ObjectModuleOutput",
    "ObjectSpec",
    "get_object_spec",
    "object_labels",
    "register_object_spec",
    "rigid_cube_domain",
    "ObjectRecognitionModule",
    "RecognitionBackend",
    "RecognitionResult",
    "VisionFrame",
    "GeometryPlan",
    "LoadPlan",
    "ManipulationPlan",
    "V5ManipulationPipeline",
    "DeterministicInstructionAdapter",
    "InstructionRequest",
    "CatalogObjectNameResolver",
    "InstructionParser",
    "InstructionPlan",
    "StackChainSpec",
    "TaskSpec",
]
