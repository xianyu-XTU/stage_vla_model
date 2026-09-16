"""Frozen INT4 OpenVLA multimodal feature extractor.

Imports are intentionally lazy so the action head and its tests work without
the heavyweight OpenVLA runtime installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from stage_vla.vla_teacher.image_preprocessing import preprocess_openvla_image

DEFAULT_MODEL_PATH = Path(r"D:\openvla\models\openvla-7b")
PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut: "


class FrozenOpenVLAFeatureExtractor:
    """Extract the final prompt-token feature; all OpenVLA weights stay frozen."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        image_preprocess: str = "none",
    ) -> None:
        self.model_path = Path(model_path)
        if image_preprocess not in ("none", "libero_official"):
            raise ValueError(f"unsupported OpenVLA image preprocessing mode: {image_preprocess}")
        self.image_preprocess = image_preprocess
        self.model: Any = None
        self.processor: Any = None
        self.torch: Any = None

    def load(self) -> "FrozenOpenVLAFeatureExtractor":
        if not self.model_path.is_dir():
            raise FileNotFoundError(self.model_path)
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

        if torch.cuda.is_available() is False:
            raise RuntimeError("INT4 OpenVLA feature extraction requires a CUDA GPU")
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        self.model.eval()
        self.model.requires_grad_(False)
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("OpenVLA freeze invariant failed")
        self.torch = torch
        return self

    def _input_device(self):
        mapping = getattr(self.model, "hf_device_map", None) or {}
        for value in mapping.values():
            if isinstance(value, int):
                return self.torch.device(f"cuda:{value}")
            if isinstance(value, str) and value.startswith("cuda"):
                return self.torch.device(value)
        return self.torch.device(self.model.device)

    @property
    def feature_dim(self) -> int:
        if self.model is None:
            raise RuntimeError("load() must be called before reading feature_dim")
        return int(self.model.config.text_config.hidden_size)

    def encode(self, images: Sequence[np.ndarray], instructions: Sequence[str]) -> np.ndarray:
        if self.model is None or self.processor is None:
            raise RuntimeError("feature extractor is not loaded")
        if len(images) != len(instructions) or not images:
            raise ValueError("images and instructions must be non-empty and have equal length")
        from PIL import Image

        torch = self.torch
        device = self._input_device()
        transformed = []
        prompts = []
        for image, instruction in zip(images, instructions, strict=True):
            array = np.asarray(image)
            if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
                raise ValueError("each image must be uint8 [H,W,3]")
            prepared = preprocess_openvla_image(array, mode=self.image_preprocess)
            pil = Image.fromarray(np.asarray(prepared, dtype=np.uint8)).convert("RGB")
            transformed.append(self.processor.image_processor.apply_transform(pil))
            prompts.append(PROMPT_TEMPLATE.format(instruction=instruction.lower().strip()))
        pixel_values = torch.stack(transformed).to(device=device, dtype=torch.bfloat16)
        tokens = self.processor.tokenizer(prompts, padding=True, return_tensors="pt")
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)
        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            final_hidden = output.hidden_states[-1]
            inserted_patch_tokens = final_hidden.shape[1] - input_ids.shape[1]
            final_indices = attention_mask.sum(dim=1) + inserted_patch_tokens - 1
            pooled = final_hidden[
                torch.arange(final_hidden.shape[0], device=final_hidden.device), final_indices
            ]
        result = pooled.float().cpu().numpy()
        if result.shape != (len(images), self.feature_dim) or not np.isfinite(result).all():
            raise RuntimeError(f"invalid OpenVLA feature output {result.shape}")
        return result
