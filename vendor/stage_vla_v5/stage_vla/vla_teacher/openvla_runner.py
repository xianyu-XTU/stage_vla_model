"""M14/M15 OpenVLA Teacher inference interface.

Teacher model is frozen and only used for inference. This is the M12-verified
loader: 4-bit NF4 + double quant + bfloat16 compute + eager attention, using
``image_processor.apply_transform`` for pixels and ``model.predict_action``
(not ``generate``) with the OpenVLA prompt.

Verified on RTX4060 8GB (M12, tools/m12_openvla_teacher_test.py):
- model_load PASS, vram_peak ~4 GB, ~0.6 s/sample.

CRITICAL ENV REQUIREMENT: must run with transformers 4.40.1 (the version the
model was saved with). The kit python ships transformers 4.57.6, which breaks
OpenVLA's prismatic remote code -> the model outputs a constant median action
regardless of image/instruction. Run with the compat shadow env:

    PYTHONPATH=E:\\openvla_compat  <kit-python> tools/m15_openvla_distill.py ...

E:\\openvla_compat contains transformers==4.40.1, tokenizers==0.19.1 and
accelerate==0.29.0 (installed with --target so the kit python is untouched).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

from stage_vla.vla_teacher.image_preprocessing import preprocess_openvla_image

DEFAULT_MODEL_PATH = r"D:\openvla\models\openvla-7b"
PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut: "


class OpenVLATeacher:
    """Frozen OpenVLA-7B teacher. Inference only, no training / no weight edits."""

    def __init__(
        self,
        model_path: str | None = None,
        load_4bit: bool = True,
        *,
        unnorm_key: str | None = None,
        image_preprocess: str = "none",
    ):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.load_4bit = load_4bit
        self.unnorm_key = unnorm_key
        if image_preprocess not in ("none", "libero_official"):
            raise ValueError(f"unsupported OpenVLA image preprocessing mode: {image_preprocess}")
        self.image_preprocess = image_preprocess
        self.model: Any = None
        self.processor: Any = None

    # ------------------------------------------------------------------ load

    def load(self):
        """Load OpenVLA-7B with the M12-verified 4-bit configuration."""
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        quant = BitsAndBytesConfig(
            load_in_4bit=self.load_4bit,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quant,
            attn_implementation="eager",
        )
        self.model.eval()
        if self.unnorm_key is not None and self.unnorm_key not in self.model.norm_stats:
            raise KeyError(f"OpenVLA unnorm key is not present in model: {self.unnorm_key}")
        return self

    def prepare_image(self, image: Image.Image | np.ndarray) -> Image.Image:
        """Return the exact image that will be passed to the model processor."""
        return preprocess_openvla_image(image, mode=self.image_preprocess)

    # ------------------------------------------------------------ inference

    def predict_action(
        self,
        image: Image.Image | np.ndarray,
        instruction: str,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> list[float]:
        """Predict the 7-D action for one image + instruction.

        Uses ``model.predict_action`` (OpenVLA norm/unnorm pipeline), not
        ``generate``. ``image`` may be a PIL image or a uint8 numpy array.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Teacher model is not loaded. Call load() first.")
        if self.unnorm_key is None:
            raise RuntimeError(
                "OpenVLA unnorm_key must be explicit; v4.1 rejected automatic dataset-key selection"
            )

        # Under the validated OpenVLA compatibility environment (transformers 4.40.1),
        # apply_transform accepts PIL/ndarray and returns [6,224,224]
        # (channel-stacked image + fused image, no batch dim).
        # The fused vision backbone expects [1,6,224,224]; without the batch dim
        # the multimodal forward breaks and the action collapses to a constant --
        # the M15 degenerate-output root cause.
        img = self.prepare_image(image)

        tokenizer = self.processor.tokenizer
        image_processor = self.processor.image_processor

        prompt = PROMPT_TEMPLATE.format(instruction=instruction.lower().strip())
        pixel_values = image_processor.apply_transform(img)  # [6,224,224]
        pixel_values = pixel_values.unsqueeze(0)  # [1,6,224,224]
        pixel_values = pixel_values.to(dtype=torch.bfloat16, device=self.model.device)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)

        with torch.inference_mode():
            action = self.model.predict_action(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                unnorm_key=self.unnorm_key,
                do_sample=do_sample,
                temperature=temperature,
            )
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (7,):
            raise RuntimeError(f"OpenVLA predict_action must return 7 values, got {action.shape}")
        if not np.isfinite(action).all():
            raise RuntimeError(f"OpenVLA predict_action returned non-finite values: {action.tolist()}")
        return action.tolist()
