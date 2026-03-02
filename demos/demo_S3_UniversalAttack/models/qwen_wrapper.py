"""
Qwen2.5-VL wrapper for Universal Adversarial Attack.

Implements the MLLMWrapper interface for Qwen2-VL / Qwen2.5-VL models.
Handles the Qwen-specific chat template and pixel_values construction.
"""

import os
import sys
from typing import Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from model_registry import get_model_info

from .mllm_wrapper import MLLMWrapper


class QwenWrapper(MLLMWrapper):
    """Wrapper for Qwen2-VL / Qwen2.5-VL models."""

    def load(self):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        info = get_model_info(self.model_key)
        hf_id = info["hf_id"]
        dtype = torch.bfloat16 if info["dtype"] == "bf16" else torch.float16

        print(f"Loading {info['short_name']} ({hf_id})...")
        self.processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            hf_id, torch_dtype=dtype, trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        # Freeze everything but allow gradient flow through vision encoder
        for param in self.model.parameters():
            param.requires_grad = False

        self.dtype = dtype
        self.img_size = info["img_size"]
        print(f"  Loaded. VRAM: ~{info['vram_bf16_gb']} GB")

    def _build_inputs(self, image: torch.Tensor, question: str, target_answer: Optional[str] = None):
        """
        Build Qwen2-VL input_ids, pixel_values, attention_mask, and labels
        from a raw image tensor and text.

        The image tensor bypasses the processor's image loading -- we inject
        pixel_values directly to maintain gradient flow.
        """
        # Build conversation messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]

        # Apply chat template to get text tokens
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if target_answer:
            text_with_answer = text + target_answer
        else:
            text_with_answer = text

        # Tokenize
        text_inputs = self.processor.tokenizer(
            text_with_answer, return_tensors="pt", padding=True
        ).to(self.device)

        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]

        # Construct pixel_values from image tensor
        # Qwen2-VL expects pixel_values as (num_patches, 3, patch_h, patch_w)
        # We resize the image to the expected size and create grid
        img = image.to(self.device, dtype=self.dtype)  # (1, 3, H, W)
        if img.shape[2:] != (self.img_size, self.img_size):
            img = F.interpolate(img, size=(self.img_size, self.img_size),
                                mode="bilinear", align_corners=False)

        # Qwen2-VL patch size is 14, temporal_patch_size=2
        patch_size = 14
        merge_size = 2
        h_patches = self.img_size // patch_size
        w_patches = self.img_size // patch_size

        # Reshape to patches: (1, 3, H, W) -> (num_patches, 3*temporal, patch_h, patch_w)
        # For single image: temporal=1, we duplicate to match temporal_patch_size=2
        img_doubled = torch.cat([img, img], dim=0)  # (2, 3, H, W)
        patches = img_doubled.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # patches: (2, 3, h_patches, w_patches, patch_size, patch_size)
        patches = patches.permute(2, 3, 0, 1, 4, 5).contiguous()
        # (h_patches, w_patches, 2, 3, patch_size, patch_size)
        patches = patches.reshape(h_patches * w_patches, 2 * 3, patch_size, patch_size)

        pixel_values = patches  # (num_patches, 6, 14, 14)

        # Image grid thw for Qwen2-VL
        image_grid_thw = torch.tensor([[1, h_patches, w_patches]],
                                      dtype=torch.long, device=self.device)

        # Build labels: mask everything except target answer tokens
        labels = None
        if target_answer:
            labels = input_ids.clone()
            # Tokenize just the prompt (without answer) to find split point
            prompt_only = self.processor.tokenizer(text, return_tensors="pt")
            prompt_len = prompt_only["input_ids"].shape[1]
            labels[:, :prompt_len] = -100  # mask prompt tokens

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels": labels,
        }

    def compute_masked_ce_loss(self, image: torch.Tensor,
                               question: str, target_answer: str) -> torch.Tensor:
        inputs = self._build_inputs(image, question, target_answer)
        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            labels=inputs["labels"],
        )
        return outputs.loss

    @torch.no_grad()
    def generate(self, image: torch.Tensor, question: str,
                 max_new_tokens: int = 100) -> str:
        inputs = self._build_inputs(image, question)

        output_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"].to(self.dtype),
            image_grid_thw=inputs["image_grid_thw"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        # Decode only the generated tokens
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(generated[0], skip_special_tokens=True)
