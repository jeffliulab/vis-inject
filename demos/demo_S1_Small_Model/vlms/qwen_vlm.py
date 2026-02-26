"""Qwen2.5-VL 完整 VLM 实现"""
from PIL import Image
from vlms.base import BaseVLM
from vlms.registry import register_vlm


@register_vlm("qwen")
class QwenVLM(BaseVLM):

    @property
    def name(self) -> str:
        return "qwen"

    def load(self) -> None:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        model_id = self.cfg.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
        dtype = torch.bfloat16 if self.cfg.get("dtype", "bf16") == "bf16" else torch.float16
        print(f"[QwenVLM] 加载模型: {model_id}")
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto"
        ).eval()
        print(f"[QwenVLM] 加载完成")

    def generate(self, image: Image.Image, question: str,
                 max_new_tokens: int = None) -> str:
        import torch
        from qwen_vl_utils import process_vision_info
        assert self._model is not None, "请先调用 load()"
        max_tokens = max_new_tokens or self.max_new_tokens

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(self._model.device)

        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=max_tokens)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
