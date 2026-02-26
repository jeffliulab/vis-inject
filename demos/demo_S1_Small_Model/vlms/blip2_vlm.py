"""BLIP-2 完整 VLM 实现"""
from PIL import Image
from vlms.base import BaseVLM
from vlms.registry import register_vlm


@register_vlm("blip2")
class BLIP2VLM(BaseVLM):

    @property
    def name(self) -> str:
        return "blip2"

    def load(self) -> None:
        import torch
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        model_id = self.cfg.get("model_id", "Salesforce/blip2-opt-2.7b")
        dtype = torch.float16 if self.cfg.get("dtype", "fp16") == "fp16" else torch.bfloat16
        print(f"[BLIP2VLM] 加载模型: {model_id}")
        self._processor = Blip2Processor.from_pretrained(model_id)
        self._model = Blip2ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto"
        ).eval()
        print(f"[BLIP2VLM] 加载完成")

    def generate(self, image: Image.Image, question: str,
                 max_new_tokens: int = None) -> str:
        import torch
        assert self._model is not None, "请先调用 load()"
        max_tokens = max_new_tokens or self.max_new_tokens
        prompt = f"Question: {question} Answer:"
        inputs = self._processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=max_tokens)
        return self._processor.decode(out[0], skip_special_tokens=True).strip()
