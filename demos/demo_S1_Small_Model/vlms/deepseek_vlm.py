"""DeepSeek-VL 完整 VLM 实现"""
from PIL import Image
from vlms.base import BaseVLM
from vlms.registry import register_vlm


@register_vlm("deepseek")
class DeepSeekVLM(BaseVLM):

    @property
    def name(self) -> str:
        return "deepseek"

    def load(self) -> None:
        import torch
        model_id = self.cfg.get("model_id", "deepseek-ai/deepseek-vl-1.3b-chat")
        dtype = torch.bfloat16 if self.cfg.get("dtype", "bf16") == "bf16" else torch.float16
        print(f"[DeepSeekVLM] 加载模型: {model_id}")
        try:
            from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
            from deepseek_vl.utils.io import load_pil_images
            self._processor = VLChatProcessor.from_pretrained(model_id)
            self._model = MultiModalityCausalLM.from_pretrained(
                model_id, torch_dtype=dtype
            ).to(self.device).eval()
            self._use_deepseek_pkg = True
        except ImportError:
            # 回退到 transformers AutoModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=dtype, trust_remote_code=True,
                device_map="auto"
            ).eval()
            self._use_deepseek_pkg = False
        print(f"[DeepSeekVLM] 加载完成")

    def generate(self, image: Image.Image, question: str,
                 max_new_tokens: int = None) -> str:
        import torch
        assert self._model is not None, "请先调用 load()"
        max_tokens = max_new_tokens or self.max_new_tokens

        if self._use_deepseek_pkg:
            conversation = [{
                "role": "User",
                "content": f"<image_placeholder>{question}",
                "images": [image],
            }, {"role": "Assistant", "content": ""}]
            pil_images = [image]
            inputs = self._processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True
            ).to(self.device)
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                )
            return self._processor.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
        else:
            # 通用回退路径
            inputs = self._processor(
                text=question, images=image, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                out = self._model.generate(**inputs, max_new_tokens=max_tokens)
            return self._processor.decode(out[0], skip_special_tokens=True).strip()
