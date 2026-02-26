"""Qwen2.5-VL 视觉编码器 — 自定义 ViT + PatchMerger (392×392)"""
import torch
from torch import Tensor
from encoders.base import BaseVisualEncoder
from encoders.registry import register_encoder


@register_encoder("qwen")
class QwenEncoder(BaseVisualEncoder):
    """
    加载 Qwen2.5-VL-3B 的视觉编码器（自定义 ViT + PatchMerger）。
    输入分辨率 392×392（= 28 × 14，patch_size=14，merge_size=2）。
    PatchMerger 将 784 个 tokens 合并为 196 个。
    显存占用：~600MB (bf16)
    """

    @property
    def name(self) -> str:
        return "qwen"

    @property
    def feature_dim(self) -> int:
        return 1536   # Qwen2.5-VL ViT 输出维度（merged tokens）

    def load(self) -> None:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration
        model_id = self.cfg.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
        dtype_str = self.cfg.get("dtype", "bf16")
        dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16

        print(f"[QwenEncoder] 加载视觉编码器: {model_id}")
        full_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto"
        )
        # 提取视觉编码器部分（包含 ViT + PatchMerger）
        self._visual = full_model.visual.eval()

        for p in self._visual.parameters():
            p.requires_grad_(False)

        self._model = self._visual
        print(f"[QwenEncoder] 加载完成")

    def encode(self, images: Tensor) -> Tensor:
        """
        通过 Qwen2.5-VL ViT + PatchMerger 提取特征，返回 [B, D]。
        注意：Qwen2.5-VL 的 visual 模块需要特定格式，这里做简化处理。
        """
        assert self._model is not None, "请先调用 load()"
        with torch.no_grad():
            # Qwen2.5-VL visual 的标准调用接口
            # pixel_values shape: [B, C, H, W]，已归一化
            out = self._visual(images)
            if isinstance(out, (tuple, list)):
                feats = out[0]
            elif hasattr(out, "last_hidden_state"):
                feats = out.last_hidden_state
            else:
                feats = out

        return feats.mean(dim=1) if feats.dim() == 3 else feats
