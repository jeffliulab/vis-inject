"""DeepSeek-VL 视觉编码器 — SigLIP-L/16 (384×384)"""
import torch
from torch import Tensor
from encoders.base import BaseVisualEncoder
from encoders.registry import register_encoder


@register_encoder("deepseek")
class DeepSeekEncoder(BaseVisualEncoder):
    """
    加载 DeepSeek-VL 的视觉编码器（SigLIP-L/16）。
    输入分辨率 384×384，标准化参数 mean=std=0.5。
    显存占用：~1GB (fp16)
    """

    @property
    def name(self) -> str:
        return "deepseek"

    @property
    def feature_dim(self) -> int:
        return 1024   # SigLIP-L 输出维度

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_id = self.cfg.get("model_id", "deepseek-ai/deepseek-vl-1.3b-chat")
        dtype_str = self.cfg.get("dtype", "fp16")
        dtype = torch.float16 if dtype_str == "fp16" else torch.bfloat16

        print(f"[DeepSeekEncoder] 加载视觉编码器: {model_id}")
        try:
            # 尝试通过 deepseek_vl 包加载
            from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
            full_model = MultiModalityCausalLM.from_pretrained(
                model_id, torch_dtype=dtype
            ).to(self.device)
            self._vision_model = full_model.vision_model.eval()
        except ImportError:
            # 回退：通过 transformers AutoModel 加载视觉塔
            from transformers import AutoModel
            full_model = AutoModel.from_pretrained(
                model_id, torch_dtype=dtype, trust_remote_code=True
            )
            # DeepSeek-VL 的视觉塔在 vision_tower 或 model.vision_tower 下
            if hasattr(full_model, "vision_tower"):
                self._vision_model = full_model.vision_tower.eval()
            elif hasattr(full_model, "model") and hasattr(full_model.model, "vision_tower"):
                self._vision_model = full_model.model.vision_tower.eval()
            else:
                raise RuntimeError("无法找到 DeepSeek-VL 的视觉编码器，请安装 deepseek_vl 包")

        for p in self._vision_model.parameters():
            p.requires_grad_(False)

        self._model = self._vision_model
        print(f"[DeepSeekEncoder] 加载完成")

    def encode(self, images: Tensor) -> Tensor:
        """通过 SigLIP-L 提取视觉特征，返回 [B, D]"""
        assert self._model is not None, "请先调用 load()"
        with torch.no_grad():
            out = self._vision_model(images)
            if hasattr(out, "last_hidden_state"):
                feats = out.last_hidden_state
            elif hasattr(out, "pooler_output"):
                return out.pooler_output  # [B, D]
            else:
                feats = out[0] if isinstance(out, (tuple, list)) else out
        return feats.mean(dim=1) if feats.dim() == 3 else feats
