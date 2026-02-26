"""BLIP-2 视觉编码器 — EVA-CLIP ViT-G/14 + Q-Former"""
import torch
from torch import Tensor
from encoders.base import BaseVisualEncoder
from encoders.registry import register_encoder


@register_encoder("blip2")
class BLIP2Encoder(BaseVisualEncoder):
    """
    加载 BLIP-2 的视觉编码器部分（EVA-CLIP ViT-G + Q-Former）。
    Q-Former 天然支持 ITC（文本-图像对比），可用于文本引导的代理损失。
    显存占用：~2GB (bf16/fp16)
    """

    @property
    def name(self) -> str:
        return "blip2"

    @property
    def feature_dim(self) -> int:
        return 1408   # EVA-CLIP ViT-G 输出维度

    def load(self) -> None:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        import torch
        model_id = self.cfg.get("model_id", "Salesforce/blip2-opt-2.7b")
        dtype_str = self.cfg.get("dtype", "fp16")
        dtype = torch.float16 if dtype_str == "fp16" else torch.bfloat16

        print(f"[BLIP2Encoder] 加载视觉编码器: {model_id}")
        full_model = Blip2ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto"
        )
        # 只保留视觉编码器和 Q-Former，释放语言模型
        self._vision_model = full_model.vision_model.eval()
        self._qformer       = full_model.qformer.eval()
        self._query_tokens  = full_model.query_tokens
        self._vision_proj   = full_model.language_projection  # 备用

        # 冻结所有参数
        for p in self._vision_model.parameters(): p.requires_grad_(False)
        for p in self._qformer.parameters():       p.requires_grad_(False)

        self._model = self._vision_model  # 标记已加载
        print(f"[BLIP2Encoder] 加载完成")

    def encode(self, images: Tensor) -> Tensor:
        """通过 EVA-CLIP ViT-G 提取视觉特征，返回 [B, D]"""
        assert self._model is not None, "请先调用 load()"
        with torch.no_grad():
            vision_out = self._vision_model(pixel_values=images)
            image_embeds = vision_out.last_hidden_state  # [B, N_patches, D]
        return image_embeds.mean(dim=1)  # [B, D]

    def get_qformer_features(self, images: Tensor) -> Tensor:
        """通过 Q-Former 提取跨模态特征（用于 ITC 损失，支持文本对齐）"""
        assert self._model is not None, "请先调用 load()"
        B = images.shape[0]
        query_tokens = self._query_tokens.expand(B, -1, -1)
        vision_out   = self._vision_model(pixel_values=images)
        image_embeds = vision_out.last_hidden_state
        qformer_out  = self._qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            return_dict=True,
        )
        return qformer_out.last_hidden_state.mean(dim=1)  # [B, D_qformer]
