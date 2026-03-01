"""Qwen2.5-VL 视觉编码器 — 自定义 ViT + PatchMerger (392×392)

架构说明:
  Qwen2_5_VLForConditionalGeneration
    └── .model  (Qwen2_5_VLModel)
          └── .visual  (Qwen2_5_VLVisionTransformer)
                ├── ViT layers
                └── PatchMerger → 输出 [N_merged, embed_dim]

图像预处理:
  [B, 3, H, W] → patch 展开 → [N_patches, C*T*pH*pW]
  grid_thw = [[1, H//patch_size, W//patch_size]]  (temporal=1)
"""
import torch
from torch import Tensor
from encoders.base import BaseVisualEncoder
from encoders.registry import register_encoder

# Qwen2.5-VL 视觉归一化参数（CLIP 风格）
_MEAN = [0.48145466, 0.4578275,  0.40821073]
_STD  = [0.26862954, 0.26130258, 0.27577711]

# 视觉编码器配置常量（与 demo3 保持一致）
_PATCH_SIZE          = 14
_MERGE_SIZE          = 2
_TEMPORAL_PATCH_SIZE = 2


@register_encoder("qwen")
class QwenEncoder(BaseVisualEncoder):
    """
    加载 Qwen2.5-VL-3B 的视觉编码器（自定义 ViT + PatchMerger）。
    输入分辨率 392×392（= 28 × 14，patch_size=14，merge_size=2）。
    PatchMerger 将 784 个 tokens 合并为 196 个。
    显存占用：~1.5GB (bf16，含完整模型但仅视觉部分参与计算)
    """

    @property
    def name(self) -> str:
        return "qwen"

    @property
    def feature_dim(self) -> int:
        # last_hidden_state via BaseModelOutputWithPooling → ViT hidden dim (pre-merger)
        return 1280

    def load(self) -> None:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration
        model_id = self.cfg.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
        dtype_str = self.cfg.get("dtype", "bf16")
        self._dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16

        print(f"[QwenEncoder] 加载视觉编码器: {model_id}")
        full_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=self._dtype, device_map="auto"
        )
        # 正确路径: ForConditionalGeneration → .model → .visual
        # 提取视觉编码器并立即释放完整模型（LLM 部分），节省 ~4GB 显存
        visual_module = full_model.model.visual
        full_model.model.visual = None   # 断开引用防止循环
        del full_model
        torch.cuda.empty_cache()

        self._visual = visual_module.eval()

        for p in self._visual.parameters():
            p.requires_grad_(False)

        self._model = self._visual

        # 预计算归一化张量（放到与模型相同的 device）
        dev = next(self._visual.parameters()).device
        self._img_mean = torch.tensor(_MEAN, device=dev, dtype=torch.float32).view(1, 3, 1, 1)
        self._img_std  = torch.tensor(_STD,  device=dev, dtype=torch.float32).view(1, 3, 1, 1)

        print(f"[QwenEncoder] 加载完成，device={dev}")

    def _image_to_pixel_values(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """
        [B, 3, H, W] (float32, [0,1]) → pixel_values [N_patches, C*T*pH*pW]
        并生成对应的 grid_thw [[T, H//14, W//14]] × B
        """
        dev = next(self._visual.parameters()).device
        x = images.to(dev, dtype=torch.float32)

        # 归一化
        x = (x - self._img_mean) / self._img_std
        x = x.to(self._dtype)

        B, C, H, W = x.shape
        PS = _PATCH_SIZE
        MS = _MERGE_SIZE
        T  = _TEMPORAL_PATCH_SIZE

        h_merge = H // (PS * MS)
        w_merge = W // (PS * MS)

        # patch 展开：[B, C, h_merge, MS, PS, w_merge, MS, PS]
        x = x.reshape(B, C, h_merge, MS, PS, w_merge, MS, PS)
        # → [B, h_merge, w_merge, MS, MS, C, PS, PS]
        x = x.permute(0, 2, 5, 3, 6, 1, 4, 7)
        # 扩展时间维 T
        x = x.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, T, -1, -1)
        # → [B * h_merge * w_merge * MS * MS, C * T * PS * PS]
        pixel_values = x.reshape(-1, C * T * PS * PS)

        # grid_thw: [B, 3]  每行 = [T=1, H//PS, W//PS]
        grid_thw = torch.tensor(
            [[1, H // PS, W // PS]] * B,
            dtype=torch.long, device=dev
        )

        return pixel_values, grid_thw

    def resize_and_encode(self, images: Tensor) -> Tensor:
        """
        完整流程覆盖：直接接受 [0,1] 图像，resize 后传入 _image_to_pixel_values
        （该方法内部自带 Qwen 专用归一化，不需要 base class 的 normalize() 步骤）
        """
        import torch.nn.functional as F
        if images.shape[-2] != self.img_size or images.shape[-1] != self.img_size:
            images = F.interpolate(images.float(), size=(self.img_size, self.img_size),
                                   mode="bilinear", align_corners=False)
        return self.encode(images)

    def encode(self, images: Tensor) -> Tensor:
        """
        [B, 3, 392, 392] (值域 [0,1]) → 特征 [B, D]

        注意：encode() 直接接受 [0,1] 的图像（由 _image_to_pixel_values 内部归一化）。
        不应在外部先调用 normalize()，否则会双重归一化。
        """
        assert self._model is not None, "请先调用 load()"

        pixel_values, grid_thw = self._image_to_pixel_values(images)

        # 不使用 torch.no_grad()，允许梯度流过用于 proxy loss 训练
        # （encoder 参数已冻结，不会被更新；只需要梯度流回 adv_img）
        out = self._visual(pixel_values, grid_thw=grid_thw)

        # 提取张量（可能返回 BaseModelOutputWithPooling 或直接 Tensor）
        if isinstance(out, torch.Tensor):
            feats = out
        elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            feats = out.last_hidden_state
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            # pooler_output 已是 [B, D]
            return out.pooler_output.float()
        else:
            # 尝试 .hidden_states[-1]
            feats = out.hidden_states[-1] if hasattr(out, "hidden_states") else out[0]

        # feats 可能是 [N_merged_total, D] 或 [B, tokens, D]
        B = images.shape[0]
        if feats.dim() == 2:
            # [N_merged_total, D]：所有 batch 的 token 拼在一起
            tokens_per_img = feats.shape[0] // B
            feats = feats.reshape(B, tokens_per_img, -1)

        return feats.mean(dim=1).float()  # [B, D]
