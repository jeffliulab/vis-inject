"""
Loss functions for AnyAttack training.

- DynamicInfoNCELoss: Self-supervised pre-training (contrastive with temperature annealing)
- BiContrastiveLoss:  Bidirectional contrastive loss for fine-tuning
- DirectMatchingLoss: Cosine similarity maximization for fine-tuning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicInfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss with exponential temperature annealing.
    Used during self-supervised pre-training on LAION-Art.

    Temperature decays from initial_temp to final_temp over total_batches
    to progressively sharpen the similarity distribution.
    """

    def __init__(self, initial_temp: float = 1.0, final_temp: float = 0.07,
                 total_batches: int = 10000):
        super().__init__()
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.total_batches = total_batches
        self.current_temp = initial_temp
        self._decay_rate = -np.log(final_temp / initial_temp) / total_batches

    def forward(self, embeddings1: torch.Tensor,
                embeddings2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings1: (B, D) original image embeddings (targets).
            embeddings2: (B, D) adversarial image embeddings.

        Returns:
            Scalar contrastive loss.
        """
        e1 = F.normalize(embeddings1, p=2, dim=1)
        e2 = F.normalize(embeddings2, p=2, dim=1)
        sim = torch.matmul(e1, e2.T) / self.current_temp
        labels = torch.arange(sim.size(0), device=sim.device)
        return F.cross_entropy(sim, labels)

    def update_temperature(self, batch_count: int):
        """Call after each batch to anneal temperature."""
        self.current_temp = self.initial_temp * np.exp(-self._decay_rate * batch_count)


class BiContrastiveLoss(nn.Module):
    """
    Bidirectional contrastive loss: image-to-text + text-to-image.
    Used during fine-tuning on COCO.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_features: torch.Tensor,
                text_features: torch.Tensor) -> tuple:
        """
        Returns:
            (loss, avg_similarity) tuple.
        """
        img = F.normalize(image_features, p=2, dim=1)
        txt = F.normalize(text_features, p=2, dim=1)
        sim = torch.matmul(img, txt.T) / self.temperature
        labels = torch.eye(sim.size(0), device=sim.device)

        loss_i2t = -torch.sum(labels * F.log_softmax(sim, dim=1), dim=1).mean()
        loss_t2i = -torch.sum(labels * F.log_softmax(sim.T, dim=1), dim=1).mean()

        avg_sim = torch.diag(torch.matmul(img, txt.T)).mean()
        return (loss_i2t + loss_t2i) / 2, avg_sim


class DirectMatchingLoss(nn.Module):
    """
    Direct cosine similarity maximization.
    Simpler alternative to contrastive loss for fine-tuning.
    """

    def forward(self, image_features: torch.Tensor,
                text_features: torch.Tensor) -> tuple:
        """
        Returns:
            (loss, avg_similarity) tuple. Loss is negative mean cosine similarity.
        """
        cos_sim = F.cosine_similarity(image_features, text_features, dim=-1)
        return -cos_sim.mean(), cos_sim.mean()
