"""
Loss functions for ACMFT emotion recognition training.

This module implements various loss functions used for training the ACMFT model,
including cross-entropy loss, label smoothing, and auxiliary losses for regularization.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionClassificationLoss(nn.Module):
    """
    Standard cross-entropy loss for emotion classification with optional label smoothing.
    """

    def __init__(
        self,
        num_emotions: int = 8,
        label_smoothing: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_emotions = num_emotions
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss with optional label smoothing.

        Args:
            logits: Emotion logits (batch_size, num_emotions)
            targets: Ground truth emotion labels (batch_size,)

        Returns:
            loss: Classification loss
        """
        # Handle empty batch case
        if logits.size(0) == 0 or targets.size(0) == 0:
            # Create a zero tensor that requires grad
            return torch.zeros(1, device=logits.device, requires_grad=True)[0]

        loss = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
            reduction="mean",
        )

        return loss


class ModalityReconstructionLoss(nn.Module):
    """
    Reconstruction loss for modality representations.
    Used as an auxiliary loss to improve feature learning.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        reconstruction_weight: float = 0.1,
    ):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight

        # Decoders for each modality
        self.visual_decoder = nn.Linear(hidden_dim, hidden_dim)
        self.audio_decoder = nn.Linear(hidden_dim, hidden_dim)
        self.hr_decoder = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        fused_representation: torch.Tensor,
        original_visual: torch.Tensor,
        original_audio: torch.Tensor,
        original_hr: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute reconstruction loss for each modality.

        Args:
            fused_representation: Fused multimodal representation
            original_visual: Original visual features
            original_audio: Original audio features
            original_hr: Original HR features

        Returns:
            total_loss: Total reconstruction loss
            individual_losses: Dictionary of individual modality reconstruction losses
        """
        # Handle empty batch case
        if fused_representation.size(0) == 0:
            zero_loss = torch.zeros(1, device=fused_representation.device, requires_grad=True)[0]
            individual_losses = {
                "visual_recon_loss": 0.0,
                "audio_recon_loss": 0.0,
                "hr_recon_loss": 0.0,
            }
            return zero_loss, individual_losses

        # Reconstruct each modality from fused representation
        reconstructed_visual = self.visual_decoder(fused_representation)
        reconstructed_audio = self.audio_decoder(fused_representation)
        reconstructed_hr = self.hr_decoder(fused_representation)

        # Calculate MSE loss for each modality
        visual_loss = F.mse_loss(reconstructed_visual, original_visual)
        audio_loss = F.mse_loss(reconstructed_audio, original_audio)
        hr_loss = F.mse_loss(reconstructed_hr, original_hr)

        # Combine losses
        total_loss = (visual_loss + audio_loss + hr_loss) * self.reconstruction_weight

        # Return total loss and individual losses
        individual_losses = {
            "visual_recon_loss": visual_loss.item(),
            "audio_recon_loss": audio_loss.item(),
            "hr_recon_loss": hr_loss.item(),
        }

        return total_loss, individual_losses


class KLDivergenceLoss(nn.Module):
    """
    KL Divergence loss for soft targets.
    Useful when training with ensemble or teacher-student setup.
    """

    def __init__(
        self,
        temperature: float = 2.0,
        kl_weight: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.kl_weight = kl_weight

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL Divergence loss.

        Args:
            student_logits: Student model logits (batch_size, num_classes)
            teacher_probs: Teacher model probabilities (batch_size, num_classes)

        Returns:
            loss: KL Divergence loss
        """
        # Scale logits by temperature
        scaled_logits = student_logits / self.temperature

        # Student probabilities
        # FIXME
        student_probs = F.softmax(scaled_logits, dim=1)

        # KL Divergence loss
        loss = (
            F.kl_div(
                F.log_softmax(scaled_logits, dim=1),
                teacher_probs,
                reduction="batchmean",
            )
            * (self.temperature**2)
            * self.kl_weight
        )

        return loss


class ACMFTLoss(nn.Module):
    """
    Combined loss function for ACMFT model training.
    Includes classification loss, optional reconstruction loss, and regularization.
    """

    def __init__(
        self,
        num_emotions: int = 8,
        label_smoothing: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
        use_reconstruction_loss: bool = False,
        reconstruction_weight: float = 0.1,
        l2_reg_weight: float = 1e-5,
    ):
        super().__init__()

        # Main classification loss
        self.classification_loss = EmotionClassificationLoss(
            num_emotions=num_emotions,
            label_smoothing=label_smoothing,
            class_weights=class_weights,
        )

        # Optional reconstruction loss
        self.use_reconstruction_loss = use_reconstruction_loss
        if use_reconstruction_loss:
            self.reconstruction_loss = ModalityReconstructionLoss(
                reconstruction_weight=reconstruction_weight,
            )

        # L2 regularization weight
        self.l2_reg_weight = l2_reg_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module,
        fused_representation: Optional[torch.Tensor] = None,
        original_visual: Optional[torch.Tensor] = None,
        original_audio: Optional[torch.Tensor] = None,
        original_hr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss for ACMFT training.

        Args:
            logits: Emotion logits (batch_size, num_emotions)
            targets: Ground truth emotion labels (batch_size,)
            model: ACMFT model for L2 regularization
            fused_representation: Fused multimodal representation (for reconstruction loss)
            original_visual: Original visual features (for reconstruction loss)
            original_audio: Original audio features (for reconstruction loss)
            original_hr: Original HR features (for reconstruction loss)

        Returns:
            total_loss: Combined loss for optimization
            loss_dict: Dictionary of individual loss components
        """
        # Handle empty batch case
        if logits.size(0) == 0 or targets.size(0) == 0:
            # Create a zero tensor that requires grad
            zero_loss = torch.zeros(1, device=logits.device, requires_grad=True)[0]
            empty_loss_dict = {"classification_loss": 0.0, "total_loss": 0.0}
            if self.use_reconstruction_loss:
                empty_loss_dict.update({
                    "visual_recon_loss": 0.0,
                    "audio_recon_loss": 0.0,
                    "hr_recon_loss": 0.0,
                    "reconstruction_loss": 0.0
                })
            if self.l2_reg_weight > 0:
                empty_loss_dict["l2_regularization"] = 0.0
            return zero_loss, empty_loss_dict

        # Classification loss
        cls_loss = self.classification_loss(logits, targets)

        # Initialize loss components
        loss_dict = {"classification_loss": cls_loss.item()}
        total_loss = cls_loss

        # Reconstruction loss (if enabled)
        if self.use_reconstruction_loss and fused_representation is not None:
            recon_loss, recon_components = self.reconstruction_loss(
                fused_representation,
                original_visual,
                original_audio,
                original_hr,
            )
            total_loss = total_loss + recon_loss
            loss_dict.update(recon_components)
            loss_dict["reconstruction_loss"] = recon_loss.item()

        # L2 regularization
        if self.l2_reg_weight > 0:
            l2_reg = torch.tensor(0.0, device=logits.device)
            for param in model.parameters():
                l2_reg = l2_reg + torch.norm(param, p=2)
            l2_reg = l2_reg * self.l2_reg_weight

            total_loss = total_loss + l2_reg
            loss_dict["l2_regularization"] = l2_reg.item()

        # Update total loss
        loss_dict["total_loss"] = total_loss.item()

        return total_loss, loss_dict
