"""
Dynamic Contextual Gating implementation for ACMFT.

This module implements the dynamic contextual gating mechanism that
adaptively weighs the importance of each modality (visual, audio, HR)
based on the current context and reliability of the modalities.
"""

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ContextEncoder(nn.Module):
    """
    Encodes the context from multimodal features for gating.
    Uses average pooling followed by linear layers.
    """

    def __init__(self, hidden_dim: int, context_dim: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)  # Pool across sequence length
        self.fc1 = nn.Linear(hidden_dim, context_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(context_dim, hidden_dim)  # Output matches modality dim

    def forward(
        self,
        visual_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        hr_feat: torch.Tensor,
        visual_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        hr_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for context encoding.

        Args:
            visual_feat: Visual features (batch, seq_len, hidden_dim)
            audio_feat: Audio features (batch, seq_len, hidden_dim)
            hr_feat: HR features (batch, seq_len, hidden_dim)
            visual_mask: Mask for valid visual samples (batch,)
            audio_mask: Mask for valid audio samples (batch,)
            hr_mask: Mask for valid hr samples (batch,)

        Returns:
            Dictionary containing encoded context for each modality.
        """
        encoded_contexts = {}
        logger.debug(
            f"ContextEncoder - Input shapes: visual={visual_feat.shape if visual_feat is not None else None}, "
            f"audio={audio_feat.shape if audio_feat is not None else None}, "
            f"hr={hr_feat.shape if hr_feat is not None else None}"
        )
        logger.debug(
            f"ContextEncoder - Masks: V={visual_mask.sum().item()}, A={audio_mask.sum().item()}, H={hr_mask.sum().item()}"
        )

        for name, feat, mask in [
            ("visual", visual_feat, visual_mask),
            ("audio", audio_feat, audio_mask),
            ("hr", hr_feat, hr_mask),
        ]:
            if feat is not None and mask.any():
                # Permute for pooling: (batch, hidden_dim, seq_len)
                pooled = self.pool(feat.permute(0, 2, 1)).squeeze(-1)  # (batch, hidden_dim)
                logger.debug(f"ContextEncoder - Pooled {name} shape: {pooled.shape}")

                # Apply layers
                encoded = self.fc2(self.relu(self.fc1(pooled)))  # (batch, hidden_dim)
                logger.debug(f"ContextEncoder - Encoded {name} shape: {encoded.shape}")

                # Zero out context for invalid samples
                encoded = encoded * mask.unsqueeze(-1).float()  # Ensure mask is broadcastable
                encoded_contexts[name] = encoded
            else:
                # If modality is entirely missing or masked out, provide zero context
                # Need to know the hidden_dim
                if feat is not None:  # Use feat's dim if available
                    h_dim = feat.shape[-1]
                    b_dim = feat.shape[0]
                else:  # Infer from other features or default
                    other_feats = [
                        f
                        for f_name, f, m in [
                            ("visual", visual_feat, visual_mask),
                            ("audio", audio_feat, audio_mask),
                            ("hr", hr_feat, hr_mask),
                        ]
                        if f is not None
                    ]
                    if other_feats:
                        h_dim = other_feats[0].shape[-1]
                        b_dim = other_feats[0].shape[0]
                    else:
                        # This case should ideally not happen if at least one modality exists
                        logger.warning("ContextEncoder: Cannot infer hidden_dim, defaulting to a common value (e.g., 256)")
                        h_dim = 256  # Default guess
                        b_dim = visual_mask.shape[0]  # Get batch dim from mask

                encoded_contexts[name] = torch.zeros(
                    b_dim,
                    h_dim,
                    device=visual_mask.device
                    if visual_mask is not None
                    else (audio_mask.device if audio_mask is not None else hr_mask.device),
                )
                logger.debug(f"ContextEncoder - Zero context for {name} shape: {encoded_contexts[name].shape}")

        return encoded_contexts


class ModalityQualityEstimator(nn.Module):
    """
    Estimates the quality/reliability of each modality.
    This helps determine modality weights in noisy or uncertain conditions.
    """

    def __init__(self, hidden_dim: int, quality_dim: int = 64, dropout: float = 0.1):
        super().__init__()

        # Quality estimator for each modality
        self.visual_quality = nn.Sequential(
            nn.Linear(hidden_dim, quality_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(quality_dim, 1), nn.Sigmoid()
        )

        self.audio_quality = nn.Sequential(
            nn.Linear(hidden_dim, quality_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(quality_dim, 1), nn.Sigmoid()
        )

        self.hr_quality = nn.Sequential(
            nn.Linear(hidden_dim, quality_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(quality_dim, 1), nn.Sigmoid()
        )

    def forward(
        self,
        visual: torch.Tensor,
        audio: torch.Tensor,
        hr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Estimate quality scores for each modality.

        Args:
            visual: Visual features (batch_size, seq_len, hidden_dim)
            audio: Audio features (batch_size, seq_len, hidden_dim)
            hr: HR features (batch_size, seq_len, hidden_dim)

        Returns:
            visual_quality: Quality score for visual modality (batch_size, 1)
            audio_quality: Quality score for audio modality (batch_size, 1)
            hr_quality: Quality score for HR modality (batch_size, 1)
        """
        # Global pooling for each modality
        visual_pooled = torch.mean(visual, dim=1)  # (batch_size, hidden_dim)
        audio_pooled = torch.mean(audio, dim=1)  # (batch_size, hidden_dim)
        hr_pooled = torch.mean(hr, dim=1)  # (batch_size, hidden_dim)

        # Estimate quality for each modality
        visual_quality = self.visual_quality(visual_pooled)  # (batch_size, 1)
        audio_quality = self.audio_quality(audio_pooled)  # (batch_size, 1)
        hr_quality = self.hr_quality(hr_pooled)  # (batch_size, 1)

        return visual_quality, audio_quality, hr_quality


class GatingNetwork(nn.Module):
    """
    Calculates gating weights (alpha, beta, gamma) based on encoded context.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Linear layers to compute gating parameters from context
        self.gate_v = nn.Linear(hidden_dim, 1)
        self.gate_a = nn.Linear(hidden_dim, 1)
        self.gate_h = nn.Linear(hidden_dim, 1)

    def forward(self, context_v: torch.Tensor, context_a: torch.Tensor, context_h: torch.Tensor) -> tuple:
        """
        Forward pass for gating network.

        Args:
            context_v: Encoded visual context (batch, hidden_dim)
            context_a: Encoded audio context (batch, hidden_dim)
            context_h: Encoded HR context (batch, hidden_dim)

        Returns:
            Tuple of gating weights (alpha, beta, gamma) after softmax.
        """
        logger.debug(
            f"GatingNetwork - Input shapes: visual={context_v.shape if context_v is not None else None}, "
            f"audio={context_a.shape if context_a is not None else None}, "
            f"hr={context_h.shape if context_h is not None else None}"
        )

        # Compute raw gating scores
        score_v = (
            self.gate_v(context_v)
            if context_v is not None
            else torch.zeros(
                context_a.shape[0] if context_a is not None else context_h.shape[0],
                1,
                device=context_v.device
                if context_v is not None
                else (context_a.device if context_a is not None else context_h.device),
            ).fill_(-float("inf"))
        )  # Use -inf for missing
        score_a = (
            self.gate_a(context_a)
            if context_a is not None
            else torch.zeros(
                context_v.shape[0] if context_v is not None else context_h.shape[0],
                1,
                device=context_a.device
                if context_a is not None
                else (context_v.device if context_v is not None else context_h.device),
            ).fill_(-float("inf"))
        )
        score_h = (
            self.gate_h(context_h)
            if context_h is not None
            else torch.zeros(
                context_v.shape[0] if context_v is not None else context_a.shape[0],
                1,
                device=context_h.device
                if context_h is not None
                else (context_v.device if context_v is not None else context_a.device),
            ).fill_(-float("inf"))
        )

        # Ensure scores are on the same device
        device = score_v.device
        score_a = score_a.to(device)
        score_h = score_h.to(device)

        # Concatenate scores for softmax
        scores = torch.cat([score_v, score_a, score_h], dim=1)  # (batch, 3)

        # Apply softmax to get weights
        weights = F.softmax(scores, dim=1)  # (batch, 3)

        # Split weights
        alpha = weights[:, 0:1]  # (batch, 1)
        beta = weights[:, 1:2]  # (batch, 1)
        gamma = weights[:, 2:3]  # (batch, 1)

        return alpha, beta, gamma


class DynamicContextualGating(nn.Module):
    """
    Applies dynamic contextual gating to fuse multimodal features.
    """

    def __init__(self, hidden_dim: int, context_dim: int):
        super().__init__()
        self.context_encoder = ContextEncoder(hidden_dim, context_dim)
        self.gating_network = GatingNetwork(hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(
        self,
        visual_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        hr_feat: torch.Tensor,
        visual_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        hr_mask: torch.Tensor,
    ) -> tuple:
        """
        Forward pass for dynamic contextual gating.

        Args:
            visual_feat: Visual features (batch, seq_len, hidden_dim)
            audio_feat: Audio features (batch, seq_len, hidden_dim)
            hr_feat: HR features (batch, seq_len, hidden_dim)
            visual_mask: Mask for valid visual samples (batch,)
            audio_mask: Mask for valid audio samples (batch,)
            hr_mask: Mask for valid hr samples (batch,)

        Returns:
            Tuple containing:
                - fused_features: Fused multimodal features (batch, seq_len, hidden_dim)
                - gating_weights: Tuple of (alpha, beta, gamma) weights (batch, 1)
        """
        batch_size = -1
        seq_len = -1
        device = None

        # Determine batch_size, seq_len, and device from available inputs
        inputs = [(visual_feat, visual_mask), (audio_feat, audio_mask), (hr_feat, hr_mask)]
        for feat, mask in inputs:
            if feat is not None:
                batch_size = feat.shape[0]
                seq_len = feat.shape[1]
                device = feat.device
                break
        if batch_size == -1:  # Fallback if all features are None (shouldn't happen with masks)
            batch_size = (
                visual_mask.shape[0]
                if visual_mask is not None
                else (audio_mask.shape[0] if audio_mask is not None else hr_mask.shape[0])
            )
            seq_len = 1  # seq_len 1 if no features
            device = (
                visual_mask.device
                if visual_mask is not None
                else (audio_mask.device if audio_mask is not None else hr_mask.device)
            )

        logger.debug(
            f"DynamicContextualGating - Input shapes: visual={visual_feat.shape if visual_feat is not None else None}, "
            f"audio={audio_feat.shape if audio_feat is not None else None}, "
            f"hr={hr_feat.shape if hr_feat is not None else None}"
        )
        logger.debug(
            f"DynamicContextualGating - Masks: V={visual_mask.sum().item()}, A={audio_mask.sum().item()}, H={hr_mask.sum().item()}"
        )

        # Ensure all features have the correct batch size (they should from collate_fn)
        # Create zero tensors for missing modalities
        if visual_feat is None:
            visual_feat = torch.zeros(batch_size, seq_len, self.hidden_dim, device=device)
        if audio_feat is None:
            audio_feat = torch.zeros(batch_size, seq_len, self.hidden_dim, device=device)
        if hr_feat is None:
            hr_feat = torch.zeros(batch_size, seq_len, self.hidden_dim, device=device)

        # Encode context using masks
        encoded_contexts = self.context_encoder(visual_feat, audio_feat, hr_feat, visual_mask, audio_mask, hr_mask)
        context_v = encoded_contexts["visual"]
        context_a = encoded_contexts["audio"]
        context_h = encoded_contexts["hr"]

        # Calculate gating weights
        alpha, beta, gamma = self.gating_network(context_v, context_a, context_h)
        logger.debug(
            f"DynamicContextualGating - Raw weights: alpha={alpha.mean().item():.3f}, beta={beta.mean().item():.3f}, gamma={gamma.mean().item():.3f}"
        )

        # Expand weights for broadcasting: (batch, 1) -> (batch, 1, 1)
        alpha = alpha.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        gamma = gamma.unsqueeze(-1)
        logger.debug(f"DynamicContextualGating - Expanded weights: alpha={alpha.shape}, beta={beta.shape}, gamma={gamma.shape}")

        # Apply gating - Use masks here to zero out contributions from invalid samples
        # Masks are (batch,), need (batch, 1, 1) to multiply with (batch, seq_len, hidden_dim)
        v_mask_exp = visual_mask.unsqueeze(-1).unsqueeze(-1).float()
        a_mask_exp = audio_mask.unsqueeze(-1).unsqueeze(-1).float()
        h_mask_exp = hr_mask.unsqueeze(-1).unsqueeze(-1).float()

        # Weighted sum of features, masked
        fused = (alpha * visual_feat * v_mask_exp) + (beta * audio_feat * a_mask_exp) + (gamma * hr_feat * h_mask_exp)

        logger.debug(f"DynamicContextualGating - Fused shape after gating: {fused.shape}")

        # Return fused features and original (batch, 1) weights
        gating_weights = (alpha.squeeze(-1), beta.squeeze(-1), gamma.squeeze(-1))
        return fused, gating_weights
