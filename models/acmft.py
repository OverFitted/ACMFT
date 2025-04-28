"""
Main Adaptive Cross-Modal Fusion Transformer (ACMFT) model implementation.

This module implements the complete ACMFT architecture for emotion recognition,
integrating modality-specific encoders, cross-modal transformer blocks,
and the dynamic contextual gating mechanism.
"""

import logging
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig
from models.fusion.gating import DynamicContextualGating
from models.fusion.transformer import CrossModalTransformerBlock


class ModalityEmbedding(nn.Module):
    """
    Projects modality-specific features to a common embedding space.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project modality-specific features to common embedding space.

        Args:
            x: Input features (batch_size, seq_len, input_dim)

        Returns:
            projected: Projected features (batch_size, seq_len, hidden_dim)
        """
        # Handle empty tensor case
        if x.size(0) == 0:
            # Return empty tensor with correct output dimensions
            return torch.zeros((0, x.size(1), self.projection[0].out_features), device=x.device)

        return self.projection(x)


class ACMFT(nn.Module):
    """
    Adaptive Cross-Modal Fusion Transformer (ACMFT) for emotion recognition.

    This model integrates visual, audio, and physiological (HR) modalities
    using cross-modal transformer blocks and dynamic contextual gating.
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        visual_encoder: Optional[nn.Module] = None,
        audio_encoder: Optional[nn.Module] = None,
        hr_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        # Use default config if not provided
        self.config = config or ModelConfig()

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Store modality encoders
        self.visual_encoder = visual_encoder
        self.audio_encoder = audio_encoder
        self.hr_encoder = hr_encoder

        # Create modality embedding layers
        self.visual_embedding = ModalityEmbedding(self.config.visual_dim, self.config.hidden_dim, self.config.dropout)
        self.audio_embedding = ModalityEmbedding(self.config.audio_dim, self.config.hidden_dim, self.config.dropout)
        self.hr_embedding = ModalityEmbedding(self.config.hr_dim, self.config.hidden_dim, self.config.dropout)

        # Create cross-modal transformer blocks
        self.cross_modal_layers = nn.ModuleList(
            [
                CrossModalTransformerBlock(
                    d_model=self.config.hidden_dim,
                    nhead=self.config.num_heads,
                    dim_feedforward=self.config.hidden_dim * 4,
                    dropout=self.config.dropout,
                    activation=self.config.activation,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        # Create dynamic contextual gating
        self.contextual_gating = DynamicContextualGating(
            hidden_dim=self.config.hidden_dim,
            context_dim=self.config.hidden_dim,
        )

        # Create emotion classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.num_emotions),
        )

    def _prepare_modality(
        self,
        visual: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        hr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare modality inputs by applying modality-specific encoders if provided.

        Args:
            visual: Visual input (can be raw images/videos or pre-extracted features)
            audio: Audio input (can be raw audio or pre-extracted features)
            hr: HR input (can be raw signals or pre-extracted features)

        Returns:
            visual_features: Visual features
            audio_features: Audio features
            hr_features: HR features
        """
        # Process visual input if encoder is provided
        if visual is not None and self.visual_encoder is not None:
            visual_features = self.visual_encoder(visual)
        else:
            visual_features = visual

        # Process audio input if encoder is provided
        if audio is not None and self.audio_encoder is not None:
            audio_features = self.audio_encoder(audio)
        else:
            audio_features = audio

        # Process HR input if encoder is provided
        if hr is not None and self.hr_encoder is not None:
            hr_features = self.hr_encoder(hr)
        else:
            hr_features = hr

        return visual_features, audio_features, hr_features

    def forward(
        self,
        visual: Optional[torch.Tensor],
        audio: Optional[torch.Tensor],
        hr: Optional[torch.Tensor],
        visual_mask: Optional[torch.Tensor] = None, # Added mask arguments
        audio_mask: Optional[torch.Tensor] = None,
        hr_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass for the ACMFT model.

        Args:
            visual: Visual input tensor (batch_size, ...)
            audio: Audio input tensor (batch_size, sequence_length)
            hr: Heart rate input tensor (batch_size, ...)
            visual_mask: Boolean mask for visual modality (batch_size,)
            audio_mask: Boolean mask for audio modality (batch_size,)
            hr_mask: Boolean mask for HR modality (batch_size,)
            return_weights: Whether to return modality weights

        Returns:
            logits: Classification logits
            weights: Optional dictionary of modality weights if return_weights=True
        """
        # Determine batch size and device from the first available modality
        batch_size = 0
        device = self.device
        if visual is not None and visual.size(0) > 0:
            batch_size = visual.size(0)
            device = visual.device
        elif audio is not None and audio.size(0) > 0:
            batch_size = audio.size(0)
            device = audio.device
        elif hr is not None and hr.size(0) > 0:
            batch_size = hr.size(0)
            device = hr.device

        # logging.debug(f"ACMFT forward: batch_size={batch_size}, device={device}")
        # if visual is not None:
        #     logging.debug(f"  Visual input shape: {visual.shape}")
        # if audio is not None:
        #     logging.debug(f"  Audio input shape: {audio.shape}")
        # if hr is not None:
        #     logging.debug(f"  HR input shape: {hr.shape}")

        # If batch size is 0 (e.g., empty batch), return empty outputs
        if batch_size == 0:
            logging.error("ACMFT: Detected empty batch, returning empty output.")
            empty_logits = torch.zeros((0, self.config.num_emotions), device=device)
            if return_weights:
                empty_weights = {
                    "visual_weight": torch.zeros(0, device=device),
                    "audio_weight": torch.zeros(0, device=device),
                    "hr_weight": torch.zeros(0, device=device),
                }
                return empty_logits, empty_weights
            else:
                return empty_logits

        # Prepare modality inputs using encoders (if available)
        # _prepare_modality handles None inputs internally
        visual_features, audio_features, hr_features = self._prepare_modality(visual, audio, hr)

        # --- Feature processing logic (assuming _prepare_modality returns features or None) ---
        seq_len = 1  # Default sequence length after encoders (can be adapted)

        # Handle cases where encoders might return None or features need default values
        # Also create default masks if they are None
        if visual_features is None:
            visual_features = torch.zeros(batch_size, seq_len, self.config.visual_dim, device=device)
            visual_mask = torch.zeros(batch_size, dtype=torch.bool, device=device) # Default mask
            logging.warning("  Visual features are None, using zeros.")
        elif visual_features.ndim == 2:  # Add sequence dimension if missing
            visual_features = visual_features.unsqueeze(1)
        if visual_mask is None: # Create default mask if features exist but mask is None
            visual_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        if audio_features is None:
            audio_features = torch.zeros(batch_size, seq_len, self.config.audio_dim, device=device)
            audio_mask = torch.zeros(batch_size, dtype=torch.bool, device=device) # Default mask
            logging.warning("  Audio features are None, using zeros.")
        elif audio_features.ndim == 2:
            audio_features = audio_features.unsqueeze(1)
        if audio_mask is None:
            audio_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        if hr_features is None:
            hr_features = torch.zeros(batch_size, seq_len, self.config.hr_dim, device=device)
            hr_mask = torch.zeros(batch_size, dtype=torch.bool, device=device) # Default mask
            # logging.warning("  HR features are None, using zeros.")
        elif hr_features.ndim == 2:
            hr_features = hr_features.unsqueeze(1)
        if hr_mask is None:
            hr_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        # logging.debug(f"  Visual features shape after prep: {visual_features.shape}")
        # logging.debug(f"  Audio features shape after prep: {audio_features.shape}")
        # logging.debug(f"  HR features shape after prep: {hr_features.shape}")

        # Project each modality to common embedding space
        try:
            visual_emb = self.visual_embedding(visual_features)
            audio_emb = self.audio_embedding(audio_features)
            hr_emb = self.hr_embedding(hr_features)
        except Exception as e:
            logging.error(f"Error during modality embedding: {e}")
            raise e

        # logging.debug(f"  Visual embedding shape: {visual_emb.shape}")
        # logging.debug(f"  Audio embedding shape: {audio_emb.shape}")
        # logging.debug(f"  HR embedding shape: {hr_emb.shape}")

        # Process through cross-modal transformer blocks
        for i, layer in enumerate(self.cross_modal_layers):
            try:
                # Pass masks to cross-modal layers if they accept them (optional)
                # Assuming CrossModalTransformerBlock.forward accepts masks as keyword args
                visual_emb, audio_emb, hr_emb = layer(
                    visual_emb, audio_emb, hr_emb,
                    visual_mask=visual_mask, audio_mask=audio_mask, hr_mask=hr_mask
                )
                # logging.debug(f"  After CrossModal Layer {i}: V={visual_emb.shape}, A={audio_emb.shape}, H={hr_emb.shape}")
            except TypeError as te:
                # Fallback if layer doesn't accept masks
                if "mask" in str(te):
                    visual_emb, audio_emb, hr_emb = layer(visual_emb, audio_emb, hr_emb)
                    # logging.debug(f"  After CrossModal Layer {i} (no masks): V={visual_emb.shape}, A={audio_emb.shape}, H={hr_emb.shape}")
                else:
                    logging.error(f"Error in CrossModal Layer {i}: {te}")
                    raise te
            except Exception as e:
                logging.error(f"Error in CrossModal Layer {i}: {e}")
                raise e

        # Apply dynamic contextual gating - Pass masks here
        try:
            fused, (alpha, beta, gamma) = self.contextual_gating(
                visual_emb, audio_emb, hr_emb,
                visual_mask=visual_mask, audio_mask=audio_mask, hr_mask=hr_mask
            )
            # logging.debug(f"  Fused shape after gating: {fused.shape}")
            # logging.debug(f"  Gating weights: alpha={alpha.shape}, beta={beta.shape}, gamma={gamma.shape}")
        except Exception as e:
            logging.error(f"Error during contextual gating: {e}")
            raise e

        # Global pooling (if sequence length > 1)
        if fused.size(1) > 1:
            fused = torch.mean(fused, dim=1)
            # logging.debug(f"  Fused shape after pooling: {fused.shape}")
        else:
            fused = fused.squeeze(1)
            # logging.debug(f"  Fused shape after squeeze: {fused.shape}")

        # Emotion classification
        try:
            emotion_logits = self.classifier(fused)
            # logging.debug(f"  Logits shape: {emotion_logits.shape}")
        except Exception as e:
            logging.error(f"Error during classification: {e}")
            raise e

        # Return logits and optionally modality weights
        if return_weights:
            weights = {
                "visual_weight": alpha,
                "audio_weight": beta,
                "hr_weight": gamma,
            }
            return emotion_logits, weights
        else:
            return emotion_logits

    def predict(
        self,
        visual: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        hr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict emotions with probabilities.

        Args:
            visual: Visual input (batch_size, ...) - raw or pre-extracted features
            audio: Audio input (batch_size, ...) - raw or pre-extracted features
            hr: HR input (batch_size, ...) - raw or pre-extracted features

        Returns:
            emotion_probs: Emotion probabilities (batch_size, num_emotions)
            predicted_emotions: Predicted emotion indices (batch_size,)
        """
        # Get logits
        logits = self.forward(visual, audio, hr)

        # Convert to probabilities
        emotion_probs = F.softmax(logits, dim=1)

        # Get predicted class
        predicted_emotions = torch.argmax(emotion_probs, dim=1)

        return emotion_probs, predicted_emotions
