"""
Dynamic Contextual Gating implementation for ACMFT.

This module implements the dynamic contextual gating mechanism that
adaptively weighs the importance of each modality (visual, audio, HR)
based on the current context and reliability of the modalities.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextEncoder(nn.Module):
    """
    Encodes the global context across all modalities to inform the gating decisions.
    """

    def __init__(self, hidden_dim: int, context_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.visual_encoder = nn.Sequential(nn.Linear(hidden_dim, context_dim), nn.ReLU(), nn.Dropout(dropout))

        self.audio_encoder = nn.Sequential(nn.Linear(hidden_dim, context_dim), nn.ReLU(), nn.Dropout(dropout))

        self.hr_encoder = nn.Sequential(nn.Linear(hidden_dim, context_dim), nn.ReLU(), nn.Dropout(dropout))

        self.global_context = nn.Sequential(nn.Linear(context_dim * 3, context_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(
        self,
        visual: torch.Tensor,
        audio: torch.Tensor,
        hr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the context encoder.

        Args:
            visual: Visual features (batch_size, seq_len, hidden_dim)
            audio: Audio features (batch_size, seq_len, hidden_dim)
            hr: HR features (batch_size, seq_len, hidden_dim)

        Returns:
            context: Global context tensor (batch_size, context_dim)
        """
        # Check for empty or problematic inputs
        if isinstance(visual, torch.Tensor) and visual.size(0) == 0:
            # Return empty context for empty batch
            context_dim = self.visual_encoder[0].out_features
            return torch.zeros(0, context_dim, device=visual.device)
            
        if isinstance(visual, list) or not isinstance(visual, torch.Tensor):
            print(f"ContextEncoder - WARNING: visual input is {type(visual)}, not a tensor!")
            raise ValueError(f"ContextEncoder requires tensor inputs, got {type(visual)} for visual")
            
        # Log input shapes for debugging
        print(f"ContextEncoder - Input shapes: visual={visual.shape}, audio={audio.shape}, hr={hr.shape}")
        
        # Global pooling for each modality to get a single vector
        visual_pooled = torch.mean(visual, dim=1)  # (batch_size, hidden_dim)
        audio_pooled = torch.mean(audio, dim=1)  # (batch_size, hidden_dim)
        hr_pooled = torch.mean(hr, dim=1)  # (batch_size, hidden_dim)
        
        print(f"ContextEncoder - Pooled shapes: visual={visual_pooled.shape}, audio={audio_pooled.shape}, hr={hr_pooled.shape}")

        # Encode each modality
        visual_encoded = self.visual_encoder(visual_pooled)  # (batch_size, context_dim)
        audio_encoded = self.audio_encoder(audio_pooled)  # (batch_size, context_dim)
        hr_encoded = self.hr_encoder(hr_pooled)  # (batch_size, context_dim)
        
        print(f"ContextEncoder - Encoded shapes: visual={visual_encoded.shape}, audio={audio_encoded.shape}, hr={hr_encoded.shape}")

        # Concatenate the encoded representations
        concat_context = torch.cat([visual_encoded, audio_encoded, hr_encoded], dim=1)  # (batch_size, context_dim * 3)
        print(f"ContextEncoder - Concat context shape: {concat_context.shape}")

        # Create global context
        global_context = self.global_context(concat_context)  # (batch_size, context_dim)
        print(f"ContextEncoder - Global context shape: {global_context.shape}")

        return global_context


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
    Dynamic Contextual Gating Network that determines the importance of each modality
    based on the global context and estimated quality of each modality.
    """

    def __init__(
        self,
        hidden_dim: int,
        context_dim: int = 256,
        gate_dim: int = 128,
        dropout: float = 0.1,
        use_quality_estimation: bool = True,
    ):
        super().__init__()
        self.use_quality_estimation = use_quality_estimation

        # Context encoder
        self.context_encoder = ContextEncoder(hidden_dim=hidden_dim, context_dim=context_dim, dropout=dropout)

        # Quality estimator (optional)
        if use_quality_estimation:
            self.quality_estimator = ModalityQualityEstimator(hidden_dim=hidden_dim, quality_dim=gate_dim // 2, dropout=dropout)

        # Gating network
        gate_input_dim = context_dim
        if use_quality_estimation:
            gate_input_dim += 3  # Add quality scores

        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, gate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_dim, 3),  # 3 modalities
        )

    def forward(
        self,
        visual: torch.Tensor,
        audio: torch.Tensor,
        hr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute dynamic weights for each modality.

        Args:
            visual: Visual features (batch_size, seq_len, hidden_dim)
            audio: Audio features (batch_size, seq_len, hidden_dim)
            hr: HR features (batch_size, seq_len, hidden_dim)

        Returns:
            alpha: Weight for visual modality (batch_size, 1)
            beta: Weight for audio modality (batch_size, 1)
            gamma: Weight for HR modality (batch_size, 1)
        """
        # Log input tensor shapes for debugging
        print(f"GatingNetwork - Input shapes: visual={visual.shape if isinstance(visual, torch.Tensor) else type(visual)}, "
              f"audio={audio.shape if isinstance(audio, torch.Tensor) else type(audio)}, "
              f"hr={hr.shape if isinstance(hr, torch.Tensor) else type(hr)}")
        
        # Check for empty batch
        if isinstance(visual, torch.Tensor) and visual.size(0) == 0:
            # Return default equal weights for empty batch
            device = visual.device
            zeros = torch.zeros(0, 1, device=device)
            print("GatingNetwork - Empty batch detected in visual, returning empty tensors")
            return zeros, zeros, zeros
            
        if isinstance(audio, torch.Tensor) and audio.size(0) == 0:
            # Return default equal weights for empty batch
            device = audio.device
            zeros = torch.zeros(0, 1, device=device)
            print("GatingNetwork - Empty batch detected in audio, returning empty tensors")
            return zeros, zeros, zeros
            
        if isinstance(hr, torch.Tensor) and hr.size(0) == 0:
            # Return default equal weights for empty batch
            device = hr.device
            zeros = torch.zeros(0, 1, device=device)
            print("GatingNetwork - Empty batch detected in hr, returning empty tensors")
            return zeros, zeros, zeros

        # Encode context
        context = self.context_encoder(visual, audio, hr)
        print(f"GatingNetwork - Context shape: {context.shape}")

        # Estimate quality scores (optional)
        if self.use_quality_estimation:
            visual_quality, audio_quality, hr_quality = self.quality_estimator(visual, audio, hr)
            print(f"GatingNetwork - Quality scores: visual={visual_quality.shape}, audio={audio_quality.shape}, hr={hr_quality.shape}")
            
            # Concatenate context with quality scores
            gate_input = torch.cat(
                [context, visual_quality.squeeze(-1), audio_quality.squeeze(-1), hr_quality.squeeze(-1)], dim=1
            )
        else:
            gate_input = context

        print(f"GatingNetwork - Gate input shape: {gate_input.shape}")

        # Compute unnormalized gating scores
        gate_scores = self.gate_network(gate_input)  # (batch_size, 3)
        print(f"GatingNetwork - Gate scores shape: {gate_scores.shape}, values: {gate_scores[:2] if gate_scores.size(0) > 1 else gate_scores}")

        # Apply softmax to get normalized weights (sum to 1)
        modality_weights = F.softmax(gate_scores, dim=1)  # (batch_size, 3)
        print(f"GatingNetwork - Modality weights after softmax: {modality_weights[:2] if modality_weights.size(0) > 1 else modality_weights}")

        # Extract individual modality weights
        alpha = modality_weights[:, 0].unsqueeze(1)  # Visual weight
        beta = modality_weights[:, 1].unsqueeze(1)  # Audio weight
        gamma = modality_weights[:, 2].unsqueeze(1)  # HR weight
        
        print(f"GatingNetwork - Final weights: alpha={alpha.shape}, beta={beta.shape}, gamma={gamma.shape}")

        return alpha, beta, gamma


class DynamicContextualGating(nn.Module):
    """
    Complete Dynamic Contextual Gating module that adaptively fuses multimodal features.
    """

    def __init__(
        self,
        hidden_dim: int,
        context_dim: int = 256,
        gate_dim: int = 128,
        dropout: float = 0.1,
        use_quality_estimation: bool = True,
        fusion_method: str = "weighted_sum",
    ):
        super().__init__()
        self.fusion_method = fusion_method

        # Gating network
        self.gating_network = GatingNetwork(
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            gate_dim=gate_dim,
            dropout=dropout,
            use_quality_estimation=use_quality_estimation,
        )

        # Additional projection for more complex fusion (optional)
        if fusion_method == "concat_projection":
            self.fusion_projection = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(
        self,
        visual: torch.Tensor,
        audio: torch.Tensor,
        hr: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Dynamically fuse multimodal features based on context.

        Args:
            visual: Visual features (batch_size, seq_len, hidden_dim)
            audio: Audio features (batch_size, seq_len, hidden_dim)
            hr: HR features (batch_size, seq_len, hidden_dim)

        Returns:
            fused: Fused multimodal representation (batch_size, seq_len, hidden_dim)
            weights: Tuple of (alpha, beta, gamma) weights for visualization
        """
        # Log input tensor shapes for debugging
        print(f"DynamicContextualGating - Input shapes: visual={visual.shape if isinstance(visual, torch.Tensor) else type(visual)}, "
              f"audio={audio.shape if isinstance(audio, torch.Tensor) else type(audio)}, "
              f"hr={hr.shape if isinstance(hr, torch.Tensor) else type(hr)}")
              
        # Check for list inputs and convert them to tensors if needed
        if isinstance(visual, list):
            if len(visual) == 0:
                print("DynamicContextualGating - WARNING: visual is an empty list!")
                # Create a default empty tensor
                device = audio.device if isinstance(audio, torch.Tensor) else (
                    hr.device if isinstance(hr, torch.Tensor) else torch.device('cpu'))
                visual = torch.zeros(0, 1, 256, device=device)  # Default empty shape
            else:
                try:
                    # Try to stack the tensors in the list
                    visual = torch.stack(visual)
                    print(f"DynamicContextualGating - Stacked visual list to tensor of shape {visual.shape}")
                except Exception as e:
                    print(f"DynamicContextualGating - ERROR: Could not stack visual list: {e}")
                    # Create a default empty tensor
                    device = audio.device if isinstance(audio, torch.Tensor) else (
                        hr.device if isinstance(hr, torch.Tensor) else torch.device('cpu'))
                    visual = torch.zeros(0, 1, 256, device=device)  # Default empty shape
        
        if isinstance(audio, list):
            if len(audio) == 0:
                print("DynamicContextualGating - WARNING: audio is an empty list!")
                # Create a default empty tensor
                device = visual.device if isinstance(visual, torch.Tensor) else (
                    hr.device if isinstance(hr, torch.Tensor) else torch.device('cpu'))
                audio = torch.zeros(0, 1, 256, device=device)  # Default empty shape
            else:
                try:
                    # Try to stack the tensors in the list
                    audio = torch.stack(audio)
                    print(f"DynamicContextualGating - Stacked audio list to tensor of shape {audio.shape}")
                except Exception as e:
                    print(f"DynamicContextualGating - ERROR: Could not stack audio list: {e}")
                    # Create a default empty tensor
                    device = visual.device if isinstance(visual, torch.Tensor) else (
                        hr.device if isinstance(hr, torch.Tensor) else torch.device('cpu'))
                    audio = torch.zeros(0, 1, 256, device=device)  # Default empty shape
        
        if isinstance(hr, list):
            if len(hr) == 0:
                print("DynamicContextualGating - WARNING: hr is an empty list!")
                # Create a default empty tensor
                device = visual.device if isinstance(visual, torch.Tensor) else (
                    audio.device if isinstance(audio, torch.Tensor) else torch.device('cpu'))
                hr = torch.zeros(0, 1, 256, device=device)  # Default empty shape
            else:
                try:
                    # Try to stack the tensors in the list
                    hr = torch.stack(hr)
                    print(f"DynamicContextualGating - Stacked hr list to tensor of shape {hr.shape}")
                except Exception as e:
                    print(f"DynamicContextualGating - ERROR: Could not stack hr list: {e}")
                    # Create a default empty tensor
                    device = visual.device if isinstance(visual, torch.Tensor) else (
                        audio.device if isinstance(audio, torch.Tensor) else torch.device('cpu'))
                    hr = torch.zeros(0, 1, 256, device=device)  # Default empty shape
        
        # Check that all inputs are tensors after potential conversion
        if not all(isinstance(x, torch.Tensor) for x in [visual, audio, hr]):
            print(f"DynamicContextualGating - ERROR: Not all inputs are tensors after conversion: "
                  f"visual={type(visual)}, audio={type(audio)}, hr={type(hr)}")
            # Create default empty tensors for any non-tensor inputs
            device = next((x.device for x in [visual, audio, hr] if isinstance(x, torch.Tensor)), torch.device('cpu'))
            if not isinstance(visual, torch.Tensor):
                visual = torch.zeros(0, 1, 256, device=device)
            if not isinstance(audio, torch.Tensor):
                audio = torch.zeros(0, 1, 256, device=device)
            if not isinstance(hr, torch.Tensor):
                hr = torch.zeros(0, 1, 256, device=device)
        
        # Ensure consistent batch sizes or use empty batches
        batch_sizes = [x.size(0) for x in [visual, audio, hr] if isinstance(x, torch.Tensor)]
        if len(set(batch_sizes)) > 1:
            print(f"DynamicContextualGating - WARNING: Inconsistent batch sizes: {batch_sizes}")
            # If any modality has batch_size=0, use empty batch for all
            if 0 in batch_sizes:
                print("DynamicContextualGating - Using empty batch for all modalities")
                device = next((x.device for x in [visual, audio, hr]), torch.device('cpu'))
                empty_shape = (0, visual.size(1), visual.size(2))
                visual = torch.zeros(empty_shape, device=device)
                audio = torch.zeros(empty_shape, device=device)
                hr = torch.zeros(empty_shape, device=device)
        
        # Now handle empty batch case consistently
        if visual.size(0) == 0 or audio.size(0) == 0 or hr.size(0) == 0:
            print("DynamicContextualGating - Handling empty batch")
            device = visual.device
            seq_len = max(visual.size(1), audio.size(1), hr.size(1))
            hidden_dim = max(visual.size(2), audio.size(2), hr.size(2))
            empty_tensor = torch.zeros(0, seq_len, hidden_dim, device=device)
            empty_weights = (torch.zeros(0, 1, device=device), 
                            torch.zeros(0, 1, device=device), 
                            torch.zeros(0, 1, device=device))
            return empty_tensor, empty_weights

        # Get dimensions from the first non-empty tensor
        batch_size, seq_len, hidden_dim = visual.shape
        print(f"DynamicContextualGating - Using shapes: batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}")

        # Compute modality weights
        try:
            alpha, beta, gamma = self.gating_network(visual, audio, hr)
            print(f"DynamicContextualGating - Got weights with shapes: alpha={alpha.shape}, beta={beta.shape}, gamma={gamma.shape}")
        except Exception as e:
            print(f"DynamicContextualGating - ERROR during gating: {e}")
            # Return default equal weights if gating fails
            alpha = torch.ones(batch_size, 1, device=visual.device) / 3
            beta = torch.ones(batch_size, 1, device=visual.device) / 3
            gamma = torch.ones(batch_size, 1, device=visual.device) / 3

        try:
            # Expand weights to match feature dimensions
            alpha_expanded = alpha.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, 1)
            beta_expanded = beta.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, 1)
            gamma_expanded = gamma.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, 1)
            
            print(f"DynamicContextualGating - Expanded weights: alpha={alpha_expanded.shape}, beta={beta_expanded.shape}, gamma={gamma_expanded.shape}")

            # Fusion methods
            if self.fusion_method == "weighted_sum":
                # Simple weighted sum
                fused = alpha_expanded * visual + beta_expanded * audio + gamma_expanded * hr
                print(f"DynamicContextualGating - Fused output shape: {fused.shape}")

            elif self.fusion_method == "concat_projection":
                # Concatenate weighted features and project
                weighted_visual = alpha_expanded * visual
                weighted_audio = beta_expanded * audio
                weighted_hr = gamma_expanded * hr

                concat_features = torch.cat([weighted_visual, weighted_audio, weighted_hr], dim=2)
                fused = self.fusion_projection(concat_features)
                print(f"DynamicContextualGating - Fused output shape: {fused.shape}")

            else:
                raise ValueError(f"Unsupported fusion method: {self.fusion_method}")

            # Return fused representation and weights (for visualization)
            return fused, (alpha[:, 0], beta[:, 0], gamma[:, 0])
        except Exception as e:
            print(f"DynamicContextualGating - ERROR during fusion: {e}")
            # Return a default fused representation with equal weights
            empty_fused = torch.zeros(batch_size, seq_len, hidden_dim, device=visual.device)
            empty_weights = (alpha, beta, gamma)
            return empty_fused, empty_weights
