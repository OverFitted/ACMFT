"""
Audio encoder implementation for ACMFT.

This module implements the audio modality encoder that processes voice/audio data
using wav2vec2 for feature extraction.
"""

from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


class AudioPreprocessor(nn.Module):
    """
    Preprocesses audio input for feature extraction.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        target_length: Optional[int] = None,
        normalize: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.normalize = normalize

    def forward(
        self,
        audio: Union[torch.Tensor, np.ndarray, List[np.ndarray], List[torch.Tensor]],
        sample_rates: Optional[Union[int, List[int]]] = None,
    ) -> torch.Tensor:
        """
        Preprocess audio for feature extraction.

        Args:
            audio: Input audio waveforms. Can be:
                - torch.Tensor: (batch_size, time)
                - np.ndarray: (batch_size, time)
                - List[np.ndarray/torch.Tensor]: List of audio waveforms
            sample_rates: Sample rates of input audio. If not provided,
                          audio is already at self.sample_rate.

        Returns:
            processed_audio: Preprocessed audio (batch_size, time)
        """
        # Handle empty inputs (None or empty list)
        if audio is None or (isinstance(audio, list) and len(audio) == 0):
            # Return empty tensor with correct shape for wav2vec2
            return torch.zeros((0, 16000))  # 1 second of silence at 16kHz

        # Handle empty tensor
        if isinstance(audio, torch.Tensor) and audio.size(0) == 0:
            return torch.zeros((0, 16000))  # 1 second of silence at 16kHz

        # Convert to torch tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        elif isinstance(audio, list):
            # Empty list check (additional safeguard)
            if len(audio) == 0:
                return torch.zeros((0, 16000))

            # Convert list of arrays to tensor
            if isinstance(audio[0], np.ndarray):
                audio = [torch.from_numpy(a).float() for a in audio]

            # Pad to same length
            max_length = max(a.size(-1) for a in audio)
            padded_audio = []
            for a in audio:
                padding = max_length - a.size(-1)
                padded = F.pad(a, (0, padding))
                padded_audio.append(padded)

            audio = torch.stack(padded_audio)

        # Ensure audio is 2D: (batch_size, time)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Resample if needed
        if sample_rates is not None:
            resampled_audio = []

            if not isinstance(sample_rates, list):
                sample_rates = [sample_rates] * audio.size(0)

            for i, (waveform, sr) in enumerate(zip(audio, sample_rates)):
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                    resampled = resampler(waveform)
                    resampled_audio.append(resampled)
                else:
                    resampled_audio.append(waveform)

            audio = torch.stack(resampled_audio)

        # Adjust length if target_length is specified
        if self.target_length is not None:
            current_length = audio.size(-1)

            if current_length < self.target_length:
                # Pad if too short
                padding = self.target_length - current_length
                audio = F.pad(audio, (0, padding))
            elif current_length > self.target_length:
                # Crop if too long
                audio = audio[..., : self.target_length]

        # Normalize to [-1, 1] if requested
        if self.normalize:
            # Normalize per audio sample
            for i in range(audio.size(0)):
                if torch.max(torch.abs(audio[i])) > 0:
                    audio[i] = audio[i] / torch.max(torch.abs(audio[i]))

        return audio


class Wav2Vec2FeatureExtractorWrapper(nn.Module):
    """
    Feature extractor using wav2vec2 for audio representation.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        output_dim: int = 768,
        use_custom_dim: bool = False,
        feature_aggregation: str = "mean",  # Options: "mean", "max", "attention"
        freeze_feature_encoder: bool = True,
        device: str = "cpu",  # Added device argument
    ):
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim
        self.feature_aggregation = feature_aggregation
        self.device = device  # Store device

        # Load wav2vec2 model and feature extractor
        # Move the model to the specified device immediately after loading
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        # The feature_processor is just for configuration/tokenization, doesn't need device usually
        self.feature_processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

        # Freeze feature encoder if requested
        if freeze_feature_encoder:
            # Freeze the CNN feature extractor part of the Wav2Vec2Model
            for param in self.wav2vec2.feature_extractor.parameters():
                param.requires_grad = False

        # Get wav2vec2 output dimension
        wav2vec2_dim = self.wav2vec2.config.hidden_size

        # Add projection layer if needed, ensure it's on the correct device
        if use_custom_dim and output_dim != wav2vec2_dim:
            self.projection = nn.Linear(wav2vec2_dim, output_dim).to(self.device)
        else:
            # Ensure Identity is registered but doesn't need device move
            self.projection = nn.Identity()

        # Attention-based aggregation if needed, ensure it's on the correct device
        if feature_aggregation == "attention":
            self.attention = nn.Sequential(nn.Linear(wav2vec2_dim, 128), nn.Tanh(), nn.Linear(128, 1)).to(self.device)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract features from audio waveforms.

        Args:
            audio: Input audio waveforms (batch_size, time) expected on self.device

        Returns:
            features: Extracted features (batch_size, output_dim)
        """
        # Ensure input is on the correct device
        audio = audio.to(self.device)

        # Check for empty tensor - handle case with no audio
        if audio.size(0) == 0:
            # Return an empty tensor with the right feature dimension
            return torch.zeros((0, self.output_dim), device=self.device)

        # Process with wav2vec2
        # Determine if gradients are needed based on requires_grad status of feature_extractor
        # Check the first conv layer's weight
        is_frozen = not self.wav2vec2.feature_extractor.conv_layers[0].conv.weight.requires_grad
        grad_context = torch.no_grad() if is_frozen else torch.enable_grad()
        with grad_context:
            # Ensure the model is in the correct mode (train/eval)
            outputs = self.wav2vec2(audio, output_hidden_states=True)

        # Get hidden states from the last layer
        last_hidden = outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size)

        # Aggregate features based on specified method
        if self.feature_aggregation == "mean":
            # Mean pooling
            pooled_features = torch.mean(last_hidden, dim=1)  # (batch_size, hidden_size)

        elif self.feature_aggregation == "max":
            # Max pooling
            pooled_features, _ = torch.max(last_hidden, dim=1)  # (batch_size, hidden_size)

        elif self.feature_aggregation == "attention":
            # Attention-based pooling
            attention_weights = self.attention(last_hidden)  # (batch_size, sequence_length, 1)
            attention_weights = F.softmax(attention_weights, dim=1)
            pooled_features = torch.sum(last_hidden * attention_weights, dim=1)  # (batch_size, hidden_size)

        else:
            raise ValueError(f"Unsupported feature aggregation method: {self.feature_aggregation}")

        # Project to desired output dimension if needed
        features = self.projection(pooled_features)

        return features


class AudioEncoder(nn.Module):
    """
    Complete audio encoder pipeline for ACMFT including preprocessing
    and feature extraction using wav2vec2.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        sample_rate: int = 16000,
        target_length: Optional[int] = None,
        output_dim: int = 768,
        feature_aggregation: str = "mean",
        freeze_feature_encoder: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)  # Use torch.device

        # Initialize preprocessor - typically runs on CPU but ensure it handles tensors from any device
        self.preprocessor = AudioPreprocessor(sample_rate=sample_rate, target_length=target_length, normalize=True)
        # No parameters, so .to(device) is not strictly needed but doesn't hurt
        # Keep preprocessor on CPU if it's faster, move data later
        # self.preprocessor.to(self.device)

        # Initialize feature extractor and pass the device
        self.feature_extractor = Wav2Vec2FeatureExtractorWrapper(
            model_name=model_name,
            output_dim=output_dim,
            use_custom_dim=(output_dim != 768),  # 768 is the base wav2vec dim
            feature_aggregation=feature_aggregation,
            freeze_feature_encoder=freeze_feature_encoder,
            device=str(self.device),  # Pass device string to wrapper
        )
        # The wrapper handles moving its internal model to the device

    def forward(
        self,
        audio: Union[torch.Tensor, np.ndarray, List[np.ndarray], List[torch.Tensor]],
        sample_rates: Optional[Union[int, List[int]]] = None,
    ) -> torch.Tensor:
        """
        Process audio through the complete pipeline.

        Args:
            audio: Input audio waveforms (batch_size, time) - expected on CPU from dataloader
            sample_rates: Sample rates of input audio

        Returns:
            features: Audio features (batch_size, output_dim) on self.device
        """
        # Preprocess audio - preprocessor handles tensor conversion/padding
        # Keep processed_audio on CPU for now if preprocessor runs faster there
        processed_audio = self.preprocessor(audio, sample_rates)
        # Move the final processed audio tensor to the target device right before feature extraction
        processed_audio = processed_audio.to(self.device)

        # Extract features - feature_extractor expects input on its device
        features = self.feature_extractor(processed_audio)

        return features


# Alias for backward compatibility
VoiceEncoder = AudioEncoder
