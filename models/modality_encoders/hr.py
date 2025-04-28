"""
Heart Rate (HR) signal encoder implementation for ACMFT.

This module implements the physiological modality encoder that processes
heart rate signals using a 1D CNN architecture for feature extraction.
"""

from typing import List, Optional, Union

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F


class HRPreprocessor(nn.Module):
    """
    Preprocesses heart rate (HR) signals for feature extraction.
    """

    def __init__(
        self,
        sample_rate: int = 256,
        target_length: Optional[int] = None,
        normalize: bool = True,
        filter_signals: bool = True,
        lowcut: float = 0.5,
        highcut: float = 4.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.normalize = normalize
        self.filter_signals = filter_signals
        self.lowcut = lowcut
        self.highcut = highcut

    def _butter_bandpass(self, lowcut: float, highcut: float, fs: float, order: int = 4):
        """
        Create a Butterworth bandpass filter design.

        Args:
            lowcut: Low cutoff frequency
            highcut: High cutoff frequency
            fs: Sampling frequency
            order: Filter order

        Returns:
            b, a: Filter coefficients
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = scipy.signal.butter(order, [low, high], btype="band")
        return b, a

    def _bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter to the data.

        Args:
            data: Input signal data

        Returns:
            filtered_data: Filtered signal data
        """
        b, a = self._butter_bandpass(self.lowcut, self.highcut, self.sample_rate)
        return scipy.signal.filtfilt(b, a, data)

    def forward(
        self,
        signals: Union[torch.Tensor, np.ndarray, List[np.ndarray], List[torch.Tensor]],
        sample_rates: Optional[Union[int, List[int]]] = None,
    ) -> torch.Tensor:
        """
        Preprocess HR signals for feature extraction.

        Args:
            signals: Input HR signals. Can be:
                - torch.Tensor: (batch_size, time)
                - np.ndarray: (batch_size, time)
                - List[np.ndarray/torch.Tensor]: List of HR signals
            sample_rates: Sample rates of input signals. If not provided,
                          assumes signals are already at self.sample_rate.

        Returns:
            processed_signals: Preprocessed HR signals (batch_size, time)
        """
        # Handle empty or None inputs
        if signals is None or (isinstance(signals, list) and len(signals) == 0):
            # Return an empty tensor with correct dimensionality for HR signals
            return torch.zeros((0, 256 if self.target_length is None else self.target_length))
            
        # Handle empty tensor
        if isinstance(signals, torch.Tensor) and signals.size(0) == 0:
            return torch.zeros((0, 256 if self.target_length is None else self.target_length))
            
        # Convert to torch tensor if needed
        if isinstance(signals, np.ndarray):
            signals = torch.from_numpy(signals).float()
        elif isinstance(signals, list):
            # Handle empty list check (additional safeguard)
            if len(signals) == 0:
                return torch.zeros((0, 256 if self.target_length is None else self.target_length))
                
            # Convert list of arrays to tensor
            if isinstance(signals[0], np.ndarray):
                signals = [torch.from_numpy(s).float() for s in signals]

            # Pad to same length
            max_length = max(s.size(-1) for s in signals)
            padded_signals = []
            for s in signals:
                padding = max_length - s.size(-1)
                padded = F.pad(s, (0, padding))
                padded_signals.append(padded)

            signals = torch.stack(padded_signals)

        # Ensure signals is 2D: (batch_size, time)
        if signals.dim() == 1:
            signals = signals.unsqueeze(0)

        # Apply bandpass filter if requested
        if self.filter_signals:
            filtered_signals = []

            # Check for empty batch before filtering
            if signals.size(0) == 0:
                return torch.zeros((0, 256 if self.target_length is None else self.target_length))
                
            # Process each signal in the batch
            for i in range(signals.size(0)):
                signal_np = signals[i].cpu().numpy()
                filtered_np = self._bandpass_filter(signal_np)
                filtered_tensor = torch.from_numpy(filtered_np).float()
                filtered_signals.append(filtered_tensor)

            # Only stack if we have signals to stack
            if filtered_signals:
                signals = torch.stack(filtered_signals)
            else:
                return torch.zeros((0, 256 if self.target_length is None else self.target_length))

        # Resample if needed
        if sample_rates is not None and signals.size(0) > 0:
            resampled_signals = []

            if not isinstance(sample_rates, list):
                sample_rates = [sample_rates] * signals.size(0)

            for i, (signal, sr) in enumerate(zip(signals, sample_rates)):
                if sr != self.sample_rate:
                    # Since torchaudio.transforms.Resample requires 1D input,
                    # we need to handle each signal separately
                    resampled_length = int(signal.size(-1) * (self.sample_rate / sr))
                    resampled = (
                        F.interpolate(
                            signal.unsqueeze(0).unsqueeze(0), size=resampled_length, mode="linear", align_corners=False
                        )
                        .squeeze(0)
                        .squeeze(0)
                    )
                    resampled_signals.append(resampled)
                else:
                    resampled_signals.append(signal)

            # Only stack if we have signals to stack
            if resampled_signals:
                signals = torch.stack(resampled_signals)
            else:
                return torch.zeros((0, 256 if self.target_length is None else self.target_length))

        # Adjust length if target_length is specified
        if self.target_length is not None and signals.size(0) > 0:
            current_length = signals.size(-1)

            if current_length < self.target_length:
                # Pad if too short
                padding = self.target_length - current_length
                signals = F.pad(signals, (0, padding))
            elif current_length > self.target_length:
                # Crop if too long
                signals = signals[..., : self.target_length]

        # Normalize if requested
        if self.normalize and signals.size(0) > 0:
            # Normalize each signal in the batch
            for i in range(signals.size(0)):
                if torch.std(signals[i]) > 0:
                    # Z-score normalization
                    signals[i] = (signals[i] - torch.mean(signals[i])) / torch.std(signals[i])

        return signals


class CNNHRFeatureExtractor(nn.Module):
    """
    Feature extractor using 1D CNN for heart rate (HR) signals.
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_dim: int = 128,
        kernel_sizes: List[int] = [3, 5, 7],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim

        # Multiple parallel convolutional branches with different kernel sizes
        # to capture different temporal patterns
        self.conv_branches = nn.ModuleList()

        for kernel_size in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(input_channels, 32, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Conv1d(32, 64, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Conv1d(64, 128, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
            )
            self.conv_branches.append(branch)

        # Calculate the output size of the convolutional branches
        # for a fixed input size (will be updated in forward pass)
        branch_output_size = 128 * len(kernel_sizes)

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(branch_output_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from HR signals.

        Args:
            x: Input HR signals (batch_size, time)

        Returns:
            features: Extracted features (batch_size, output_dim)
        """
        batch_size = x.size(0)
        
        # Handle empty input
        if batch_size == 0:
            # Return empty tensor with the correct feature dimension
            return torch.zeros((0, self.output_dim), device=x.device)

        # Add channel dimension if not present
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, time)

        # Process through each convolutional branch
        branch_outputs = []
        for branch in self.conv_branches:
            branch_output = branch(x)

            # Global pooling to handle variable length inputs
            pooled = F.adaptive_avg_pool1d(branch_output, 1).view(batch_size, -1)
            branch_outputs.append(pooled)

        # Concatenate branch outputs
        concat_features = torch.cat(branch_outputs, dim=1)

        # Apply fully connected layers
        features = self.fc_layers(concat_features)

        return features


class HRVFeatureExtractor(nn.Module):
    """
    Feature extractor for heart rate variability (HRV) metrics.
    This extractor computes common HRV features from RR intervals.
    """

    def __init__(
        self,
        output_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim

        # Linear layers to project HRV features to output dimension
        self.projection = nn.Sequential(
            nn.Linear(6, 16),  # 6 HRV features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, output_dim),
        )

    def _compute_hrv_features(self, rr_intervals: torch.Tensor) -> torch.Tensor:
        """
        Compute HRV features from RR intervals.

        Args:
            rr_intervals: RR intervals in seconds (batch_size, num_intervals)

        Returns:
            hrv_features: HRV features (batch_size, 6)
        """
        batch_size = rr_intervals.size(0)
        hrv_features = torch.zeros(batch_size, 6, device=rr_intervals.device)

        for i in range(batch_size):
            rr = rr_intervals[i]
            valid_rr = rr[rr > 0]  # Filter out zero or padding values

            if len(valid_rr) > 1:
                # 1. Mean RR
                mean_rr = torch.mean(valid_rr)
                hrv_features[i, 0] = mean_rr

                # 2. SDNN - Standard deviation of NN intervals
                sdnn = torch.std(valid_rr)
                hrv_features[i, 1] = sdnn

                # 3. RMSSD - Root mean square of successive differences
                rmssd = torch.sqrt(torch.mean(torch.diff(valid_rr) ** 2))
                hrv_features[i, 2] = rmssd

                # 4. pNN50 - Percentage of successive RR intervals differing by more than 50ms
                nn50 = torch.sum(torch.abs(torch.diff(valid_rr)) > 0.05)
                pnn50 = nn50 / (len(valid_rr) - 1) if len(valid_rr) > 1 else torch.tensor(0.0)
                hrv_features[i, 3] = pnn50

                # 5. HR - Heart rate
                hr = 60 / mean_rr
                hrv_features[i, 4] = hr

                # 6. CV - Coefficient of variation
                cv = sdnn / mean_rr
                hrv_features[i, 5] = cv

        # Normalize features
        for j in range(6):
            if torch.max(hrv_features[:, j]) > 0:
                hrv_features[:, j] = (hrv_features[:, j] - torch.mean(hrv_features[:, j])) / (
                    torch.std(hrv_features[:, j]) + 1e-8
                )

        return hrv_features

    def forward(self, rr_intervals: torch.Tensor) -> torch.Tensor:
        """
        Extract HRV features from RR intervals.

        Args:
            rr_intervals: RR intervals in seconds (batch_size, num_intervals)

        Returns:
            features: Extracted features (batch_size, output_dim)
        """
        # Compute HRV features
        hrv_features = self._compute_hrv_features(rr_intervals)

        # Project to output dimension
        features = self.projection(hrv_features)

        return features


class CNNHREncoder(nn.Module):
    """
    Complete HR encoder pipeline for ACMFT including preprocessing
    and feature extraction using 1D CNN.
    """

    def __init__(
        self,
        sample_rate: int = 256,
        target_length: Optional[int] = 2560,
        output_dim: int = 128,
        use_hrv_features: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.use_hrv_features = use_hrv_features

        # Initialize preprocessor
        self.preprocessor = HRPreprocessor(
            sample_rate=sample_rate,
            target_length=target_length,
            normalize=True,
            filter_signals=True,
        )

        # Initialize CNN feature extractor
        self.cnn_extractor = CNNHRFeatureExtractor(
            input_channels=1,
            output_dim=output_dim if not use_hrv_features else output_dim - 32,
            kernel_sizes=[3, 5, 7],
            dropout=0.1,
        )

        # Initialize HRV feature extractor (optional)
        if use_hrv_features:
            self.hrv_extractor = HRVFeatureExtractor(
                output_dim=32,
                dropout=0.1,
            )

    def forward(
        self,
        signals: Union[torch.Tensor, np.ndarray, List[np.ndarray], List[torch.Tensor]],
        rr_intervals: Optional[torch.Tensor] = None,
        sample_rates: Optional[Union[int, List[int]]] = None,
    ) -> torch.Tensor:
        """
        Process HR signals through the complete pipeline.

        Args:
            signals: Input HR signals
            rr_intervals: RR intervals in seconds (optional, for HRV features)
            sample_rates: Sample rates of input signals

        Returns:
            features: HR features (batch_size, output_dim)
        """
        # Preprocess signals
        processed_signals = self.preprocessor(signals, sample_rates)
        processed_signals = processed_signals.to(self.device)

        # Extract CNN features
        cnn_features = self.cnn_extractor(processed_signals)

        # Extract HRV features if enabled and RR intervals provided
        if self.use_hrv_features and rr_intervals is not None:
            rr_intervals = rr_intervals.to(self.device)
            hrv_features = self.hrv_extractor(rr_intervals)

            # Concatenate CNN and HRV features
            features = torch.cat([cnn_features, hrv_features], dim=1)
        else:
            features = cnn_features

        return features


class LSTMHRFeatureExtractor(nn.Module):
    """
    Feature extractor using LSTM for heart rate (HR) signals.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_dim: int = 128,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM layers for sequential processing
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Calculate LSTM output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        # Attention mechanism for weighted temporal aggregation
        self.attention = nn.Sequential(nn.Linear(lstm_output_size, 64), nn.Tanh(), nn.Linear(64, 1))

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_output_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from HR signals using LSTM.

        Args:
            x: Input HR signals (batch_size, time)

        Returns:
            features: Extracted features (batch_size, output_dim)
        """
        # Reshape input for LSTM: (batch_size, time, channels)
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch_size, time, 1)
        elif x.dim() == 3:
            x = x.transpose(1, 2)  # (batch_size, time, channels)

        # Process through LSTM
        lstm_out, _ = self.lstm(x)  # (batch_size, time, hidden_size*num_directions)

        # Apply attention mechanism
        attn_scores = self.attention(lstm_out)  # (batch_size, time, 1)
        attn_weights = F.softmax(attn_scores, dim=1)

        # Weighted sum of LSTM outputs
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch_size, hidden_size*num_directions)

        # Apply fully connected layers
        features = self.fc_layers(context)

        return features


class LSTMHREncoder(nn.Module):
    """
    Complete HR encoder pipeline for ACMFT using LSTM for feature extraction.
    """

    def __init__(
        self,
        sample_rate: int = 256,
        target_length: Optional[int] = 2560,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_dim: int = 128,
        bidirectional: bool = True,
        use_hrv_features: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.use_hrv_features = use_hrv_features

        # Initialize preprocessor
        self.preprocessor = HRPreprocessor(
            sample_rate=sample_rate,
            target_length=target_length,
            normalize=True,
            filter_signals=True,
        )

        # Initialize LSTM feature extractor
        self.lstm_extractor = LSTMHRFeatureExtractor(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_dim=output_dim if not use_hrv_features else output_dim - 32,
            bidirectional=bidirectional,
            dropout=0.1,
        )

        # Initialize HRV feature extractor (optional)
        if use_hrv_features:
            self.hrv_extractor = HRVFeatureExtractor(
                output_dim=32,
                dropout=0.1,
            )

    def forward(
        self,
        signals: Union[torch.Tensor, np.ndarray, List[np.ndarray], List[torch.Tensor]],
        rr_intervals: Optional[torch.Tensor] = None,
        sample_rates: Optional[Union[int, List[int]]] = None,
    ) -> torch.Tensor:
        """
        Process HR signals through the complete LSTM pipeline.

        Args:
            signals: Input HR signals
            rr_intervals: RR intervals in seconds (optional, for HRV features)
            sample_rates: Sample rates of input signals

        Returns:
            features: HR features (batch_size, output_dim)
        """
        # Preprocess signals
        processed_signals = self.preprocessor(signals, sample_rates)
        processed_signals = processed_signals.to(self.device)

        # Extract LSTM features
        lstm_features = self.lstm_extractor(processed_signals)

        # Extract HRV features if enabled and RR intervals provided
        if self.use_hrv_features and rr_intervals is not None:
            rr_intervals = rr_intervals.to(self.device)
            hrv_features = self.hrv_extractor(rr_intervals)

            # Concatenate LSTM and HRV features
            features = torch.cat([lstm_features, hrv_features], dim=1)
        else:
            features = lstm_features

        return features
