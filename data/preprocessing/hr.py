"""
Heart Rate (HR) signal preprocessing for ACMFT.

This module provides functions for preprocessing physiological HR signals
for the ACMFT emotion recognition system.
"""

import numpy as np
import torch
import scipy.signal
from typing import Union, Tuple, Optional, List, Dict
from pathlib import Path
import logging
import pandas as pd


def preprocess_hr_signal(
    hr_path: Union[str, Path],
    target_sr: int = 256,
    target_length: Optional[int] = None,
    normalize: bool = True,
    bandpass_filter: bool = True,
    lowcut: float = 0.5,
    highcut: float = 4.0,
    remove_artifacts: bool = True,
    extract_features: bool = False,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Preprocess HR signal for the HR encoder.
    
    Args:
        hr_path: Path to the HR signal file (CSV, NPY, etc.)
        target_sr: Target sampling rate
        target_length: Target length in samples (None for original length)
        normalize: Whether to normalize signal
        bandpass_filter: Whether to apply bandpass filtering
        lowcut: Low cutoff frequency for bandpass filter
        highcut: High cutoff frequency for bandpass filter
        remove_artifacts: Whether to remove artifacts from the signal
        extract_features: Whether to extract HR features
        
    Returns:
        Either:
            preprocessed_signal: Preprocessed HR signal
        Or:
            features: Dictionary of extracted HR features
    """
    # Load HR signal
    signal = load_hr_signal(hr_path)
    
    if signal is None:
        # Return empty signal
        if extract_features:
            return {
                'hr_signal': np.zeros(target_length or 2560, dtype=np.float32),
                'hr_mean': 0.0,
                'hr_std': 0.0,
                'rmssd': 0.0,
                'sdnn': 0.0,
                'pnn50': 0.0,
                'lf_hf_ratio': 0.0,
            }
        else:
            return np.zeros(target_length or 2560, dtype=np.float32)
    
    # Apply bandpass filter
    if bandpass_filter:
        signal = bandpass_filter_signal(
            signal, 
            lowcut=lowcut, 
            highcut=highcut, 
            fs=target_sr, 
            order=4
        )
    
    # Remove artifacts
    if remove_artifacts:
        signal = remove_signal_artifacts(signal)
    
    # Adjust length if target_length is specified
    if target_length is not None:
        signal = adjust_signal_length(signal, target_length)
    
    # Normalize
    if normalize:
        signal = normalize_signal(signal)
    
    # Extract features
    if extract_features:
        features = extract_hr_features(signal, fs=target_sr)
        return features
    else:
        return signal


def load_hr_signal(
    file_path: Union[str, Path]
) -> Optional[np.ndarray]:
    """
    Load HR signal from a file.
    
    Args:
        file_path: Path to the HR signal file
        
    Returns:
        signal: HR signal as numpy array, or None if loading fails
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logging.warning(f"HR signal file not found: {file_path}")
        return None
    
    try:
        # Load based on file extension
        if file_path.suffix.lower() == '.npy':
            # Load numpy file
            signal = np.load(file_path)
        elif file_path.suffix.lower() == '.csv':
            # Load CSV file
            df = pd.read_csv(file_path)
            
            # Try to find the column with HR data
            hr_column = None
            for possible_name in ['hr', 'HR', 'heart_rate', 'Heart Rate', 'HeartRate', 'value']:
                if possible_name in df.columns:
                    hr_column = possible_name
                    break
            
            if hr_column is None and len(df.columns) > 1:
                # Assume the second column contains HR data
                hr_column = df.columns[1]
            
            if hr_column is None:
                # Just use the first column
                hr_column = df.columns[0]
            
            signal = df[hr_column].values
        elif file_path.suffix.lower() in ['.txt', '.dat']:
            # Load text file
            signal = np.loadtxt(file_path)
        else:
            logging.warning(f"Unsupported HR signal file format: {file_path.suffix}")
            return None
        
        # Ensure signal is 1D
        if signal.ndim > 1:
            signal = signal.flatten()
        
        return signal
    
    except Exception as e:
        logging.warning(f"Failed to load HR signal file {file_path}: {e}")
        return None


def bandpass_filter_signal(
    signal: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4
) -> np.ndarray:
    """
    Apply bandpass filter to signal.
    
    Args:
        signal: Input signal
        lowcut: Low cutoff frequency
        highcut: High cutoff frequency
        fs: Sampling frequency
        order: Filter order
        
    Returns:
        filtered_signal: Filtered signal
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Create bandpass filter
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    
    # Apply filter
    filtered_signal = scipy.signal.filtfilt(b, a, signal)
    
    return filtered_signal


def remove_signal_artifacts(
    signal: np.ndarray,
    threshold: float = 3.0
) -> np.ndarray:
    """
    Remove artifacts from signal using a threshold-based approach.
    
    Args:
        signal: Input signal
        threshold: Z-score threshold for identifying artifacts
        
    Returns:
        cleaned_signal: Signal with artifacts removed
    """
    # Calculate mean and standard deviation
    mean = np.mean(signal)
    std = np.std(signal)
    
    # Calculate z-scores
    z_scores = np.abs((signal - mean) / std)
    
    # Create mask for values within threshold
    mask = z_scores <= threshold
    
    # Create cleaned signal
    cleaned_signal = signal.copy()
    
    # Replace artifacts with mean or interpolated values
    for i in range(len(signal)):
        if not mask[i]:
            # Find nearest valid values before and after
            before_idx = i - 1
            while before_idx >= 0 and not mask[before_idx]:
                before_idx -= 1
            
            after_idx = i + 1
            while after_idx < len(signal) and not mask[after_idx]:
                after_idx += 1
            
            # Interpolate or use mean
            if before_idx >= 0 and after_idx < len(signal):
                # Linear interpolation
                before_val = signal[before_idx]
                after_val = signal[after_idx]
                weight = (i - before_idx) / (after_idx - before_idx)
                cleaned_signal[i] = before_val * (1 - weight) + after_val * weight
            else:
                # Use mean if interpolation not possible
                cleaned_signal[i] = mean
    
    return cleaned_signal


def adjust_signal_length(
    signal: np.ndarray,
    target_length: int
) -> np.ndarray:
    """
    Adjust signal length to target length.
    
    Args:
        signal: Input signal
        target_length: Target length
        
    Returns:
        adjusted_signal: Signal with adjusted length
    """
    current_length = len(signal)
    
    if current_length == target_length:
        # No adjustment needed
        return signal
    
    elif current_length < target_length:
        # Pad signal to target length
        padding = target_length - current_length
        return np.pad(signal, (0, padding), mode='constant', constant_values=signal[-1])
    
    else:
        # Crop signal to target length
        return signal[:target_length]


def normalize_signal(
    signal: np.ndarray,
    method: str = 'zscore'
) -> np.ndarray:
    """
    Normalize signal.
    
    Args:
        signal: Input signal
        method: Normalization method ('zscore', 'minmax', 'unit')
        
    Returns:
        normalized_signal: Normalized signal
    """
    if method == 'zscore':
        # Z-score normalization
        mean = np.mean(signal)
        std = np.std(signal)
        if std > 0:
            return (signal - mean) / std
        else:
            return signal - mean
    
    elif method == 'minmax':
        # Min-max normalization to [0, 1]
        min_val = np.min(signal)
        max_val = np.max(signal)
        if max_val > min_val:
            return (signal - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(signal)
    
    elif method == 'unit':
        # Unit normalization
        norm = np.linalg.norm(signal)
        if norm > 0:
            return signal / norm
        else:
            return signal
    
    else:
        raise ValueError(f"Unsupported normalization method: {method}")


def extract_hr_features(
    signal: np.ndarray,
    fs: float = 256.0
) -> Dict[str, np.ndarray]:
    """
    Extract HR features from signal.
    
    Args:
        signal: Input HR signal
        fs: Sampling frequency
        
    Returns:
        features: Dictionary of HR features
    """
    # Basic time-domain features
    mean_hr = np.mean(signal)
    std_hr = np.std(signal)
    
    # Detect R-peaks for HRV analysis
    r_peaks, _ = detect_r_peaks(signal, fs)
    
    # Calculate RR intervals
    rr_intervals = np.diff(r_peaks) / fs  # in seconds
    
    # HRV features
    if len(rr_intervals) > 1:
        # RMSSD - Root Mean Square of Successive Differences
        rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
        
        # SDNN - Standard Deviation of NN intervals
        sdnn = np.std(rr_intervals)
        
        # pNN50 - Percentage of successive RR intervals differing by more than 50ms
        pnn50 = np.sum(np.abs(np.diff(rr_intervals)) > 0.05) / len(np.diff(rr_intervals))
        
        # Frequency domain features
        lf_hf_ratio = calculate_lf_hf_ratio(rr_intervals)
    else:
        rmssd = 0.0
        sdnn = 0.0
        pnn50 = 0.0
        lf_hf_ratio = 1.0
    
    # Create features dictionary
    features = {
        'hr_signal': signal,
        'hr_mean': mean_hr,
        'hr_std': std_hr,
        'rmssd': rmssd,
        'sdnn': sdnn,
        'pnn50': pnn50,
        'lf_hf_ratio': lf_hf_ratio,
    }
    
    return features


def detect_r_peaks(
    signal: np.ndarray,
    fs: float = 256.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect R-peaks in ECG/HR signal.
    
    Args:
        signal: Input signal
        fs: Sampling frequency
        
    Returns:
        r_peaks: Sample indices of detected R-peaks
        r_peak_amplitudes: Amplitudes of detected R-peaks
    """
    # Filter signal for better peak detection
    filtered_signal = bandpass_filter_signal(signal, 5.0, 15.0, fs)
    
    # Find peaks with minimum distance based on sampling rate
    # (assuming maximum heart rate of 180 BPM)
    min_distance = int(fs * 60 / 180)
    
    # Find peaks
    peaks, _ = scipy.signal.find_peaks(
        filtered_signal,
        distance=min_distance,
        height=0.5 * np.std(filtered_signal)
    )
    
    # Get peak amplitudes
    peak_amplitudes = signal[peaks]
    
    return peaks, peak_amplitudes


def calculate_lf_hf_ratio(
    rr_intervals: np.ndarray,
    lf_band: Tuple[float, float] = (0.04, 0.15),
    hf_band: Tuple[float, float] = (0.15, 0.4)
) -> float:
    """
    Calculate LF/HF ratio from RR intervals.
    
    Args:
        rr_intervals: RR intervals in seconds
        lf_band: Low frequency band (Hz)
        hf_band: High frequency band (Hz)
        
    Returns:
        lf_hf_ratio: LF/HF ratio
    """
    # Check if there are enough intervals
    if len(rr_intervals) < 4:
        return 1.0  # Default value
    
    try:
        # Interpolate RR intervals to get evenly sampled signal
        rr_x = np.cumsum(rr_intervals)
        rr_x = rr_x - rr_x[0]  # Start at 0
        
        # 4Hz sampling for interpolation
        fs = 4.0
        rr_x_interp = np.arange(0, rr_x[-1], 1/fs)
        rr_y_interp = np.interp(rr_x_interp, rr_x, rr_intervals)
        
        # Remove mean
        rr_y_interp = rr_y_interp - np.mean(rr_y_interp)
        
        # Calculate PSD
        f, psd = scipy.signal.welch(rr_y_interp, fs, nperseg=len(rr_y_interp))
        
        # Calculate power in LF and HF bands
        lf_power = np.trapz(psd[(f >= lf_band[0]) & (f <= lf_band[1])], 
                            f[(f >= lf_band[0]) & (f <= lf_band[1])])
        
        hf_power = np.trapz(psd[(f >= hf_band[0]) & (f <= hf_band[1])], 
                            f[(f >= hf_band[0]) & (f <= hf_band[1])])
        
        # Calculate ratio
        if hf_power > 0:
            lf_hf_ratio = lf_power / hf_power
        else:
            lf_hf_ratio = 1.0
        
        return lf_hf_ratio
    
    except Exception as e:
        logging.warning(f"Error calculating LF/HF ratio: {e}")
        return 1.0  # Default value


def augment_hr_signal(
    signal: np.ndarray,
    noise_level: float = 0.01,
    time_warp: bool = False,
    warp_factor: float = 0.1
) -> np.ndarray:
    """
    Apply data augmentation to HR signal.
    
    Args:
        signal: Input HR signal
        noise_level: Amplitude of Gaussian noise to add
        time_warp: Whether to apply time warping
        warp_factor: Factor for time warping
        
    Returns:
        augmented_signal: Augmented HR signal
    """
    # Start with a copy of the original signal
    augmented_signal = signal.copy()
    
    # Add Gaussian noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * np.std(signal), len(signal))
        augmented_signal = augmented_signal + noise
    
    # Apply time warping
    if time_warp:
        # Create time warping function
        x = np.linspace(0, 1, len(signal))
        warp = np.sin(2 * np.pi * x) * warp_factor
        warp_x = x + warp
        
        # Ensure warped x is within [0, 1]
        warp_x = np.clip(warp_x, 0, 1)
        
        # Sort warped x to ensure it's monotonically increasing
        sort_idx = np.argsort(warp_x)
        warp_x = warp_x[sort_idx]
        sorted_signal = augmented_signal[sort_idx]
        
        # Interpolate back to original grid
        augmented_signal = np.interp(x, warp_x, sorted_signal)
    
    return augmented_signal