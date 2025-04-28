"""
Audio data preprocessing for ACMFT.

This module provides functions for preprocessing audio data
for the ACMFT emotion recognition system.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import numpy as np


def preprocess_audio(
    audio_path: Union[str, Path],
    target_sr: int = 16000,
    target_length: Optional[int] = None,
    normalize: bool = True,
    trim_silence: bool = True,
    apply_filters: bool = False,
    extract_features: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
    """
    Preprocess audio for the audio encoder.

    Args:
        audio_path: Path to the audio file
        target_sr: Target sampling rate
        target_length: Target length in samples (None for original length)
        normalize: Whether to normalize audio to [-1, 1]
        trim_silence: Whether to trim leading and trailing silence
        apply_filters: Whether to apply preprocessing filters (HPF, LPF)
        extract_features: Whether to extract features instead of returning raw waveform

    Returns:
        Either:
            preprocessed_audio: Preprocessed audio waveform
            sr: Sampling rate
        Or:
            features: Extracted audio features if extract_features=True
    """
    # Load audio
    audio_path = Path(audio_path)

    if not audio_path.exists():
        logging.warning(f"Audio file not found: {audio_path}")
        # Return empty audio
        if extract_features:
            return np.zeros((128, 128), dtype=np.float32)
        else:
            return np.zeros(target_length or 16000, dtype=np.float32), target_sr

    try:
        # Load audio with librosa
        audio, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        logging.warning(f"Failed to load audio file {audio_path}: {e}")
        if extract_features:
            return np.zeros((128, 128), dtype=np.float32)
        else:
            return np.zeros(target_length or 16000, dtype=np.float32), target_sr

    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Trim silence
    if trim_silence:
        audio, _ = librosa.effects.trim(audio, top_db=30)

    # Apply filters
    if apply_filters:
        # High-pass filter to remove very low frequencies (< 80 Hz)
        audio = librosa.effects.preemphasis(audio, coef=0.97)

        # Low-pass filter to remove high frequencies (> 8000 Hz)
        # This is a simple approximation of low-pass filtering
        if sr > 16000:
            nyquist = sr // 2
            cutoff = 8000 / nyquist
            b, a = librosa.filters.butter(4, cutoff, btype="low")
            audio = librosa.filters.filtfilt(b, a, audio)

    # Adjust length if target_length is specified
    if target_length is not None:
        if len(audio) < target_length:
            # Pad if too short
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding), "constant")
        elif len(audio) > target_length:
            # Crop if too long
            audio = audio[:target_length]

    # Normalize
    if normalize:
        # Normalize to [-1, 1]
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

    # Extract features if requested
    if extract_features:
        features = extract_audio_features(audio, sr=sr)
        return features
    else:
        return audio, sr


def extract_audio_features(
    audio: np.ndarray,
    sr: int = 16000,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    feature_type: str = "melspectrogram",
) -> np.ndarray:
    """
    Extract audio features from waveform.

    Args:
        audio: Audio waveform
        sr: Sampling rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length between frames
        feature_type: Feature type ("melspectrogram", "mfcc", "chroma")

    Returns:
        features: Extracted audio features
    """
    if feature_type == "melspectrogram":
        # Extract mel spectrogram
        S = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        # Convert to log scale
        S_db = librosa.power_to_db(S, ref=np.max)
        # Normalize to [0, 1]
        S_db_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
        features = S_db_norm

    elif feature_type == "mfcc":
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        # Normalize
        mfccs_norm = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / (np.std(mfccs, axis=1, keepdims=True) + 1e-8)
        features = mfccs_norm

    elif feature_type == "chroma":
        # Extract chroma features
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        features = chroma

    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")

    return features


def augment_audio(
    audio: np.ndarray,
    sr: int = 16000,
    noise_level: float = 0.005,
    pitch_shift: Optional[int] = None,
    speed_perturb: Optional[float] = None,
    time_stretch: Optional[float] = None,
) -> np.ndarray:
    """
    Apply data augmentation to audio.

    Args:
        audio: Audio waveform
        sr: Sampling rate
        noise_level: Noise level for adding Gaussian noise (0 to disable)
        pitch_shift: Pitch shift in semitones (None to disable)
        speed_perturb: Speed perturbation factor (None to disable)
        time_stretch: Time stretch factor (None to disable)

    Returns:
        augmented_audio: Augmented audio waveform
    """
    # Start with a copy of the original audio
    augmented_audio = audio.copy()

    # Add Gaussian noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, len(audio))
        augmented_audio = augmented_audio + noise

        # Ensure the augmented audio is still in the valid range
        augmented_audio = np.clip(augmented_audio, -1.0, 1.0)

    # Pitch shifting
    if pitch_shift is not None:
        augmented_audio = librosa.effects.pitch_shift(augmented_audio, sr=sr, n_steps=pitch_shift)

    # Speed perturbation (resampling)
    if speed_perturb is not None:
        augmented_audio = librosa.effects.time_stretch(augmented_audio, rate=speed_perturb)

    # Time stretching (without changing pitch)
    if time_stretch is not None:
        augmented_audio = librosa.effects.time_stretch(augmented_audio, rate=time_stretch)

    return augmented_audio


def apply_spec_augment(
    spectrogram: np.ndarray,
    time_mask_param: int = 40,
    freq_mask_param: int = 30,
    n_time_masks: int = 1,
    n_freq_masks: int = 1,
) -> np.ndarray:
    """
    Apply SpecAugment to a spectrogram.

    Args:
        spectrogram: Input spectrogram (freq, time)
        time_mask_param: Maximum time mask length
        freq_mask_param: Maximum frequency mask length
        n_time_masks: Number of time masks
        n_freq_masks: Number of frequency masks

    Returns:
        augmented_spectrogram: Augmented spectrogram
    """
    # Start with a copy of the original spectrogram
    augmented_spec = spectrogram.copy()

    # Get dimensions
    freq_bins, time_bins = augmented_spec.shape

    # Apply time masks
    for _ in range(n_time_masks):
        t = np.random.randint(0, time_mask_param)
        t0 = np.random.randint(0, time_bins - t)
        augmented_spec[:, t0 : t0 + t] = 0

    # Apply frequency masks
    for _ in range(n_freq_masks):
        f = np.random.randint(0, freq_mask_param)
        f0 = np.random.randint(0, freq_bins - f)
        augmented_spec[f0 : f0 + f, :] = 0

    return augmented_spec
