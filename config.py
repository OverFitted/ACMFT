"""
Configuration settings for the ACMFT emotion recognition system.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    # Input dimensions for each modality
    visual_dim: int = 512  # Output dimension of visual encoder (ResNet/FaceNet)
    audio_dim: int = 768  # Output dimension of audio encoder (wav2vec2)
    hr_dim: int = 128  # Output dimension of HR encoder (1D CNN)

    # Transformer architecture
    hidden_dim: int = 256  # Hidden dimension in transformer
    num_heads: int = 8  # Number of attention heads
    num_layers: int = 4  # Number of transformer layers
    dropout: float = 0.1  # Dropout rate
    activation: str = "gelu"  # Activation function in transformer

    # Output
    num_emotions: int = 8  # Number of emotion classes


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Basic training parameters
    batch_size: int = 32
    num_epochs: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Learning rate scheduler
    lr_scheduler: str = "cosine"  # Options: "cosine", "step", "plateau"
    lr_scheduler_params: Dict = None
    warmup_epochs: int = 1

    # Optimization
    optimizer: str = "adamw"  # Options: "adam", "adamw", "sgd"
    gradient_clip_val: float = 1.0

    # Early stopping
    patience: int = 10
    min_delta: float = 0.001

    # Mixed precision training
    use_amp: bool = True  # Use automatic mixed precision

    # Progressive training
    freeze_encoders: bool = True  # Initially freeze the modality encoders
    unfreeze_after: int = 10  # Unfreeze encoders after N epochs

    def __post_init__(self):
        if self.lr_scheduler_params is None:
            if self.lr_scheduler == "cosine":
                self.lr_scheduler_params = {}
            elif self.lr_scheduler == "step":
                self.lr_scheduler_params = {"step_size": 30, "gamma": 0.1}
            elif self.lr_scheduler == "plateau":
                self.lr_scheduler_params = {"factor": 0.1, "patience": 5}


@dataclass
class DataConfig:
    """Configuration for data processing."""

    # Dataset paths (to be set by the user)
    iemocap_path: str = ""  # Path to IEMOCAP dataset
    ravdess_path: str = ""  # Path to RAVDESS dataset

    # Data splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Cross-validation
    num_folds: int = 5

    # Data augmentation
    use_augmentation: bool = True

    # Visual augmentation
    visual_augmentation: Dict = None

    # Audio augmentation
    audio_augmentation: Dict = None

    # HR augmentation
    hr_augmentation: Dict = None

    # Emotion mapping (customize based on your needs)
    emotion_mapping: Dict[str, int] = None

    def __post_init__(self):
        if self.emotion_mapping is None:
            self.emotion_mapping = {
                "happiness": 0,
                "sadness": 1,
                "anger": 2,
                "fear": 3,
                "surprise": 4,
                "disgust": 5,
                "contempt": 6,
                "neutral": 7,
            }

        if self.visual_augmentation is None:
            self.visual_augmentation = {
                "random_crop": True,
                "random_flip": True,
                "color_jitter": True,
                "rotation": True,
                "occlusion": True,
            }

        if self.audio_augmentation is None:
            self.audio_augmentation = {"noise": True, "pitch_shift": True, "time_stretch": True, "spec_augment": True}

        if self.hr_augmentation is None:
            self.hr_augmentation = {"noise": True, "masking": True, "time_warp": True}


@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000

    # Model serving
    model_path: str = "models/acmft_final.pt"
    device: str = "cuda"  # "cuda" or "cpu"

    # Inference settings
    batch_size: int = 1
    use_quantization: bool = False
    timeout: float = 30.0  # Timeout in seconds

    # Caching
    use_cache: bool = True
    cache_size: int = 100


# Create a global configuration
class ACMFTConfig:
    """Global configuration for ACMFT system."""

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        data_config: Optional[DataConfig] = None,
        deployment_config: Optional[DeploymentConfig] = None,
    ):
        self.model = model_config or ModelConfig()
        self.training = training_config or TrainingConfig()
        self.data = data_config or DataConfig()
        self.deployment = deployment_config or DeploymentConfig()


# Default configuration
default_config = ACMFTConfig()
