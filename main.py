"""
Main script for training and evaluating the ACMFT emotion recognition model.

This script serves as the entry point for the ACMFT emotion recognition system,
handling command-line arguments, model initialization, training, and evaluation.
"""

import argparse
import json
import logging
import os

import numpy as np
import torch

from config import ACMFTConfig, DataConfig, ModelConfig, TrainingConfig
from data.datasets import create_emotion_dataloaders
from models.acmft import ACMFT
from models.modality_encoders.audio import AudioEncoder
from models.modality_encoders.hr import CNNHREncoder
from models.modality_encoders.visual import VisualEncoder
from training.trainer import ACMFTTrainer
from utils.visualization import plot_confusion_matrix, plot_training_history


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        args: Command-line arguments
    """
    parser = argparse.ArgumentParser(description="ACMFT Emotion Recognition")

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Operation mode: train or test",
    )

    # Data paths
    parser.add_argument("--iemocap_dir", type=str, default=None, help="Path to IEMOCAP dataset")
    parser.add_argument("--ravdess_dir", type=str, default=None, help="Path to RAVDESS dataset")

    # Model configuration
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training configuration
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory for saving checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory for saving logs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    # Modality selection
    parser.add_argument("--modalities", type=str, nargs="+", default=["visual", "audio"], help="Modalities to use")

    # Device configuration
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda or cpu)")

    # Add log level argument
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

    # Parse arguments
    args = parser.parse_args()

    # Set device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Configure logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s - %(levelname)s - %(message)s")

    return args


def create_model(args, config: ACMFTConfig) -> ACMFT:
    """
    Create and initialize the ACMFT model.

    Args:
        args: Command-line arguments
        config: Model configuration

    Returns:
        model: Initialized ACMFT model
    """
    # Create modality encoders
    visual_encoder = None
    audio_encoder = None
    hr_encoder = None

    if "visual" in args.modalities:
        visual_encoder = VisualEncoder(
            face_detector_type="yolo",
            feature_extractor_type="facenet",
            output_dim=config.model.visual_dim,
            device=args.device,
        )

    if "audio" in args.modalities:
        audio_encoder = AudioEncoder(
            model_name="facebook/wav2vec2-base-960h",
            output_dim=config.model.audio_dim,
            feature_aggregation="mean",
            device=args.device,
        )

    if "hr" in args.modalities:
        hr_encoder = CNNHREncoder(
            output_dim=config.model.hr_dim,
            device=args.device,
        )

    # Create ACMFT model
    model = ACMFT(
        config=config.model,
        visual_encoder=visual_encoder,
        audio_encoder=audio_encoder,
        hr_encoder=hr_encoder,
    )

    # Move model to device
    model = model.to(args.device)

    return model


def train_model(args, config: ACMFTConfig):
    """
    Train the ACMFT model.

    Args:
        args: Command-line arguments
        config: Model configuration
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create dataloaders
    dataloaders = create_emotion_dataloaders(
        iemocap_dir=args.iemocap_dir,
        ravdess_dir=args.ravdess_dir,
        batch_size=config.training.batch_size,
        modalities=args.modalities,
        preprocess=True,
        cache_processed=False,  # disable cache to ensure fresh audio/hr data
    )

    # Create model
    model = create_model(args, config)

    # Create trainer
    trainer = ACMFTTrainer(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        config=config.training,
        device=args.device,
        experiment_name="acmft",
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )

    # Resume from checkpoint if specified
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    # Train model
    history = trainer.train()

    # Plot training history
    plot_training_history(
        history["train_loss"],
        history["val_loss"],
        history["val_acc"],
        save_path=os.path.join(args.log_dir, "training_history.png"),
    )

    # Test model
    test_metrics = trainer.test(dataloaders["test"])

    # Plot confusion matrix
    plot_confusion_matrix(test_metrics["confusion_matrix"], save_path=os.path.join(args.log_dir, "confusion_matrix.png"))

    # Print test metrics
    logging.info("\nTest Metrics:")
    logging.info(f"Loss: {test_metrics['loss']:.4f}")
    logging.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
    logging.info(f"Macro F1: {test_metrics['macro_f1']:.4f}")
    logging.info(f"Weighted F1: {test_metrics['weighted_f1']:.4f}")

    # Save metrics to file
    with open(os.path.join(args.log_dir, "test_metrics.json"), "w") as f:
        json.dump(
            {
                "loss": test_metrics["loss"],
                "accuracy": test_metrics["accuracy"],
                "macro_precision": test_metrics["macro_precision"],
                "macro_recall": test_metrics["macro_recall"],
                "macro_f1": test_metrics["macro_f1"],
                "weighted_f1": test_metrics["weighted_f1"],
                "class_metrics": [
                    {
                        "class": i,
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                    }
                    for i, metrics in enumerate(test_metrics["class_metrics"])
                ],
            },
            f,
            indent=4,
        )


def test_model(args, config: ACMFTConfig):
    """
    Test the ACMFT model.

    Args:
        args: Command-line arguments
        config: Model configuration
    """
    # Create dataloaders
    dataloaders = create_emotion_dataloaders(
        iemocap_dir=args.iemocap_dir,
        ravdess_dir=args.ravdess_dir,
        batch_size=config.training.batch_size,
        modalities=args.modalities,
        preprocess=True,
        cache_processed=False,  # disable cache to ensure fresh audio/hr data
    )

    # Create model
    model = create_model(args, config)

    # Load checkpoint
    if args.resume is None:
        raise ValueError("Must specify checkpoint to load for testing")

    checkpoint = torch.load(args.resume, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Create trainer
    trainer = ACMFTTrainer(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        config=config.training,
        device=args.device,
        experiment_name="acmft",
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )

    # Test model
    test_metrics = trainer.test(dataloaders["test"])

    # Plot confusion matrix
    plot_confusion_matrix(test_metrics["confusion_matrix"], save_path=os.path.join(args.log_dir, "confusion_matrix.png"))

    # Print test metrics
    logging.info("\nTest Metrics:")
    logging.info(f"Loss: {test_metrics['loss']:.4f}")
    logging.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
    logging.info(f"Macro F1: {test_metrics['macro_f1']:.4f}")
    logging.info(f"Weighted F1: {test_metrics['weighted_f1']:.4f}")

    # Save metrics to file
    with open(os.path.join(args.log_dir, "test_metrics.json"), "w") as f:
        json.dump(
            {
                "loss": test_metrics["loss"],
                "accuracy": test_metrics["accuracy"],
                "macro_precision": test_metrics["macro_precision"],
                "macro_recall": test_metrics["macro_recall"],
                "macro_f1": test_metrics["macro_f1"],
                "weighted_f1": test_metrics["weighted_f1"],
                "class_metrics": [
                    {
                        "class": i,
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                    }
                    for i, metrics in enumerate(test_metrics["class_metrics"])
                ],
            },
            f,
            indent=4,
        )


def main():
    """
    Main function.
    """
    # Parse command-line arguments
    args = parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(args.log_dir, "acmft.log"))],
    )

    # Create config
    model_config = ModelConfig(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    training_config = TrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    data_config = DataConfig(
        iemocap_path=args.iemocap_dir,
        ravdess_path=args.ravdess_dir,
    )

    config = ACMFTConfig(
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
    )

    # Run selected mode
    if args.mode == "train":
        train_model(args, config)
    elif args.mode == "test":
        test_model(args, config)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
