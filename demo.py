"""
Demo script for the ACMFT emotion recognition system.

This script provides a simple demonstration of how to use the ACMFT model
for emotion recognition using either test data or webcam input.
"""

import argparse
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from config import ModelConfig
from models.acmft import ACMFT
from models.modality_encoders.audio import AudioEncoder
from models.modality_encoders.hr import CNNHREncoder
from models.modality_encoders.visual import VisualEncoder


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        args: Command-line arguments
    """
    parser = argparse.ArgumentParser(description="ACMFT Demo")

    # Demo mode
    parser.add_argument(
        "--mode", type=str, default="file", choices=["file", "webcam"], help="Demo mode: process files or use webcam"
    )

    # Input files
    parser.add_argument("--visual", type=str, default=None, help="Path to visual input (image or video file)")
    parser.add_argument("--audio", type=str, default=None, help="Path to audio input (wav file)")
    parser.add_argument("--hr", type=str, default=None, help="Path to HR signal input (npy or csv file)")

    # Model checkpoint
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")

    # Device
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda or cpu)")

    # Parse arguments
    args = parser.parse_args()

    # Set device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


def create_model(args) -> ACMFT:
    """
    Create and initialize the ACMFT model.

    Args:
        args: Command-line arguments

    Returns:
        model: Initialized ACMFT model
    """
    # Create model config
    config = ModelConfig(
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
    )

    # Create modality encoders
    visual_encoder = VisualEncoder(
        face_detector_type="mtcnn",
        feature_extractor_type="facenet",
        output_dim=config.visual_dim,
        device=args.device,
    )

    audio_encoder = AudioEncoder(
        model_name="facebook/wav2vec2-base-960h",
        output_dim=config.audio_dim,
        feature_aggregation="mean",
        device=args.device,
    )

    hr_encoder = CNNHREncoder(
        output_dim=config.hr_dim,
        device=args.device,
    )

    # Create ACMFT model
    model = ACMFT(
        config=config,
        visual_encoder=visual_encoder,
        audio_encoder=audio_encoder,
        hr_encoder=hr_encoder,
    )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Move model to device and set to evaluation mode
    model = model.to(args.device)
    model.eval()

    return model


def load_inputs(args) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Load inputs from files.

    Args:
        args: Command-line arguments

    Returns:
        visual_input: Visual input tensor or None
        audio_input: Audio input tensor or None
        hr_input: HR input tensor or None
    """
    visual_input = None
    audio_input = None
    hr_input = None

    # Load visual input
    if args.visual is not None:
        visual_path = Path(args.visual)

        if not visual_path.exists():
            print(f"Visual file not found: {visual_path}")
        else:
            try:
                if visual_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    # Load image
                    image = Image.open(visual_path).convert("RGB")
                    # Convert to tensor (C, H, W)
                    visual_input = torch.tensor(np.array(image)).permute(2, 0, 1).float()
                    # Add batch dimension
                    visual_input = visual_input.unsqueeze(0)
                elif visual_path.suffix.lower() in [".mp4", ".avi", ".mov"]:
                    # Extract frame from video
                    cap = cv2.VideoCapture(str(visual_path))
                    ret, frame = cap.read()
                    cap.release()

                    if ret:
                        # Convert BGR to RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Convert to tensor (C, H, W)
                        visual_input = torch.tensor(frame).permute(2, 0, 1).float()
                        # Add batch dimension
                        visual_input = visual_input.unsqueeze(0)
                    else:
                        print(f"Failed to read frame from video: {visual_path}")
            except Exception as e:
                print(f"Error loading visual input: {e}")

    # Load audio input
    if args.audio is not None:
        audio_path = Path(args.audio)

        if not audio_path.exists():
            print(f"Audio file not found: {audio_path}")
        else:
            try:
                # Load audio
                audio, sr = librosa.load(audio_path, sr=16000)
                # Convert to tensor
                audio_input = torch.tensor(audio).float()
                # Add batch dimension
                audio_input = audio_input.unsqueeze(0)
            except Exception as e:
                print(f"Error loading audio input: {e}")

    # Load HR input
    if args.hr is not None:
        hr_path = Path(args.hr)

        if not hr_path.exists():
            print(f"HR file not found: {hr_path}")
        else:
            try:
                if hr_path.suffix.lower() == ".npy":
                    # Load numpy file
                    hr_signal = np.load(hr_path)
                elif hr_path.suffix.lower() == ".csv":
                    # Load CSV file
                    import pandas as pd

                    df = pd.read_csv(hr_path)

                    # Try to find the column with HR data
                    hr_column = None
                    for possible_name in ["hr", "HR", "heart_rate", "HeartRate", "value"]:
                        if possible_name in df.columns:
                            hr_column = possible_name
                            break

                    if hr_column is None and len(df.columns) > 1:
                        # Assume the second column contains HR data
                        hr_column = df.columns[1]

                    if hr_column is None:
                        # Just use the first column
                        hr_column = df.columns[0]

                    hr_signal = df[hr_column].values
                else:
                    print(f"Unsupported HR file format: {hr_path.suffix}")
                    hr_signal = None

                if hr_signal is not None:
                    # Ensure signal is 1D
                    if hr_signal.ndim > 1:
                        hr_signal = hr_signal.flatten()

                    # Convert to tensor
                    hr_input = torch.tensor(hr_signal).float()
                    # Add batch dimension
                    hr_input = hr_input.unsqueeze(0)
            except Exception as e:
                print(f"Error loading HR input: {e}")

    return visual_input, audio_input, hr_input


def process_webcam_input(model, args):
    """
    Process webcam input for real-time emotion recognition.

    Args:
        model: ACMFT model
        args: Command-line arguments
    """
    # Emotion labels
    emotion_labels = ["Neutral", "Happiness", "Sadness", "Anger", "Fear", "Surprise", "Disgust", "Contempt"]

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Press 'q' to quit")

    while True:
        # Read frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame")
            break

        # Process frame
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to tensor
            input_tensor = torch.tensor(rgb_frame).permute(2, 0, 1).float().unsqueeze(0)
            input_tensor = input_tensor.to(args.device)

            # Run inference
            with torch.no_grad():
                emotion_probs, predicted_emotion = model.predict(visual=input_tensor)

            # Convert to numpy
            emotion_probs = emotion_probs.cpu().numpy()[0]
            predicted_emotion = predicted_emotion.cpu().numpy()[0]

            # Draw results on frame
            emotion_name = emotion_labels[predicted_emotion]
            confidence = emotion_probs[predicted_emotion]

            # Draw rectangle and text
            cv2.putText(
                frame,
                f"Emotion: {emotion_name} ({confidence:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Show frame
            cv2.imshow("ACMFT Emotion Recognition", frame)

        except Exception as e:
            print(f"Error processing frame: {e}")

        # Check for key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


def process_file_input(model, args):
    """
    Process file inputs for emotion recognition.

    Args:
        model: ACMFT model
        args: Command-line arguments
    """
    # Load inputs
    visual_input, audio_input, hr_input = load_inputs(args)

    # Check if any input is available
    if visual_input is None and audio_input is None and hr_input is None:
        print("Error: No valid input provided")
        return

    # Move inputs to device
    if visual_input is not None:
        visual_input = visual_input.to(args.device)
    if audio_input is not None:
        audio_input = audio_input.to(args.device)
    if hr_input is not None:
        hr_input = hr_input.to(args.device)

    # Emotion labels
    emotion_labels = ["Neutral", "Happiness", "Sadness", "Anger", "Fear", "Surprise", "Disgust", "Contempt"]

    # Run inference
    try:
        print("Running emotion recognition...")
        start_time = time.time()

        with torch.no_grad():
            emotion_probs, predicted_emotion = model.predict(visual=visual_input, audio=audio_input, hr=hr_input)

        end_time = time.time()
        inference_time = end_time - start_time

        # Convert to numpy
        emotion_probs = emotion_probs.cpu().numpy()[0]
        predicted_emotion = predicted_emotion.cpu().numpy()[0]

        # Print results
        print("\nResults:")
        print(f"Predicted emotion: {emotion_labels[predicted_emotion]}")
        print(f"Confidence: {emotion_probs[predicted_emotion]:.4f}")
        print(f"Inference time: {inference_time:.4f} seconds")

        # Print all probabilities
        print("\nEmotion Probabilities:")
        for i, label in enumerate(emotion_labels):
            print(f"{label}: {emotion_probs[i]:.4f}")

        # Plot probabilities
        plt.figure(figsize=(10, 6))
        plt.bar(emotion_labels, emotion_probs)
        plt.xlabel("Emotion")
        plt.ylabel("Probability")
        plt.title("Emotion Recognition Results")
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error during inference: {e}")


def main():
    """
    Main function.
    """
    # Parse command-line arguments
    args = parse_args()

    # Create model
    model = create_model(args)

    # Process input based on mode
    if args.mode == "file":
        process_file_input(model, args)
    elif args.mode == "webcam":
        process_webcam_input(model, args)
    else:
        print(f"Error: Unsupported mode {args.mode}")


if __name__ == "__main__":
    main()
