"""
Visual data preprocessing for ACMFT.

This module provides functions for preprocessing visual data (face images/videos)
for the ACMFT emotion recognition system.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

try:
    from ultralytics import YOLO
except ImportError:
    logging.warning("ultralytics not installed. Install with: pip install ultralytics")
    YOLO = None


def preprocess_face_image(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
    detect_face: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> np.ndarray:
    """
    Preprocess a face image for the visual encoder.

    Args:
        image_path: Path to the image file
        target_size: Target image size (height, width)
        normalize: Whether to normalize pixel values to [-1, 1]
        detect_face: Whether to detect and crop the face
        device: Device to use for face detection

    Returns:
        preprocessed_image: Preprocessed face image
    """
    # Load image
    image_path = Path(image_path)

    if not image_path.exists():
        logging.warning(f"Image file not found: {image_path}")
        # Return black image
        return np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)

    # Check if it's an image or video file
    is_video = image_path.suffix.lower() in [".mp4", ".avi", ".mov"]

    if is_video:
        # Extract a frame from the video
        return preprocess_face_video(
            video_path=image_path,
            target_size=target_size,
            normalize=normalize,
            detect_face=detect_face,
            device=device,
            extract_single_frame=True,
        )

    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        logging.warning(f"Failed to read image: {image_path}")
        return np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect and crop face if requested
    if detect_face:
        face = detect_and_crop_face(image, device=device)
        if face is not None:
            image = face

    # Resize image
    image = cv2.resize(image, (target_size[1], target_size[0]))

    # Normalize pixel values if requested
    if normalize:
        image = image.astype(np.float32) / 127.5 - 1.0

    return image


def preprocess_face_video(
    video_path: Union[str, Path],
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
    detect_face: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_frames: Optional[int] = None,
    extract_single_frame: bool = False,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Preprocess a face video for the visual encoder.

    Args:
        video_path: Path to the video file
        target_size: Target image size (height, width)
        normalize: Whether to normalize pixel values to [-1, 1]
        detect_face: Whether to detect and crop the face
        device: Device to use for face detection
        max_frames: Maximum number of frames to extract (None for all)
        extract_single_frame: Whether to extract a single frame (middle frame)

    Returns:
        preprocessed_frames: Preprocessed video frames or single frame
    """
    # Load video
    video_path = Path(video_path)

    if not video_path.exists():
        logging.warning(f"Video file not found: {video_path}")
        # Return black frame(s)
        if extract_single_frame:
            return np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)
        else:
            return [np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)]

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.warning(f"Failed to open video: {video_path}")
        if extract_single_frame:
            return np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)
        else:
            return [np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)]

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if extract_single_frame:
        # Extract middle frame
        middle_frame_idx = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            logging.warning(f"Failed to read frame from video: {video_path}")
            return np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect and crop face if requested
        if detect_face:
            face = detect_and_crop_face(frame, device=device)
            if face is not None:
                frame = face

        # Resize frame
        frame = cv2.resize(frame, (target_size[1], target_size[0]))

        # Normalize pixel values if requested
        if normalize:
            frame = frame.astype(np.float32) / 127.5 - 1.0

        return frame

    # Extract multiple frames
    frames = []
    frame_stride = 1

    if max_frames is not None and total_frames > max_frames:
        frame_stride = total_frames // max_frames

    # Initialize face detector if needed
    face_detector = None
    if detect_face:
        # Initialize YOLOv11 face detector
        face_detector = YOLO("yolo11n.pt")
        face_detector.conf = 0.5  # Confidence threshold
        face_detector.iou = 0.5  # IoU threshold

    # Read frames
    frame_idx = 0
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            break

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect and crop face if requested
        if detect_face:
            face = detect_and_crop_face(frame, device=device, detector=face_detector, margin_percent=20)
            if face is not None:
                frame = face

        # Resize frame
        frame = cv2.resize(frame, (target_size[1], target_size[0]))

        # Normalize pixel values if requested
        if normalize:
            frame = frame.astype(np.float32) / 127.5 - 1.0

        frames.append(frame)
        frame_idx += frame_stride

    cap.release()

    if not frames:
        logging.warning(f"No frames extracted from video: {video_path}")
        return [np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)]

    return frames


def detect_and_crop_face(
    image: np.ndarray,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    detector: Optional[YOLO] = None,
    margin_percent: int = 20,
) -> Optional[np.ndarray]:
    """
    Detect and crop the face from an image using YOLOv11.

    Args:
        image: Input image
        device: Device to use for face detection
        detector: Optional pre-initialized YOLO detector
        margin_percent: Percentage of margin to add around the face

    Returns:
        face: Cropped face image, or None if no face detected
    """
    # Check if YOLO is available
    if YOLO is None:
        logging.warning("YOLO not available, face detection skipped")
        return image

    # Initialize detector if not provided
    if detector is None:
        detector = YOLO("yolo11n.pt")
        detector.conf = 0.5  # Confidence threshold
        detector.iou = 0.5  # IoU threshold

    # Detect face
    try:
        # Run detection
        results = detector(image, verbose=False)

        # Check if any faces detected
        if len(results) == 0 or len(results[0].boxes) == 0:
            logging.debug("No face detected in image")
            return None

        # Get detection boxes
        boxes = results[0].boxes

        if len(boxes) == 0:
            logging.debug("No face detected in image")
            return None

        # Get the box with highest confidence
        confs = boxes.conf
        best_idx = int(torch.argmax(confs))

        # Get coordinates (x1, y1, x2, y2)
        box = boxes.xyxy[best_idx].cpu().numpy()

        # Expand bounding box by specified margin to include more context
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        margin_x, margin_y = width * (margin_percent / 100), height * (margin_percent / 100)

        x1 = max(0, int(x1 - margin_x))
        y1 = max(0, int(y1 - margin_y))
        x2 = min(image.shape[1], int(x2 + margin_x))
        y2 = min(image.shape[0], int(y2 + margin_y))

        # Crop face
        face = image[y1:y2, x1:x2]

        # Ensure the cropped region is valid
        if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
            logging.debug("Invalid face crop, returning original image")
            return image

        return face

    except Exception as e:
        logging.warning(f"Error during face detection: {e}", exc_info=e)
        raise e
        return image
