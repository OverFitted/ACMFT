"""
Visual encoder implementation for ACMFT.

This module implements the visual modality encoder that processes facial images
using YOLOv11 for face detection and alignment, followed by ResNet/FaceNet for
feature extraction.
"""

from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image

# Import ultralytics for YOLOv11 face detection
try:
    from facenet_pytorch import InceptionResnetV1  # Keep FaceNet for feature extraction
    from ultralytics import YOLO
except ImportError:
    print("Warning: required dependencies not installed. Install with: pip install ultralytics facenet-pytorch")


class YOLOFaceDetector(nn.Module):
    """
    YOLOv11-based face detector that detects and aligns faces in images.
    """

    def __init__(
        self,
        image_size: int = 160,
        margin: int = 20,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.image_size = image_size
        self.margin = margin
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        # Initialize YOLOv11 model
        self.yolo = YOLO("yolo11n.pt")

        # Configure model parameters
        self.yolo.conf = conf_threshold
        self.yolo.iou = iou_threshold
        self.yolo.to(device)

    def train(self, mode: bool = True):
        """Sets the module in training mode, handling potential conflicts with ultralytics."""
        # Set the training mode for this module itself
        self.training = mode
        # Iterate through all *direct* children modules registered with PyTorch
        for module in self.children():
            # Explicitly skip the 'yolo' submodule if it exists and has a 'train' method
            # The ultralytics YOLO model should not be trained in this context.
            if hasattr(self, "yolo") and module is self.yolo:
                # self.yolo.eval() # Keep yolo in eval mode if necessary
                continue  # Skip calling train on the yolo model
            module.train(mode)  # Call train on other standard PyTorch children
        return self

    def forward(
        self,
        images: Union[torch.Tensor, np.ndarray, List[np.ndarray], List[Image.Image]],
        return_probability: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Detect and align faces in the input images.

        Args:
            images: Input images (batch of images)
            return_probability: Whether to return face detection probability

        Returns:
            If return_probability is False:
                faces: Detected and aligned faces (batch_size, 3, image_size, image_size)
            If return_probability is True:
                faces: Detected and aligned faces (batch_size, 3, image_size, image_size)
                probs: Face detection probabilities (batch_size,)
        """
        try:  # Start of the main try block
            # Determine if input is a batch (Tensor or List)
            is_tensor_batch = isinstance(images, torch.Tensor) and images.ndim == 4
            is_list_batch = isinstance(images, list)
            batch_mode = is_tensor_batch or is_list_batch

            faces_list = []
            probs_list = []
            batch_size = 0  # Initialize batch size

            # Process batch or single image
            if batch_mode:
                # Determine batch size
                if is_tensor_batch:
                    batch_size = images.size(0)
                elif is_list_batch:  # is_list_batch
                    batch_size = len(images)

                if batch_size > 0:
                    # Process each image in the batch
                    for i in range(batch_size):
                        # Get the single image from the batch/list
                        single_image = images[i]
                        # Use internal try-except for single image processing to avoid breaking the whole batch
                        try:
                            face, prob = self._process_single_image(single_image)
                        except Exception as single_e:
                            print(f"Error processing image {i} in batch: {single_e}")
                            face, prob = None, 0.0  # Mark as failed

                        # Handle cases where face detection might fail for an image in the batch
                        if face is None:
                            # Create a dummy tensor for this image
                            face = torch.zeros(3, self.image_size, self.image_size, device=self.device)
                            prob = 0.0  # Ensure prob is a float

                        faces_list.append(face)
                        # Ensure prob is added as a float or tensor element compatible with torch.tensor later
                        probs_list.append(
                            float(prob)
                            if isinstance(prob, (int, float, np.number))
                            else prob.item()
                            if isinstance(prob, torch.Tensor) and prob.numel() == 1
                            else 0.0
                        )

                    # Stack results if faces were processed
                    if faces_list:
                        faces = torch.stack(faces_list)
                        # Ensure probs is a list of numbers before converting to tensor
                        probs = torch.tensor(probs_list, device=self.device, dtype=torch.float32)
                    else:  # Handle case where the input list/tensor was empty
                        faces = torch.zeros(0, 3, self.image_size, self.image_size, device=self.device)
                        probs = torch.zeros(0, device=self.device)
                else:  # Handle empty input list/tensor
                    faces = torch.zeros(0, 3, self.image_size, self.image_size, device=self.device)
                    probs = torch.zeros(0, device=self.device)

            else:  # Handle case with a single image (not in a list or tensor batch)
                batch_size = 1
                try:
                    face, prob = self._process_single_image(images)
                except Exception as single_e:
                    print(f"Error processing single image: {single_e}")
                    face, prob = None, 0.0

                if face is not None:
                    faces = face.unsqueeze(0)  # Create a batch of size 1
                    # Ensure prob is a number
                    prob_val = (
                        float(prob)
                        if isinstance(prob, (int, float, np.number))
                        else prob.item()
                        if isinstance(prob, torch.Tensor) and prob.numel() == 1
                        else 0.0
                    )
                    probs = torch.tensor([prob_val], device=self.device, dtype=torch.float32)
                else:
                    # Create dummy tensor if no face detected in the single image
                    faces = torch.zeros(1, 3, self.image_size, self.image_size, device=self.device)
                    probs = torch.zeros(1, device=self.device)

        except Exception as e:  # Catch errors in the overall batch logic
            print(f"Error during face detection batch processing: {e}")
            # Ensure consistent return shape on error, try to determine batch size if possible
            batch_size_on_error = 0  # Default to 0 for unknown/error state
            if isinstance(images, torch.Tensor) and images.ndim == 4:
                batch_size_on_error = images.size(0)
            elif isinstance(images, list):
                batch_size_on_error = len(images)  # len([]) is 0, which is correct

            faces = torch.zeros(batch_size_on_error, 3, self.image_size, self.image_size, device=self.device)
            probs = torch.zeros(batch_size_on_error, device=self.device)
            # Optionally re-raise the exception if debugging is needed
            # raise e

        # Return faces and optionally probabilities
        if return_probability:
            # Ensure probs is always a tensor even if errors occurred
            if not isinstance(probs, torch.Tensor):
                # This might happen if probs_list was somehow not converted in the batch case
                probs = torch.tensor(probs if isinstance(probs, list) else [probs], device=self.device, dtype=torch.float32)
            # Ensure faces is always a tensor
            if not isinstance(faces, torch.Tensor):
                # This case should ideally not happen with the error handling, but as a safeguard:
                faces = torch.zeros(
                    probs.shape[0] if probs.numel() > 0 else 0, 3, self.image_size, self.image_size, device=self.device
                )

            # Final check for shape consistency
            if faces.shape[0] != probs.shape[0]:
                print(
                    f"Warning: Mismatch between faces batch size ({faces.shape[0]}) and probs batch size ({probs.shape[0]}). Adjusting."
                )
                # Attempt to fix based on probs size, assuming probs is more reliable after list processing
                target_bs = probs.shape[0]
                # Fallback to zeros with the target batch size
                faces = torch.zeros(target_bs, 3, self.image_size, self.image_size, device=self.device)

            return faces, probs
        else:
            # Ensure faces is always a tensor
            if not isinstance(faces, torch.Tensor):
                # Fallback similar to above, determine batch size if possible
                bs = 1  # Default for single image case or unknown
                if isinstance(images, torch.Tensor) and images.ndim == 4:
                    bs = images.size(0)
                elif isinstance(images, list):
                    bs = len(images)
                faces = torch.zeros(bs, 3, self.image_size, self.image_size, device=self.device)

            return faces

    def _process_single_image(self, image):
        """
        Process a single image for face detection and alignment.

        Args:
            image: Single input image

        Returns:
            face: Processed face tensor
            prob: Detection confidence
        """
        # Convert image to format expected by YOLO
        if isinstance(image, torch.Tensor):
            # If tensor, convert to numpy for YOLO
            if image.dim() == 3:  # (C, H, W) or (H, W, C)
                # Make sure the values are valid by clamping
                image = torch.clamp(image, 0, 1)  # Ensure values are in [0, 1]

                # Check if the tensor is in (H, W, C) format
                if image.shape[2] == 3 and image.shape[0] != 3:  # HWC format
                    # Already in the format expected by numpy, just convert
                    img_np = (image.cpu().numpy() * 255).astype(np.uint8)
                # Check if tensor is in (C, H, W) format
                elif image.shape[0] == 3:  # CHW format
                    # Convert from CHW to HWC
                    img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                else:
                    # For other channel configurations, try to handle gracefully
                    print(f"Warning: Unexpected tensor shape: {image.shape}")
                    if image.shape[0] == 224 and image.shape[1] == 224 and image.shape[2] == 3:
                        # This is likely a [224, 224, 3] tensor in HWC format
                        img_np = (image.cpu().numpy() * 255).astype(np.uint8)
                    else:
                        # Try to convert to a valid image format
                        img_np = (image.cpu().numpy() * 255).astype(np.uint8)
                        # Ensure we have a 3D array
                        if img_np.ndim == 2:
                            img_np = np.stack([img_np] * 3, axis=-1)
            else:
                # For non-standard tensor dimensions, try to normalize and convert safely
                # Convert to numpy first
                img_np = image.cpu().numpy()
                # If the tensor has values outside [0, 1], normalize to [0, 255]
                if img_np.max() > 1.0:
                    img_np = img_np.astype(np.uint8)  # Assuming values are already in [0, 255]
                else:
                    img_np = (img_np * 255).astype(np.uint8)  # Convert from [0, 1] to [0, 255]
        elif isinstance(image, np.ndarray):
            # For numpy arrays, ensure values are in correct range
            if image.dtype == np.float32 or image.dtype == np.float64:
                if image.max() <= 1.0:
                    # Convert from [0, 1] to [0, 255]
                    img_np = (image * 255).astype(np.uint8)
                else:
                    # Already in [0, 255] range
                    img_np = image.astype(np.uint8)
            else:
                # Integer type, assume already in correct range
                img_np = image
        elif isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Ensure image has correct dimensions for YOLO
        if img_np.ndim == 2:  # Grayscale
            img_np = np.stack([img_np] * 3, axis=-1)  # Convert to RGB
        elif img_np.ndim == 3 and img_np.shape[-1] == 1:  # Single channel with explicit dimension
            img_np = np.concatenate([img_np] * 3, axis=-1)  # Convert to RGB
        elif img_np.ndim == 3 and img_np.shape[-1] != 3 and img_np.shape[-1] != 4:
            # Handle unusual channel configuration
            print(f"Warning: Unusual image shape {img_np.shape}, attempting to reshape")
            if img_np.shape[0] == 3 or img_np.shape[0] == 4:  # Possibly channel-first format
                img_np = np.transpose(img_np, (1, 2, 0))
                if img_np.shape[-1] == 4:  # RGBA
                    img_np = img_np[:, :, :3]  # Remove alpha channel
            else:
                raise ValueError(f"Cannot process image with shape {img_np.shape}")
        elif img_np.ndim == 3 and img_np.shape[-1] == 4:  # RGBA
            img_np = img_np[:, :, :3]  # Remove alpha channel

        # Ensure we have a valid image at this point
        if img_np.shape[0] == 0 or img_np.shape[1] == 0:
            print(f"Warning: Invalid image shape after processing: {img_np.shape}")
            return None, 0.0

        try:
            # Run YOLO detection
            results = self.yolo(img_np, verbose=False)

            # Extract face with highest confidence
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                if len(boxes) > 0:
                    # Get the box with highest confidence
                    confs = boxes.conf
                    best_idx = int(torch.argmax(confs))

                    # Get box coordinates (x1, y1, x2, y2)
                    box = boxes.xyxy[best_idx].cpu().numpy()
                    conf = float(confs[best_idx])

                    # Add margin to bounding box (similar to MTCNN)
                    x1, y1, x2, y2 = box
                    width, height = x2 - x1, y2 - y1
                    margin_x, margin_y = width * (self.margin / 100), height * (self.margin / 100)

                    x1 = max(0, x1 - margin_x)
                    y1 = max(0, y1 - margin_y)
                    x2 = min(img_np.shape[1], x2 + margin_x)
                    y2 = min(img_np.shape[0], y2 + margin_y)

                    # Crop and resize face
                    face = img_np[int(y1) : int(y2), int(x1) : int(x2)]

                    # Resize to target size
                    face_pil = Image.fromarray(face)
                    face_pil = face_pil.resize((self.image_size, self.image_size))
                    face_tensor = torch.tensor(np.array(face_pil), device=self.device).permute(2, 0, 1).float() / 255.0

                    return face_tensor, conf
        except Exception as e:
            print(f"Error during YOLO face detection: {e}")
            return None, 0.0

        # Return None if no face detected
        return None, 0.0


class ResNetFeatureExtractor(nn.Module):
    """
    ResNet-based feature extractor for facial emotion recognition.
    """

    def __init__(
        self,
        resnet_version: str = "resnet50",
        pretrained: bool = True,
        output_dim: int = 512,
    ):
        super().__init__()

        # Load pre-trained ResNet
        if resnet_version == "resnet18":
            self.resnet = models.resnet18(pretrained=pretrained)
            resnet_output_dim = 512
        elif resnet_version == "resnet34":
            self.resnet = models.resnet34(pretrained=pretrained)
            resnet_output_dim = 512
        elif resnet_version == "resnet50":
            self.resnet = models.resnet50(pretrained=pretrained)
            resnet_output_dim = 2048
        elif resnet_version == "resnet101":
            self.resnet = models.resnet101(pretrained=pretrained)
            resnet_output_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet version: {resnet_version}")

        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])

        # Add a new fully connected layer for output dimension adjustment
        self.fc = nn.Linear(resnet_output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from facial images.

        Args:
            x: Input facial images (batch_size, 3, height, width)

        Returns:
            features: Extracted features (batch_size, output_dim)
        """
        # Extract features using ResNet
        features = self.features(x)

        # Flatten features
        features = features.view(features.size(0), -1)

        # Apply final fully connected layer
        features = self.fc(features)

        return features


class FaceNetFeatureExtractor(nn.Module):
    """
    FaceNet (InceptionResNetV1) based feature extractor for facial emotion recognition.
    """

    def __init__(
        self,
        pretrained: str = "vggface2",  # Options: "vggface2", "casia-webface"
        output_dim: int = 512,
    ):
        super().__init__()

        # Load pre-trained FaceNet
        self.facenet = InceptionResnetV1(pretrained=pretrained)
        facenet_output_dim = 512

        # Add a projection layer if needed
        if output_dim != facenet_output_dim:
            self.projection = nn.Linear(facenet_output_dim, output_dim)
        else:
            self.projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from facial images.

        Args:
            x: Input facial images (batch_size, 3, height, width)

        Returns:
            features: Extracted features (batch_size, output_dim)
        """
        # Extract features using FaceNet
        features = self.facenet(x)

        # Apply projection if needed
        features = self.projection(features)

        return features


class VisualEncoder(nn.Module):
    """
    Complete visual encoder pipeline for ACMFT including face detection,
    alignment, and feature extraction.
    """

    def __init__(
        self,
        face_detector_type: str = "yolo",
        feature_extractor_type: str = "facenet",  # Options: "facenet", "resnet"
        image_size: int = 160,
        output_dim: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.face_detector_type = face_detector_type
        self.feature_extractor_type = feature_extractor_type
        self.output_dim = output_dim  # Store output_dim

        # Initialize face detector
        if face_detector_type == "yolo":
            self.face_detector = YOLOFaceDetector(image_size=image_size, device=device)
        else:
            raise ValueError(f"Unsupported face detector type: {face_detector_type}")

        # Initialize feature extractor
        if feature_extractor_type == "facenet":
            self.feature_extractor = FaceNetFeatureExtractor(output_dim=output_dim)
        elif feature_extractor_type == "resnet":
            self.feature_extractor = ResNetFeatureExtractor(resnet_version="resnet50", output_dim=output_dim)
        else:
            raise ValueError(f"Unsupported feature extractor type: {feature_extractor_type}")

    def forward(
        self,
        images: Union[torch.Tensor, np.ndarray, List[np.ndarray], List[Image.Image]],
        detect_faces: bool = True,
    ) -> torch.Tensor:
        """
        Process images through the complete visual pipeline.

        Args:
            images: Input images (batch of images)
            detect_faces: Whether to perform face detection (set to False if images
                          are already cropped faces)

        Returns:
            features: Visual features (batch_size, output_dim)
        """
        # Detect and align faces if needed
        if detect_faces:
            faces = self.face_detector(images)
        else:
            # Ensure input images are tensors if not detecting faces
            if isinstance(images, (np.ndarray, list)):
                # Basic conversion, assumes list of numpy arrays or PIL images
                # Needs refinement based on actual expected input format when detect_faces=False
                processed_images = []
                for img in images:
                    if isinstance(img, np.ndarray):
                        img_pil = Image.fromarray(img)
                    elif isinstance(img, Image.Image):
                        img_pil = img
                    else:
                        # Handle unexpected type or raise error
                        print(f"Warning: Unsupported image type {type(img)} when detect_faces=False")
                        continue  # Or handle appropriately
                    # Resize and convert to tensor (assuming 3 channels)
                    img_resized = img_pil.resize((self.face_detector.image_size, self.face_detector.image_size))
                    img_tensor = torch.tensor(np.array(img_resized), device=self.device).permute(2, 0, 1).float() / 255.0
                    processed_images.append(img_tensor)
                if processed_images:
                    faces = torch.stack(processed_images)
                else:
                    faces = torch.empty(0, 3, self.face_detector.image_size, self.face_detector.image_size, device=self.device)
            elif isinstance(images, torch.Tensor):
                faces = images  # Assume tensor is already correct shape/type
            else:
                raise TypeError(f"Unsupported input type {type(images)} when detect_faces=False")

        # Handle case where face detection returns None or an empty tensor
        if faces is None or faces.shape[0] == 0:
            # If batch_size is still 0 (e.g., empty list input), return shape (0, output_dim)
            # Otherwise, if faces.shape[0] was 0, use 0 for batch size.
            actual_batch_size = faces.shape[0] if faces is not None else 0

            # Create a dummy tensor with zeros, shape (0, output_dim)
            features = torch.zeros(actual_batch_size, self.output_dim, device=self.device)
            return features

        # Ensure faces tensor is on the correct device before feature extraction
        faces = faces.to(self.device)

        # Extract features from faces
        features = self.feature_extractor(faces)

        return features


# For backward compatibility and simpler imports
FaceEncoder = VisualEncoder
