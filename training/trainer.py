"""
Trainer implementation for ACMFT model.

This module provides a training framework for the ACMFT model, including
training loops, validation, testing, and model saving/loading.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import TrainingConfig
from models.acmft import ACMFT
from training.losses import ACMFTLoss


class EarlyStopping:
    """
    Early stopping handler to monitor validation metrics and stop training
    if no improvement is seen for a specified number of epochs.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "min",
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change in monitored value to qualify as improvement
            mode: 'min' if lower metric is better, 'max' if higher metric is better
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.early_stop = False

    def __call__(self, metric: float) -> Tuple[bool, bool]:
        """
        Check if training should be stopped based on validation metric.

        Args:
            metric: Current validation metric

        Returns:
            is_improved: Whether the metric improved or not
            should_stop: Whether training should be stopped
        """
        if self.mode == "min":
            score = -metric
            is_improved = score > self.best_score + self.min_delta
        else:
            score = metric
            is_improved = score > self.best_score + self.min_delta

        if is_improved:
            self.best_score = score
            self.counter = 0
            return True, False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return False, True
            return False, False


class ACMFTTrainer:
    """
    Trainer for the ACMFT model.
    """

    def __init__(
        self,
        model: ACMFT,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[TrainingConfig] = None,
        class_weights: Optional[torch.Tensor] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        experiment_name: str = "acmft_experiment",
        checkpoint_dir: str = "./checkpoints",
        log_dir: str = "./logs",
    ):
        """
        Initialize ACMFT trainer.

        Args:
            model: ACMFT model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            config: Training configuration
            class_weights: Optional tensor of class weights for imbalanced data
            device: Device to use for training
            experiment_name: Name of the experiment for logging
            checkpoint_dir: Directory to save model checkpoints
            log_dir: Directory for tensorboard logs
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()
        self.device = device
        self.experiment_name = experiment_name

        # Create directories if not exist
        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Tensorboard writer
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Move model to device
        self.model = self.model.to(self.device)

        # Loss function
        self.criterion = ACMFTLoss(
            num_emotions=self.model.config.num_emotions,
            label_smoothing=0.1,
            class_weights=class_weights.to(device) if class_weights is not None else None,
            use_reconstruction_loss=True,
            reconstruction_weight=0.1,
            l2_reg_weight=self.config.weight_decay,
        )

        # Optimizer
        if self.config.optimizer == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=0,  # Handled in loss
            )
        elif self.config.optimizer == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=0,  # Handled in loss
            )
        elif self.config.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=0,  # Handled in loss
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

        # Learning rate scheduler
        if self.config.lr_scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.num_epochs, **self.config.lr_scheduler_params
            )
        elif self.config.lr_scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, **self.config.lr_scheduler_params)
        elif self.config.lr_scheduler == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **self.config.lr_scheduler_params)
        else:
            self.scheduler = None

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
            mode="min",  # Lower validation loss is better
        )

        # Scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.config.use_amp else None

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []

    def _process_batch(
        self,
        batch: Dict[str, torch.Tensor],
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Process a batch of data and compute model outputs.

        Args:
            batch: Batch of data with modalities and labels
            return_features: Whether to return intermediate features

        Returns:
            logits: Model output logits
            targets: Ground truth labels
            features: Optional dictionary of modality features
        """
        # Extract data and move to device
        visual = batch.get("visual")
        audio = batch.get("audio")
        hr = batch.get("hr")
        targets = batch["labels"].to(self.device)
        
        # Log batch information for debugging
        print("\n===== Batch Info =====")
        print(f"Target shape: {targets.shape}")
        
        if visual is not None:
            if isinstance(visual, torch.Tensor):
                print(f"Visual: tensor with shape {visual.shape}")
            elif isinstance(visual, list):
                print(f"Visual: list with {len(visual)} items")
                if len(visual) > 0 and isinstance(visual[0], torch.Tensor):
                    print(f"  First visual tensor shape: {visual[0].shape}")
            else:
                print(f"Visual: {type(visual)}")
        else:
            print("Visual: None")
            
        if audio is not None:
            if isinstance(audio, torch.Tensor):
                print(f"Audio: tensor with shape {audio.shape}")
            elif isinstance(audio, list):
                print(f"Audio: list with {len(audio)} items")
                if len(audio) > 0 and isinstance(audio[0], torch.Tensor):
                    print(f"  First audio tensor shape: {audio[0].shape}")
            else:
                print(f"Audio: {type(audio)}")
        else:
            print("Audio: None")
            
        if hr is not None:
            if isinstance(hr, torch.Tensor):
                print(f"HR: tensor with shape {hr.shape}")
            elif isinstance(hr, list):
                print(f"HR: list with {len(hr)} items")
                if len(hr) > 0 and isinstance(hr[0], torch.Tensor):
                    print(f"  First HR tensor shape: {hr[0].shape}")
            else:
                print(f"HR: {type(hr)}")
        else:
            print("HR: None")
        
        # Check for suspicious batch sizes - early detection of problematic batches
        if targets.size(0) == 0:
            print("WARNING: Found empty targets batch!")
            import sys
            sys.exit("EARLY EXIT: Empty target batch detected")
            
        if visual is not None and isinstance(visual, torch.Tensor) and visual.size(0) == 0:
            print("WARNING: Found empty visual tensor!")
            import sys
            sys.exit("EARLY EXIT: Empty visual tensor detected")
            
        if audio is not None and isinstance(audio, torch.Tensor) and audio.size(0) == 0:
            print("WARNING: Found empty audio tensor!")
            import sys
            sys.exit("EARLY EXIT: Empty audio tensor detected")
            
        if hr is not None and isinstance(hr, torch.Tensor) and hr.size(0) == 0:
            print("WARNING: Found empty HR tensor!")
            import sys
            sys.exit("EARLY EXIT: Empty HR tensor detected")
        
        # Move modality data to device if present
        if visual is not None:
            if isinstance(visual, list):
                # Handle the case when the list contains numpy arrays
                processed_visual = []
                for v in visual:
                    if torch.is_tensor(v):
                        processed_visual.append(v.to(self.device))
                    elif isinstance(v, np.ndarray):
                        # Convert numpy array to tensor and move to device
                        tensor_v = torch.from_numpy(v.copy()).float().to(self.device)
                        processed_visual.append(tensor_v)
                visual = processed_visual
            elif torch.is_tensor(visual):
                visual = visual.to(self.device)
            elif isinstance(visual, np.ndarray):
                visual = torch.from_numpy(visual.copy()).float().to(self.device)

        if audio is not None:
            if isinstance(audio, list):
                # Handle the case when the list contains numpy arrays
                processed_audio = []
                for a in audio:
                    if torch.is_tensor(a):
                        processed_audio.append(a.to(self.device))
                    elif isinstance(a, np.ndarray):
                        # Convert numpy array to tensor and move to device
                        tensor_a = torch.from_numpy(a.copy()).float().to(self.device)
                        processed_audio.append(tensor_a)
                audio = processed_audio
            elif torch.is_tensor(audio):
                audio = audio.to(self.device)
            elif isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio.copy()).float().to(self.device)

        if hr is not None:
            if isinstance(hr, list):
                # Handle the case when the list contains numpy arrays
                processed_hr = []
                for h in hr:
                    if torch.is_tensor(h):
                        processed_hr.append(h.to(self.device))
                    elif isinstance(h, np.ndarray):
                        # Convert numpy array to tensor and move to device
                        tensor_h = torch.from_numpy(h.copy()).float().to(self.device)
                        processed_hr.append(tensor_h)
                hr = processed_hr
            elif torch.is_tensor(hr):
                hr = hr.to(self.device)
            elif isinstance(hr, np.ndarray):
                hr = torch.from_numpy(hr.copy()).float().to(self.device)

        # Forward pass
        if return_features:
            # Check if we have an empty batch (batch_size = 0)
            if (
                (visual is not None and hasattr(visual, "size") and visual.size(0) == 0)
                or (audio is not None and hasattr(audio, "size") and audio.size(0) == 0)
                or (hr is not None and hasattr(hr, "size") and hr.size(0) == 0)
                or targets.size(0) == 0
            ):
                # Return empty tensors that match expected output format
                empty_logits = torch.zeros(0, self.model.config.num_emotions, device=self.device)
                empty_features = {
                    "visual_weight": torch.zeros(0, device=self.device),
                    "audio_weight": torch.zeros(0, device=self.device),
                    "hr_weight": torch.zeros(0, device=self.device),
                }
                return empty_logits, targets, empty_features

            output, weights = self.model(visual, audio, hr, return_weights=True)

            # Return logits, targets, and features
            return output, targets, weights
        else:
            # Check if we have an empty batch (batch_size = 0)
            if (
                (visual is not None and hasattr(visual, "size") and visual.size(0) == 0)
                or (audio is not None and hasattr(audio, "size") and audio.size(0) == 0)
                or (hr is not None and hasattr(hr, "size") and hr.size(0) == 0)
                or targets.size(0) == 0
            ):
                # Return empty tensors that match expected output format
                empty_logits = torch.zeros(0, self.model.config.num_emotions, device=self.device)
                return empty_logits, targets, None

            output = self.model(visual, audio, hr)

            # Return logits and targets
            return output, targets, None

    def _train_epoch(self) -> Dict[str, float]:
        """
        Train the model for one epoch.

        Returns:
            metrics: Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.config.num_epochs}")

        # Freeze/unfreeze encoders based on configuration
        if self.config.freeze_encoders and self.current_epoch < self.config.unfreeze_after:
            if self.model.visual_encoder is not None:
                for param in self.model.visual_encoder.parameters():
                    param.requires_grad = False
            if self.model.audio_encoder is not None:
                for param in self.model.audio_encoder.parameters():
                    param.requires_grad = False
            if self.model.hr_encoder is not None:
                for param in self.model.hr_encoder.parameters():
                    param.requires_grad = False
        elif self.config.freeze_encoders and self.current_epoch == self.config.unfreeze_after:
            # Unfreeze encoders
            if self.model.visual_encoder is not None:
                for param in self.model.visual_encoder.parameters():
                    param.requires_grad = True
            if self.model.audio_encoder is not None:
                for param in self.model.audio_encoder.parameters():
                    param.requires_grad = True
            if self.model.hr_encoder is not None:
                for param in self.model.hr_encoder.parameters():
                    param.requires_grad = True

        # Training loop
        for batch in pbar:
            self.optimizer.zero_grad()

            # Mixed precision training
            if self.config.use_amp:
                with torch.cuda.amp.autocast():
                    logits, targets, features = self._process_batch(batch, return_features=True)
                    loss, loss_dict = self.criterion(
                        logits,
                        targets,
                        self.model,
                        fused_representation=features.get("fused") if features else None,
                        original_visual=features.get("visual") if features else None,
                        original_audio=features.get("audio") if features else None,
                        original_hr=features.get("hr") if features else None,
                    )

                # Skip backward pass if batch is empty
                if logits.size(0) == 0 or targets.size(0) == 0:
                    # Skip this batch without updating gradients
                    continue

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                logits, targets, features = self._process_batch(batch, return_features=True)
                loss, loss_dict = self.criterion(
                    logits,
                    targets,
                    self.model,
                    fused_representation=features.get("fused") if features else None,
                    original_visual=features.get("visual") if features else None,
                    original_audio=features.get("audio") if features else None,
                    original_hr=features.get("hr") if features else None,
                )

                # Skip backward pass if batch is empty
                if logits.size(0) == 0 or targets.size(0) == 0:
                    # Skip this batch without updating gradients
                    continue

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.config.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)

                self.optimizer.step()

            # Compute accuracy
            preds = torch.argmax(logits, dim=1)
            correct = (preds == targets).sum().item()

            # Update metrics
            total_loss += loss.item() * targets.size(0)
            total_correct += correct
            total_samples += targets.size(0)

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": loss.item(),
                    "acc": correct / targets.size(0),
                }
            )

        # Compute epoch metrics
        if total_samples == 0:
            # Handle case where all batches were empty
            epoch_loss = 0.0
            epoch_acc = 0.0
        else:
            epoch_loss = total_loss / total_samples
            epoch_acc = total_correct / total_samples

        # Update learning rate
        if self.scheduler is not None and self.config.lr_scheduler != "plateau":
            self.scheduler.step()

        # Return metrics
        return {
            "loss": epoch_loss,
            "acc": epoch_acc,
        }

    def _validate_epoch(self) -> Dict[str, float]:
        """
        Validate the model on the validation set.

        Returns:
            metrics: Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_targets = []

        # Validation loop
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Forward pass
                logits, targets, _ = self._process_batch(batch)

                # Compute loss
                loss, _ = self.criterion(logits, targets, self.model)

                # Compute accuracy
                if logits.size(0) == 0 or targets.size(0) == 0:
                    # Skip empty batches
                    continue

                preds = torch.argmax(logits, dim=1)
                correct = (preds == targets).sum().item()

                # Store predictions and targets for metrics
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

                # Update metrics
                total_loss += loss.item() * targets.size(0)
                total_correct += correct
                total_samples += targets.size(0)

        # Compute epoch metrics
        if total_samples == 0:
            # Handle case where all batches were empty
            epoch_loss = 0.0
            epoch_acc = 0.0
            class_accuracy = {c: 0.0 for c in range(self.model.config.num_emotions)}
            return {
                "loss": epoch_loss,
                "acc": epoch_acc,
                "class_accuracy": class_accuracy,
            }

        epoch_loss = total_loss / total_samples
        epoch_acc = total_correct / total_samples

        # Combine predictions and targets
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        # Compute per-class accuracy
        class_correct = {}
        class_total = {}
        for c in range(self.model.config.num_emotions):
            class_mask = all_targets == c
            class_correct[c] = (all_preds[class_mask] == c).sum().item()
            class_total[c] = class_mask.sum().item()

        class_accuracy = {
            c: class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0 for c in range(self.model.config.num_emotions)
        }

        # Return metrics
        return {
            "loss": epoch_loss,
            "acc": epoch_acc,
            "class_accuracy": class_accuracy,
        }

    def train(self) -> Dict[str, List[float]]:
        """
        Train the model for the specified number of epochs.

        Returns:
            history: Dictionary of training and validation metrics history
        """
        print(f"Starting training for {self.config.num_epochs} epochs...")

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch

            # Train epoch
            start_time = time.time()
            train_metrics = self._train_epoch()
            train_time = time.time() - start_time

            # Validate epoch
            val_metrics = self._validate_epoch()

            # Update history
            self.train_losses.append(train_metrics["loss"])
            self.val_losses.append(val_metrics["loss"])
            self.val_accs.append(val_metrics["acc"])

            # Print metrics
            print(
                f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                f"Time: {train_time:.2f}s - "
                f"Train Loss: {train_metrics['loss']:.4f} - "
                f"Train Acc: {train_metrics['acc']:.4f} - "
                f"Val Loss: {val_metrics['loss']:.4f} - "
                f"Val Acc: {val_metrics['acc']:.4f}"
            )

            # Log to tensorboard
            self.writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
            self.writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            self.writer.add_scalar("Accuracy/train", train_metrics["acc"], epoch)
            self.writer.add_scalar("Accuracy/val", val_metrics["acc"], epoch)

            # Log per-class accuracy
            for cls, acc in val_metrics["class_accuracy"].items():
                self.writer.add_scalar(f"Accuracy/val_class_{cls}", acc, epoch)

            # Log learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("LearningRate", current_lr, epoch)

            # Update scheduler if using plateau
            if self.scheduler is not None and self.config.lr_scheduler == "plateau":
                self.scheduler.step(val_metrics["loss"])

            # Check for improvement and save checkpoint
            improved, should_stop = self.early_stopping(val_metrics["loss"])

            if improved:
                self.best_val_loss = val_metrics["loss"]
                self.best_val_acc = val_metrics["acc"]

                # Save best model
                self.save_checkpoint("best_model.pt")
                print(f"Saved best model with val loss: {self.best_val_loss:.4f} and val acc: {self.best_val_acc:.4f}")

            # Save regular checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

            # Check early stopping
            if should_stop:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # Close tensorboard writer
        self.writer.close()

        # Return training history
        return {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
            "val_acc": self.val_accs,
        }

    def save_checkpoint(self, filename: str) -> None:
        """
        Save model checkpoint.

        Args:
            filename: Name of the checkpoint file
        """
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accs": self.val_accs,
            "model_config": self.model.config.__dict__,
            "training_config": self.config.__dict__,
        }

        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state if available
        if checkpoint["scheduler_state_dict"] is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load training state
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_val_acc = checkpoint["best_val_acc"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.val_accs = checkpoint["val_accs"]

        print(
            f"Loaded checkpoint from epoch {self.current_epoch + 1} with "
            f"val loss: {self.best_val_loss:.4f} and val acc: {self.best_val_acc:.4f}"
        )

    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Test the model on a test set.

        Args:
            test_loader: DataLoader for test data

        Returns:
            metrics: Dictionary of test metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_targets = []

        # Test loop
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                # Forward pass
                logits, targets, _ = self._process_batch(batch)

                # Compute loss
                loss, _ = self.criterion(logits, targets, self.model)

                # Compute accuracy
                preds = torch.argmax(logits, dim=1)
                correct = (preds == targets).sum().item()

                # Store predictions and targets for metrics
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

                # Update metrics
                total_loss += loss.item() * targets.size(0)
                total_correct += correct
                total_samples += targets.size(0)

        # Compute metrics
        test_loss = total_loss / total_samples
        test_acc = total_correct / total_samples

        # Combine predictions and targets
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        # Compute confusion matrix
        confusion_matrix = torch.zeros(self.model.config.num_emotions, self.model.config.num_emotions, dtype=torch.long)
        for t, p in zip(all_targets, all_preds):
            confusion_matrix[t, p] += 1

        # Compute per-class metrics
        class_metrics = []
        for c in range(self.model.config.num_emotions):
            # True positives, false positives, false negatives
            tp = confusion_matrix[c, c].item()
            fp = confusion_matrix[:, c].sum().item() - tp
            fn = confusion_matrix[c, :].sum().item() - tp

            # Precision, recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            class_metrics.append(
                {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )

        # Compute macro-averaged metrics
        macro_precision = sum(m["precision"] for m in class_metrics) / len(class_metrics)
        macro_recall = sum(m["recall"] for m in class_metrics) / len(class_metrics)
        macro_f1 = sum(m["f1"] for m in class_metrics) / len(class_metrics)

        # Weighted F1 score (weighted by class frequency)
        class_weights = torch.bincount(all_targets) / len(all_targets)
        weighted_f1 = sum(class_metrics[c]["f1"] * class_weights[c].item() for c in range(self.model.config.num_emotions))

        # Return metrics
        return {
            "loss": test_loss,
            "accuracy": test_acc,
            "confusion_matrix": confusion_matrix,
            "class_metrics": class_metrics,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }
