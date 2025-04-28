"""
Visualization utilities for ACMFT results.

This module provides functions for visualizing model training,
evaluation results, and modality weights.
"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def plot_confusion_matrix(
    conf_matrix: Union[np.ndarray, torch.Tensor],
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
):
    """
    Plot confusion matrix for model evaluation.

    Args:
        conf_matrix: Confusion matrix (n_classes, n_classes)
        class_names: List of class names
        normalize: Whether to normalize the confusion matrix
        title: Plot title
        cmap: Colormap
        figsize: Figure size
        save_path: Path to save the figure (if None, the figure is displayed)
    """
    # Convert to numpy if needed
    if isinstance(conf_matrix, torch.Tensor):
        conf_matrix = conf_matrix.cpu().numpy()

    # Normalize if requested
    if normalize:
        conf_matrix = conf_matrix.astype("float") / (conf_matrix.sum(axis=1, keepdims=True) + 1e-6)

    # Default class names if not provided
    if class_names is None:
        emotions = ["Neutral", "Happiness", "Sadness", "Anger", "Fear", "Surprise", "Disgust", "Contempt"]
        class_names = emotions[: conf_matrix.shape[0]]

    # Create figure
    plt.figure(figsize=figsize)

    # Plot confusion matrix
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0,
        vmax=1 if normalize else None,
        linewidths=0.5,
    )

    # Set labels
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)

    # Rotate x labels
    plt.xticks(rotation=45, ha="right")

    # Tight layout
    plt.tight_layout()

    # Save or display
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_training_history(
    train_loss: List[float],
    val_loss: List[float],
    val_acc: List[float],
    title: str = "Training History",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
):
    """
    Plot training history (loss and accuracy curves).

    Args:
        train_loss: List of training loss values
        val_loss: List of validation loss values
        val_acc: List of validation accuracy values
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (if None, the figure is displayed)
    """
    epochs = range(1, len(train_loss) + 1)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot loss
    ax1.plot(epochs, train_loss, "bo-", label="Training Loss")
    ax1.plot(epochs, val_loss, "ro-", label="Validation Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Plot accuracy
    ax2.plot(epochs, val_acc, "go-", label="Validation Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Set overall title
    fig.suptitle(title, fontsize=16)

    # Tight layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # Save or display
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_modality_weights(
    weights: Dict[str, List[float]],
    emotions: Optional[List[str]] = None,
    title: str = "Modality Weights",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
):
    """
    Plot modality weights from dynamic contextual gating.

    Args:
        weights: Dictionary with keys 'visual', 'audio', 'hr' and values as lists of weights
        emotions: List of emotion names (if None, generic names are used)
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (if None, the figure is displayed)
    """
    # Default emotion names if not provided
    if emotions is None:
        emotions = [f"Emotion {i + 1}" for i in range(len(next(iter(weights.values()))))]

    # Data preparation
    modalities = list(weights.keys())
    x = np.arange(len(emotions))
    width = 0.8 / len(modalities)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot bars for each modality
    for i, modality in enumerate(modalities):
        ax.bar(x + (i - len(modalities) / 2 + 0.5) * width, weights[modality], width, label=modality.capitalize())

    # Set labels and title
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Weight")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(emotions, rotation=45, ha="right")
    ax.legend()

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")

    # Set y-axis limits
    ax.set_ylim([0, 1])

    # Tight layout
    plt.tight_layout()

    # Save or display
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_attention_heatmap(
    attention_weights: torch.Tensor,
    title: str = "Attention Weights",
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
):
    """
    Plot attention weights as a heatmap.

    Args:
        attention_weights: Attention weights tensor (head, seq_len, seq_len)
        title: Plot title
        cmap: Colormap
        figsize: Figure size
        save_path: Path to save the figure (if None, the figure is displayed)
    """
    # Convert to numpy if needed
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()

    # Get number of attention heads
    n_heads = attention_weights.shape[0]

    # Create figure with subplots for each head
    fig, axs = plt.subplots(n_heads // 2 + n_heads % 2, min(2, n_heads), figsize=figsize, squeeze=False)

    # Flatten axs for easy iteration
    axs_flat = axs.flatten()

    # Plot each attention head
    for i in range(n_heads):
        if i < len(axs_flat):
            sns.heatmap(
                attention_weights[i],
                ax=axs_flat[i],
                cmap=cmap,
                vmin=0,
                vmax=1,
                square=True,
                cbar=True if i % 2 == 1 else False,
            )
            axs_flat[i].set_title(f"Head {i + 1}")
            axs_flat[i].set_xlabel("Target")
            axs_flat[i].set_ylabel("Source")

    # Hide empty subplots
    for i in range(n_heads, len(axs_flat)):
        axs_flat[i].axis("off")

    # Set overall title
    fig.suptitle(title, fontsize=16)

    # Tight layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save or display
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def visualize_cross_modal_fusion(
    visual_features: torch.Tensor,
    audio_features: torch.Tensor,
    hr_features: torch.Tensor,
    fused_features: torch.Tensor,
    method: str = "tsne",
    title: str = "Cross-Modal Feature Visualization",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
):
    """
    Visualize cross-modal fusion features in 2D.

    Args:
        visual_features: Visual features tensor (batch_size, hidden_dim)
        audio_features: Audio features tensor (batch_size, hidden_dim)
        hr_features: HR features tensor (batch_size, hidden_dim)
        fused_features: Fused features tensor (batch_size, hidden_dim)
        method: Dimensionality reduction method ('pca', 'tsne', 'umap')
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure (if None, the figure is displayed)
    """
    try:
        # Convert to numpy
        visual_features = visual_features.detach().cpu().numpy()
        audio_features = audio_features.detach().cpu().numpy()
        hr_features = hr_features.detach().cpu().numpy()
        fused_features = fused_features.detach().cpu().numpy()

        # Dimensionality reduction
        if method == "pca":
            from sklearn.decomposition import PCA

            reducer = PCA(n_components=2)
        elif method == "tsne":
            from sklearn.manifold import TSNE

            reducer = TSNE(n_components=2, perplexity=30, n_iter=1000)
        elif method == "umap":
            try:
                import umap

                reducer = umap.UMAP(n_components=2)
            except ImportError:
                print("UMAP not installed. Using t-SNE instead.")
                from sklearn.manifold import TSNE

                reducer = TSNE(n_components=2, perplexity=30, n_iter=1000)
        else:
            raise ValueError(f"Unsupported method: {method}")

        # Combine features for joint embedding
        all_features = np.vstack([visual_features, audio_features, hr_features, fused_features])

        # Apply dimensionality reduction
        embedded = reducer.fit_transform(all_features)

        # Split back into modalities
        batch_size = visual_features.shape[0]
        visual_embedded = embedded[:batch_size]
        audio_embedded = embedded[batch_size : 2 * batch_size]
        hr_embedded = embedded[2 * batch_size : 3 * batch_size]
        fused_embedded = embedded[3 * batch_size : 4 * batch_size]

        # Create figure
        plt.figure(figsize=figsize)

        # Plot each modality
        plt.scatter(visual_embedded[:, 0], visual_embedded[:, 1], c="b", label="Visual", alpha=0.7)
        plt.scatter(audio_embedded[:, 0], audio_embedded[:, 1], c="g", label="Audio", alpha=0.7)
        plt.scatter(hr_embedded[:, 0], hr_embedded[:, 1], c="r", label="HR", alpha=0.7)
        plt.scatter(fused_embedded[:, 0], fused_embedded[:, 1], c="purple", label="Fused", alpha=0.7)

        # Set labels and title
        plt.title(title)
        plt.xlabel(f"{method.upper()} Dimension 1")
        plt.ylabel(f"{method.upper()} Dimension 2")
        plt.legend()

        # Add grid
        plt.grid(True, linestyle="--", alpha=0.3)

        # Tight layout
        plt.tight_layout()

        # Save or display
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    except Exception as e:
        print(f"Error visualizing cross-modal fusion: {e}")
