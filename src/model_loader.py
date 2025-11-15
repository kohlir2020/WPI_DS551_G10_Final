"""Utility for loading pre-trained RL model checkpoints."""

from typing import Optional

import torch


def load_checkpoint(
    model_path: str, device: str = "cpu"
) -> Optional[torch.nn.Module]:
    """
    Load a PyTorch model from checkpoint.

    Args:
        model_path: Path to model checkpoint file
        device: Device to load model to ("cpu" or "cuda")

    Returns:
        Loaded model or None if loading fails
    """
    try:
        model = torch.load(model_path, map_location=device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None


def save_checkpoint(model: torch.nn.Module, save_path: str) -> None:
    """Save model checkpoint."""
    torch.save(model, save_path)
    print(f"Model saved to {save_path}")