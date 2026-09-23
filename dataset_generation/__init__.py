"""NumPy-only Stage 2 trajectory generation built on the Stage 1 solver."""

from .config import GenerationConfig
from .generate import generate_dataset
from .io import load_dataset, save_dataset, validate_dataset

__all__ = ["GenerationConfig", "generate_dataset", "load_dataset",
           "save_dataset", "validate_dataset"]
