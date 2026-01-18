from pathlib import Path
from typing import Dict

import torch


def save_checkpoint(state: Dict, checkpoint_dir: Path, filename: str) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / filename
    torch.save(state, path)
    return path
