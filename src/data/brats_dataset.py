from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class BratsNpyDataset(Dataset):
    def __init__(self, images_dir: Path, labels_dir: Path, eps: float = 1e-6) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.eps = eps
        self.image_paths = sorted(self.images_dir.glob("*.npy"))
        if not self.image_paths:
            raise FileNotFoundError(f"No .npy files found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _normalize(self, volume: np.ndarray) -> np.ndarray:
        normed = volume.astype(np.float32)
        for c in range(normed.shape[0]):
            channel = normed[c]
            mean = channel.mean()
            std = channel.std()
            normed[c] = (channel - mean) / (std + self.eps)
        return normed

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]
        label_path = self.labels_dir / image_path.name
        if not label_path.exists():
            raise FileNotFoundError(f"Label not found for {image_path.name}")

        image = np.load(image_path)
        label = np.load(label_path)
        if image.ndim != 4:
            raise ValueError(f"Expected image shape (C,D,H,W), got {image.shape}")
        if label.ndim != 3:
            raise ValueError(f"Expected label shape (D,H,W), got {label.shape}")

        image = self._normalize(image)
        return torch.from_numpy(image), torch.from_numpy(label).long()


class BratsNiftiDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split_file: Path,
        modalities: Sequence[str] = ("t1", "t1ce", "t2", "flair"),
        label_suffix: str = "seg",
        extensions: Sequence[str] = (".nii.gz", ".nii"),
        eps: float = 1e-6,
    ) -> None:
        try:
            import nibabel as nib  # type: ignore
        except ImportError as exc:
            raise ImportError("nibabel is required for NIfTI loading") from exc

        self._nib = nib
        self.data_root = Path(data_root)
        self.split_file = Path(split_file)
        self.modalities = list(modalities)
        self.label_suffix = label_suffix
        self.extensions = list(extensions)
        self.eps = eps
        self.case_ids = self._load_case_ids()
        if not self.case_ids:
            raise FileNotFoundError(f"No case IDs found in {self.split_file}")

    def _load_case_ids(self) -> List[str]:
        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")
        return [line.strip() for line in self.split_file.read_text().splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.case_ids)

    def _resolve_case_dir(self, case_entry: str) -> Tuple[Path, str]:
        entry_path = Path(case_entry)
        if entry_path.is_absolute():
            case_dir = entry_path
        else:
            if entry_path.parts and entry_path.parts[0] == self.data_root.name:
                entry_path = Path(*entry_path.parts[1:])
            case_dir = self.data_root / entry_path
            if not case_dir.exists():
                case_dir = self.data_root / entry_path.name
        case_id = case_dir.name
        return case_dir, case_id

    def _find_nifti(self, case_dir: Path, basename: str) -> Path:
        for ext in self.extensions:
            candidate = case_dir / f"{basename}{ext}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Missing file for {basename} with extensions {self.extensions}")

    def _load_nifti(self, path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        return self._nib.load(str(path)).get_fdata().astype(np.float32)

    def _normalize(self, volume: np.ndarray) -> np.ndarray:
        mean = volume.mean()
        std = volume.std()
        return (volume - mean) / (std + self.eps)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        case_entry = self.case_ids[idx]
        case_dir, case_id = self._resolve_case_dir(case_entry)
        channels = []
        for modality in self.modalities:
            basename = f"{case_id}_{modality}"
            path = self._find_nifti(case_dir, basename)
            channels.append(self._normalize(self._load_nifti(path)))
        image = np.stack(channels, axis=0)

        label_basename = f"{case_id}_{self.label_suffix}"
        label_path = self._find_nifti(case_dir, label_basename)
        label = self._load_nifti(label_path)
        if label.ndim != 3:
            raise ValueError(f"Expected label shape (D,H,W), got {label.shape}")

        return torch.from_numpy(image), torch.from_numpy(label).long()
