"""
PyTorch Dataset and Data Augmentations for Landmark Sequences.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import random
from typing import List, Tuple, Optional


def _random_rotation_matrix(max_angle_rad: float = 0.20) -> np.ndarray:
    """Generate a random 3D rotation matrix with small angles (yaw, pitch, roll)."""
    angles = np.random.uniform(-max_angle_rad, max_angle_rad, size=3)
    ax, ay, az = angles

    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(ax), -np.sin(ax)],
        [0.0, np.sin(ax), np.cos(ax)]
    ], dtype=np.float32)

    ry = np.array([
        [np.cos(ay), 0.0, np.sin(ay)],
        [0.0, 1.0, 0.0],
        [-np.sin(ay), 0.0, np.cos(ay)]
    ], dtype=np.float32)

    rz = np.array([
        [np.cos(az), -np.sin(az), 0.0],
        [np.sin(az), np.cos(az), 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    return (rz @ ry @ rx).astype(np.float32)


from src.data.vocabulary import get_class_index, get_class_word


class LandmarkSequenceDataset(Dataset):
    """
    PyTorch Dataset for 144D landmark feature sequences with rich spatial, temporal,
    and class-targeted augmentations for ASL recognition.
    """
    def __init__(
        self,
        samples: List[np.ndarray],
        labels: List[int],
        target_seq_len: int = 30,
        augment: bool = False
    ):
        self.samples = samples
        self.labels = labels
        self.target_seq_len = target_seq_len
        self.augment = augment

        # Cache targeted class indices for custom augmentation policies
        self.target_classes = {
            "THANKYOU": get_class_index("THANKYOU"),
            "GOOD": get_class_index("GOOD"),
            "PLEASE": get_class_index("PLEASE"),
            "HAPPY": get_class_index("HAPPY"),
            "NOW": get_class_index("NOW"),
            "TODAY": get_class_index("TODAY"),
            "NEED": get_class_index("NEED"),
            "HOSPITAL": get_class_index("HOSPITAL"),
            "ME": get_class_index("ME")
        }

    def __len__(self) -> int:
        return len(self.samples)

    def _augment_spatial(self, seq: np.ndarray, label: int) -> np.ndarray:
        """
        Applies 3D rotation, scaling, translation, and joint jitter in 144D landmark space.
        Uses class-specific adaptive parameters to resolve known confusion boundaries.
        """
        aug_seq = seq.copy()
        T, D = aug_seq.shape

        is_thank_good = (label == self.target_classes["THANKYOU"] or label == self.target_classes["GOOD"])
        is_please_happy = (label == self.target_classes["PLEASE"] or label == self.target_classes["HAPPY"])
        is_need_please = (label == self.target_classes["NEED"] or label == self.target_classes["PLEASE"])
        is_now_today = (label == self.target_classes["NOW"] or label == self.target_classes["TODAY"])

        # 1. 3D Rotation applied consistently across the temporal window
        rot_prob = 0.85 if (is_thank_good or is_need_please) else 0.60
        max_angle = 0.28 if is_thank_good else 0.20

        if random.random() < rot_prob:
            rot_mat = _random_rotation_matrix(max_angle_rad=max_angle)
            for t in range(T):
                # Left hand (0..63 -> 21x3)
                lh = aug_seq[t, :63].reshape(21, 3)
                if np.max(np.abs(lh)) > 1e-4:
                    aug_seq[t, :63] = (lh @ rot_mat.T).flatten()
                # Right hand (63..126 -> 21x3)
                rh = aug_seq[t, 63:126].reshape(21, 3)
                if np.max(np.abs(rh)) > 1e-4:
                    aug_seq[t, 63:126] = (rh @ rot_mat.T).flatten()
                # Rotate 3D body relative vectors if present
                if D >= 144:
                    for offset in range(126, 144, 3):
                        v = aug_seq[t, offset:offset+3].reshape(1, 3)
                        if np.max(np.abs(v)) > 1e-4:
                            aug_seq[t, offset:offset+3] = (v @ rot_mat.T).flatten()

        # 2. Random spatial scaling (0.80 to 1.20 for boundary widening)
        scale_prob = 0.85 if (is_need_please or is_thank_good) else 0.70
        scale_min, scale_max = (0.80, 1.20) if is_need_please else (0.85, 1.15)

        if random.random() < scale_prob:
            scale = random.uniform(scale_min, scale_max)
            active_mask = np.max(np.abs(aug_seq[:, :126]), axis=1) > 1e-4
            aug_seq[active_mask, :126] *= scale

        # 3. Small translation shift on active hand and body anchors
        shift_prob = 0.75 if (is_please_happy or is_now_today) else 0.50
        if random.random() < shift_prob:
            shift = np.random.normal(0, 0.025 if is_please_happy else 0.018, size=(1, 3)).astype(np.float32)
            for t in range(T):
                lh = aug_seq[t, :63].reshape(21, 3)
                if np.max(np.abs(lh)) > 1e-4:
                    aug_seq[t, :63] = (lh + shift).flatten()
                rh = aug_seq[t, 63:126].reshape(21, 3)
                if np.max(np.abs(rh)) > 1e-4:
                    aug_seq[t, 63:126] = (rh + shift).flatten()
                if D >= 144:
                    # Apply small relative shift to wrist-to-body anchor vectors
                    shift_vec = shift.flatten()
                    if np.max(np.abs(lh)) > 1e-4:
                        aug_seq[t, 126:129] += shift_vec
                        aug_seq[t, 132:135] += shift_vec
                    if np.max(np.abs(rh)) > 1e-4:
                        aug_seq[t, 129:132] += shift_vec
                        aug_seq[t, 135:138] += shift_vec

        # 4. Gaussian joint jitter
        jitter_prob = 0.85 if (is_thank_good or is_need_please) else 0.70
        noise_std = 0.016 if is_thank_good else 0.012
        if random.random() < jitter_prob:
            active_mask = np.max(np.abs(aug_seq[:, :126]), axis=1) > 1e-4
            noise = np.random.normal(0, noise_std, aug_seq.shape).astype(np.float32)
            aug_seq[active_mask] += noise[active_mask]

        # 5. Random frame dropout (zeroing out random 5% of frames)
        if random.random() < 0.35:
            drop_count = max(1, int(T * 0.05))
            drop_idx = random.sample(range(T), min(drop_count, T))
            aug_seq[drop_idx] = 0.0

        return aug_seq

    def _augment_temporal_warp(self, seq: np.ndarray, label: int) -> np.ndarray:
        """
        Time-warps the sequence by ±20-30% speed variation using linear interpolation.
        """
        T = len(seq)
        if T < 4:
            return seq

        is_fast_slow = (label in (self.target_classes["NOW"], self.target_classes["TODAY"], self.target_classes["PLEASE"]))
        speed_range = (0.75, 1.25) if is_fast_slow else (0.80, 1.20)
        speed_factor = random.uniform(*speed_range)
        new_len = max(4, int(round(T * speed_factor)))

        orig_indices = np.linspace(0, T - 1, num=T)
        warped_indices = np.linspace(0, T - 1, num=new_len)

        warped_seq = np.zeros((new_len, seq.shape[1]), dtype=np.float32)
        for d in range(seq.shape[1]):
            warped_seq[:, d] = np.interp(warped_indices, orig_indices, seq[:, d])

        return warped_seq

    def _pad_or_sample(self, seq: np.ndarray) -> np.ndarray:
        """
        Resample or pad sequence to fixed target_seq_len.
        """
        num_frames = len(seq)
        if num_frames == self.target_seq_len:
            return seq
        elif num_frames > self.target_seq_len:
            # Linear temporal subsampling
            indices = np.linspace(0, num_frames - 1, self.target_seq_len, dtype=int)
            return seq[indices]
        else:
            # Zero padding in front
            pad_len = self.target_seq_len - num_frames
            padding = np.zeros((pad_len, seq.shape[1]), dtype=np.float32)
            return np.vstack([padding, seq])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        seq = self.samples[idx]
        label_y = self.labels[idx]
        
        if self.augment:
            if random.random() < 0.60:
                seq = self._augment_temporal_warp(seq, label_y)
            seq = self._augment_spatial(seq, label_y)
            
        seq = self._pad_or_sample(seq)
        tensor_x = torch.from_numpy(seq).float()
        
        return tensor_x, label_y
