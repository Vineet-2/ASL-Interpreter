"""
Training script for BiGRU + Attention Sign Recognition Classifier.
Features Stratified 5-Fold Cross-Validation, Regularization, Landmark Data Augmentations,
and automated ONNX export.
"""
import os
import sys
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple, List
from sklearn.model_selection import StratifiedKFold

from src.models.bigru_model import BiGRUSignClassifier
from src.training.dataset import LandmarkSequenceDataset
from src.training.export_onnx import export_to_onnx
from src.data.vocabulary import FULL_WORDS, CORE_WORDS
from src.config import BASE_DIR, MODELS_DIR, SEQUENCE_LENGTH, FEATURE_DIM, NUM_CLASSES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def delete_previous_models():
    """Deletes existing model weights to ensure clean retraining."""
    models_dir = Path(MODELS_DIR)
    targets = [
        models_dir / "bigru_best.pth",
        models_dir / "bigru_sign_model.pth",
        models_dir / "bigru_sign_model.onnx",
    ]
    for target in targets:
        if target.exists():
            try:
                target.unlink()
                logger.info(f"Deleted previous model file: {target}")
            except Exception as e:
                logger.warning(f"Could not delete {target}: {e}")


def load_dataset_samples(npz_path: Path, num_classes: int) -> Tuple[List[np.ndarray], List[int]]:
    """Loads and filters valid non-zero landmark sequences from an NPZ file."""
    if not npz_path.exists():
        return [], []
    try:
        data = np.load(npz_path, allow_pickle=True)
        raw_seqs = list(data["sequences"])
        raw_labels = list(data["labels"])
        samples, labels = [], []
        for s, l in zip(raw_seqs, raw_labels):
            if np.count_nonzero(s) > 0 and int(l) < num_classes:
                samples.append(s)
                labels.append(int(l))
        return samples, labels
    except Exception as e:
        logger.warning(f"Failed to load {npz_path}: {e}")
        return [], []


from src.data.vocabulary import FULL_WORDS, CORE_WORDS, get_class_index


def oversample_targeted_classes(
    samples: List[np.ndarray],
    labels: List[int],
    target_words: List[str] = ["THANKYOU", "GOOD", "PLEASE", "HAPPY", "NOW", "TODAY", "NEED"],
    factor: int = 2
) -> Tuple[List[np.ndarray], List[int]]:
    """Oversamples underrepresented / narrow-boundary classes in training split."""
    target_indices = {get_class_index(w, use_extension=True) for w in target_words if get_class_index(w, use_extension=True) != -1}
    out_samples = list(samples)
    out_labels = list(labels)
    for s, l in zip(samples, labels):
        if l in target_indices:
            for _ in range(factor - 1):
                out_samples.append(s)
                out_labels.append(l)
    return out_samples, out_labels


def train_single_fold(
    train_samples: List[np.ndarray],
    train_labels: List[int],
    val_samples: List[np.ndarray],
    val_labels: List[int],
    num_classes: int,
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 1e-3,
    dropout: float = 0.35,
    weight_decay: float = 1e-3,
    device: Optional[torch.device] = None,
    fold_idx: int = 1
) -> Tuple[nn.Module, float, Dict]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_s_aug, train_l_aug = oversample_targeted_classes(train_samples, train_labels, factor=2)
    train_dataset = LandmarkSequenceDataset(train_s_aug, train_l_aug, target_seq_len=SEQUENCE_LENGTH, augment=True)
    val_dataset = LandmarkSequenceDataset(val_samples, val_labels, target_seq_len=SEQUENCE_LENGTH, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = BiGRUSignClassifier(
        input_dim=FEATURE_DIM,
        hidden_dim=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(y)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += len(y)

        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                loss = criterion(logits, y)
                v_loss += loss.item() * len(y)
                preds = torch.argmax(logits, dim=1)
                v_correct += (preds == y).sum().item()
                v_total += len(y)

        val_loss = v_loss / max(1, v_total)
        val_acc = v_correct / max(1, v_total)
        scheduler.step()

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs:
            logger.info(
                f"[Fold {fold_idx}] Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.1f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}%"
            )

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, best_val_acc, {"best_val_acc": best_val_acc, "final_train_acc": train_acc}


def train_model(
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    n_splits: int = 5,
    save_path: Optional[str] = None
) -> Tuple[nn.Module, str, Dict]:
    """
    Main training routine:
    1. Deletes previous models.
    2. Loads real extracted training data.
    3. Runs Stratified 5-Fold Cross-Validation for checkpoint selection.
    4. Saves the best PyTorch model checkpoint.
    5. Exports to ONNX runtime format.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(FULL_WORDS)
    logger.info(f"Initiating Sign Model Training on {device} across {num_classes} ASL vocabulary classes.")

    # 1. Delete previous models
    delete_previous_models()

    train_npz = BASE_DIR / "extracted_landmarks" / "train_filtered.npz"
    train_samples, train_labels = load_dataset_samples(train_npz, num_classes)

    if len(train_samples) == 0:
        raise ValueError(
            f"No valid non-zero sequences found in '{train_npz}'. "
            "Please ensure extract_landmarks.py has completed successfully."
        )

    logger.info(f"Loaded {len(train_samples)} training samples across {num_classes} classes from {train_npz}")

    # Stratified K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accs = []
    best_overall_acc = 0.0
    best_overall_model = None

    samples_arr = np.array(train_samples, dtype=object)
    labels_arr = np.array(train_labels, dtype=np.int64)

    logger.info(f"Starting Stratified {n_splits}-Fold Cross-Validation...")
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(samples_arr, labels_arr), 1):
        logger.info(f"\n--- Running Fold {fold_idx}/{n_splits} ({len(train_idx)} train, {len(val_idx)} val samples) ---")
        f_train_s = [samples_arr[i] for i in train_idx]
        f_train_l = [int(labels_arr[i]) for i in train_idx]
        f_val_s = [samples_arr[i] for i in val_idx]
        f_val_l = [int(labels_arr[i]) for i in val_idx]

        model, fold_val_acc, _ = train_single_fold(
            train_samples=f_train_s,
            train_labels=f_train_l,
            val_samples=f_val_s,
            val_labels=f_val_l,
            num_classes=num_classes,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            fold_idx=fold_idx
        )
        fold_accs.append(fold_val_acc)
        logger.info(f"Fold {fold_idx} Best Validation Accuracy: {fold_val_acc*100:.2f}%")

        if fold_val_acc > best_overall_acc or best_overall_model is None:
            best_overall_acc = fold_val_acc
            best_overall_model = model

    mean_cv_acc = float(np.mean(fold_accs)) * 100.0
    std_cv_acc = float(np.std(fold_accs)) * 100.0
    logger.info(f"\n=======================================================")
    logger.info(f"Stratified {n_splits}-Fold CV Accuracy: {mean_cv_acc:.2f}% (+/- {std_cv_acc:.2f}%)")
    logger.info(f"Best Fold Validation Accuracy:    {best_overall_acc*100:.2f}%")
    logger.info(f"=======================================================\n")

    # Save best PyTorch checkpoint
    models_dir = Path(MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    out_pth = save_path or str(models_dir / "bigru_best.pth")
    torch.save(best_overall_model.state_dict(), out_pth)
    logger.info(f"Saved PyTorch model to: {out_pth}")

    # Export to ONNX
    out_onnx = str(models_dir / "bigru_sign_model.onnx")
    try:
        export_to_onnx(out_pth, out_onnx, num_classes=num_classes)
    except Exception as e:
        logger.warning(f"ONNX export warning: {e}")

    return best_overall_model, out_pth, {
        "mean_cv_accuracy": mean_cv_acc,
        "std_cv_accuracy": std_cv_acc,
        "best_fold_accuracy": best_overall_acc * 100.0,
        "fold_accuracies": [acc * 100.0 for acc in fold_accs]
    }


if __name__ == "__main__":
    train_model(epochs=35)
