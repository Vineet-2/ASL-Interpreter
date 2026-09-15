"""
Comprehensive Evaluation and Benchmark script for BiGRU Sign Recognition Model.
Computes Accuracy, Precision, Recall, F1-Score (Macro, Weighted, Micro),
Per-Class Metrics Breakdown, and Inference Latency.
"""
import os
import sys
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None

import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score, confusion_matrix

from src.models.bigru_model import BiGRUSignClassifier
from src.training.dataset import LandmarkSequenceDataset
from src.data.vocabulary import FULL_WORDS, CORE_WORDS, get_class_word
from src.config import BASE_DIR, MODELS_DIR, SEQUENCE_LENGTH, FEATURE_DIM, NUM_CLASSES


def evaluate_model(
    model_path: Optional[str] = None,
    test_npz: Optional[str] = None,
    batch_size: int = 32
) -> Dict:
    """
    Evaluates the trained BiGRU model on the held-out test split and prints full performance matrices.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(FULL_WORDS)
    test_file = test_npz or str(BASE_DIR / "extracted_landmarks" / "test_filtered.npz")

    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test data file '{test_file}' not found.")

    data = np.load(test_file, allow_pickle=True)
    raw_seqs = list(data["sequences"])
    raw_labels = list(data["labels"])
    
    samples, labels = [], []
    for s, l in zip(raw_seqs, raw_labels):
        if np.count_nonzero(s) > 0 and int(l) < num_classes:
            samples.append(s)
            labels.append(int(l))

    if len(samples) == 0:
        raise ValueError(f"No valid non-zero sequences found in test set '{test_file}'.")

    test_dataset = LandmarkSequenceDataset(samples, labels, target_seq_len=SEQUENCE_LENGTH, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load Model
    model = BiGRUSignClassifier(
        input_dim=FEATURE_DIM,
        hidden_dim=128,
        num_layers=2,
        num_classes=num_classes
    ).to(device)

    pth = model_path or str(BASE_DIR / "saved_models" / "bigru_best.pth")
    if not os.path.exists(pth):
        raise FileNotFoundError(f"Trained model checkpoint '{pth}' not found.")

    model.load_state_dict(torch.load(pth, map_location=device, weights_only=True))
    model.eval()

    # Benchmark Inference
    all_preds = []
    all_top5 = []
    all_targets = []
    latencies = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            t0 = time.perf_counter()
            logits, _ = model(x)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0 / len(y))

            probs = torch.softmax(logits, dim=-1)
            top1 = torch.argmax(probs, dim=-1).cpu().numpy()
            _, top5 = torch.topk(probs, k=min(5, num_classes), dim=-1)

            all_preds.extend(top1.tolist())
            all_top5.extend(top5.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())

    # Calculate Performance Matrices
    top1_acc = accuracy_score(all_targets, all_preds) * 100.0
    top5_correct = sum(target in top5_list for target, top5_list in zip(all_targets, all_top5))
    top5_acc = (top5_correct / len(all_targets)) * 100.0
    avg_latency = float(np.mean(latencies))

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted", zero_division=0)
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(all_targets, all_preds, average="micro", zero_division=0)

    # Per-Class Metrics
    target_names = [get_class_word(i, use_extension=True) for i in range(num_classes)]
    present_classes = sorted(list(set(all_targets) | set(all_preds)))
    present_names = [get_class_word(i, use_extension=True) for i in present_classes]
    
    report_dict = classification_report(
        all_targets,
        all_preds,
        labels=present_classes,
        target_names=present_names,
        output_dict=True,
        zero_division=0
    )
    
    report_str = classification_report(
        all_targets,
        all_preds,
        labels=present_classes,
        target_names=present_names,
        zero_division=0
    )

    print("\n" + "=" * 65)
    print("   ASL SIGN RECOGNITION AGENT: TEST PERFORMANCE METRICS   ")
    print("=" * 65)
    print(f"Total Test Samples Evaluated: {len(all_targets)}")
    print(f"Vocabulary Classes:           {num_classes}")
    print("-" * 65)
    print(f"Top-1 Accuracy:               {top1_acc:.2f}%")
    print(f"Top-5 Accuracy:               {top5_acc:.2f}%")
    print(f"Macro Precision:              {prec_macro * 100.0:.2f}%")
    print(f"Macro Recall:                 {rec_macro * 100.0:.2f}%")
    print(f"Macro F1-Score:               {f1_macro * 100.0:.2f}%")
    print(f"Weighted Precision:           {prec_weighted * 100.0:.2f}%")
    print(f"Weighted Recall:              {rec_weighted * 100.0:.2f}%")
    print(f"Weighted F1-Score:            {f1_weighted * 100.0:.2f}%")
    print(f"Average Inference Latency:    {avg_latency:.2f} ms / sample")
    print("=" * 65)
    print("\nPER-CLASS DETAILED BREAKDOWN:")
    print(report_str)
    print("=" * 65 + "\n")

    return {
        "accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "precision_macro": prec_macro * 100.0,
        "recall_macro": rec_macro * 100.0,
        "f1_macro": f1_macro * 100.0,
        "precision_weighted": prec_weighted * 100.0,
        "recall_weighted": rec_weighted * 100.0,
        "f1_weighted": f1_weighted * 100.0,
        "latency_ms": avg_latency,
        "sample_count": len(all_targets),
        "per_class_report": report_dict,
        "report_str": report_str
    }


if __name__ == "__main__":
    evaluate_model()
