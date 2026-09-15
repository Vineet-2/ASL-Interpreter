"""
Generate and Export Overall and Class-Wise Confusion Matrix for ASL Sign Model.
"""
import os
import sys
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import shutil

from src.models.bigru_model import BiGRUSignClassifier
from src.training.dataset import LandmarkSequenceDataset
from src.data.vocabulary import FULL_WORDS, get_class_word
from src.config import BASE_DIR, MODELS_DIR, SEQUENCE_LENGTH, FEATURE_DIM


def compute_confusion_matrices():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(FULL_WORDS)
    test_file = BASE_DIR / "extracted_landmarks" / "test_filtered.npz"
    model_file = BASE_DIR / "saved_models" / "bigru_best.pth"

    data = np.load(test_file, allow_pickle=True)
    raw_seqs = list(data["sequences"])
    raw_labels = list(data["labels"])

    samples, labels = [], []
    for s, l in zip(raw_seqs, raw_labels):
        if np.count_nonzero(s) > 0 and int(l) < num_classes:
            samples.append(s)
            labels.append(int(l))

    test_dataset = LandmarkSequenceDataset(samples, labels, target_seq_len=SEQUENCE_LENGTH, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = BiGRUSignClassifier(
        input_dim=FEATURE_DIM,
        hidden_dim=128,
        num_layers=2,
        num_classes=num_classes
    ).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits, _ = model(x)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(y.numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    class_names = [get_class_word(i, use_extension=True) for i in range(num_classes)]

    # 1. Overall Confusion Matrix (59x59)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    out_cm_csv = BASE_DIR / "saved_models" / "confusion_matrix.csv"
    df_cm.to_csv(out_cm_csv)
    print(f"Saved Overall Confusion Matrix CSV to: {out_cm_csv}")

    # 2. Class-Wise One-vs-Rest Confusion Matrix Metrics
    total_samples = len(y_true)
    class_wise_records = []
    confusion_pairs = []

    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        fn = int(np.sum(cm[i, :]) - tp)
        fp = int(np.sum(cm[:, i]) - tp)
        tn = int(total_samples - (tp + fn + fp))
        
        support = tp + fn
        accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Find most common misclassifications for this class
        misclassified = []
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                misclassified.append(f"{class_names[j]} ({cm[i, j]})")
                confusion_pairs.append({
                    "True_Sign": name,
                    "Predicted_Sign": class_names[j],
                    "Count": int(cm[i, j])
                })

        class_wise_records.append({
            "Sign": name,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Support": support,
            "Accuracy": round(accuracy * 100.0, 2),
            "Precision": round(precision * 100.0, 2),
            "Recall": round(recall * 100.0, 2),
            "Specificity": round(specificity * 100.0, 2),
            "F1_Score": round(f1 * 100.0, 2),
            "Confused_With": ", ".join(misclassified) if misclassified else "None (100% Correct)"
        })

    df_class_wise = pd.DataFrame(class_wise_records)
    out_class_csv = BASE_DIR / "saved_models" / "class_wise_confusion_matrix.csv"
    df_class_wise.to_csv(out_class_csv, index=False)
    print(f"Saved Class-Wise Confusion Matrix CSV to: {out_class_csv}")

    # 3. Generate High-Res Confusion Matrix Heatmap Plot
    plt.figure(figsize=(24, 20), dpi=200)
    sns.set_theme(style="white")
    
    # Normalize by row (True Class) for percentage heatmap
    cm_norm = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-12)
    
    ax = sns.heatmap(
        cm_norm,
        annot=False,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Recall / Recognition Rate'}
    )
    plt.title(f"ASL 59-Sign BiGRU Recognition Model: Normalized Confusion Matrix (Test Acc: {accuracy_score(y_true, y_pred)*100:.2f}%)", fontsize=18, pad=20, weight="bold")
    plt.xlabel("Predicted Sign", fontsize=14, labelpad=12, weight="bold")
    plt.ylabel("True Sign (Ground Truth)", fontsize=14, labelpad=12, weight="bold")
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    out_png = BASE_DIR / "saved_models" / "confusion_matrix.png"
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"Saved Confusion Matrix Heatmap Plot to: {out_png}")

    # Copy to brain artifact directory if exists
    artifact_dir = Path(r"C:\Users\vinee\.gemini\antigravity-ide\brain\faf3152b-d5d4-4b4b-9d34-8801c6c9486c")
    if artifact_dir.exists():
        shutil.copy(out_png, artifact_dir / "confusion_matrix.png")
        print(f"Copied heatmap image to artifact directory: {artifact_dir / 'confusion_matrix.png'}")

    # Print top confusion pairs
    df_pairs = pd.DataFrame(confusion_pairs)
    if not df_pairs.empty:
        df_pairs = df_pairs.sort_values(by="Count", ascending=False)
        out_pairs_csv = BASE_DIR / "saved_models" / "top_confusions.csv"
        df_pairs.to_csv(out_pairs_csv, index=False)

    return df_cm, df_class_wise, df_pairs


if __name__ == "__main__":
    compute_confusion_matrices()
