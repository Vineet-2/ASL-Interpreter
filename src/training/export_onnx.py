"""
ONNX Model Exporter for BiGRU Sign Classifier.
"""
import torch
import numpy as np
import os
from pathlib import Path
import logging

try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

from src.models.bigru_model import BiGRUSignClassifier
from src.data.vocabulary import FULL_WORDS, CORE_WORDS
from src.config import BASE_DIR, MODELS_DIR, SEQUENCE_LENGTH, FEATURE_DIM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_to_onnx(
    pytorch_model_path: str,
    onnx_out_path: str,
    num_classes: int = len(FULL_WORDS)
):
    device = torch.device("cpu")
    model = BiGRUSignClassifier(
        input_dim=FEATURE_DIM,
        hidden_dim=128,
        num_layers=2,
        num_classes=num_classes
    ).to(device)

    if os.path.exists(pytorch_model_path):
        model.load_state_dict(torch.load(pytorch_model_path, map_location=device))
        logger.info(f"Loaded weights from {pytorch_model_path}")
    else:
        logger.warning(f"PyTorch model path {pytorch_model_path} not found. Exporting initialized architecture.")

    model.eval()

    dummy_input = torch.randn(1, SEQUENCE_LENGTH, FEATURE_DIM, requires_grad=False)
    
    Path(onnx_out_path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_out_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_landmarks"],
        output_names=["logits", "attention_weights"],
        dynamic_axes={
            "input_landmarks": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size"},
            "attention_weights": {0: "batch_size", 1: "seq_len"}
        }
    )
    logger.info(f"Successfully exported ONNX model to: {onnx_out_path}")

    # Verify ONNX runtime
    if HAS_ONNX:
        session = ort.InferenceSession(onnx_out_path)
        ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outs = session.run(None, ort_inputs)
        logger.info(f"ONNX Runtime verification succeeded! Output shape: {ort_outs[0].shape}")


if __name__ == "__main__":
    pth = BASE_DIR / "saved_models" / "bigru_best.pth"
    out_onnx = BASE_DIR / "saved_models" / "bigru_sign_model.onnx"
    export_to_onnx(str(pth), str(out_onnx))
