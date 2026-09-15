"""
Sign Recognition Agent: PyTorch / ONNX BiGRU inference and sliding-window continuous segmentation.
"""
import torch
import numpy as np
import os
from typing import List, Optional, Tuple
import logging

try:
    import onnxruntime as ort
    HAS_ONNX = True
except Exception:
    HAS_ONNX = False

from src.agents.base_agent import BaseAgent
from src.models.schema import ConversationState, SignPrediction
from src.models.bigru_model import BiGRUSignClassifier
from src.data.vocabulary import get_class_word, FULL_WORDS, CORE_WORDS
from src.agents.geometric_classifier import classify_sequence as geometric_classify
from src.config import SEQUENCE_LENGTH, FEATURE_DIM, MODELS_DIR, NUM_CLASSES


def resolve_model_path(model_path: Optional[str] = None) -> Optional[str]:
    """Pick ONNX or PyTorch weights from MODELS_DIR when the caller does not pass a path."""
    if model_path and os.path.exists(model_path):
        return model_path
    candidates = [
        MODELS_DIR / "bigru_sign_model.onnx",
        MODELS_DIR / "bigru_best.pth",
        MODELS_DIR / "bigru_sign_model.pth",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None

logger = logging.getLogger(__name__)


class SignRecognitionAgent(BaseAgent):
    """
    Sign Recognition Agent: Classifies landmark sequences into ASL signs using BiGRU + Attention.
    Supports sliding-window continuous segmentation and confidence thresholding.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.50,
        window_size: int = SEQUENCE_LENGTH,
        step_size: int = 15,
        timeout: float = 0.5
    ):
        super().__init__(name="SignRecognitionAgent", timeout=timeout)
        self.confidence_threshold = confidence_threshold
        self.window_size = window_size
        self.step_size = step_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[BiGRUSignClassifier] = None
        self.onnx_session = None
        self.weights_loaded = False
        self.backend = "geometric"

        # Load weights if present (Orchestrator used to pass None, so nothing ever loaded)
        self._init_model(resolve_model_path(model_path))

    def _init_model(self, model_path: Optional[str]):
        if model_path and os.path.exists(model_path):
            if model_path.endswith(".onnx") and HAS_ONNX:
                try:
                    self.onnx_session = ort.InferenceSession(model_path)
                    self.weights_loaded = True
                    self.backend = "onnx"
                    logger.info(f"Loaded ONNX model from {model_path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load ONNX: {e}")
            
            # PyTorch checkpoint
            try:
                raw = torch.load(model_path, map_location=self.device)
                state_dict = raw
                if isinstance(raw, dict) and "model_state_dict" in raw:
                    state_dict = raw["model_state_dict"]
                elif isinstance(raw, dict) and "state_dict" in raw:
                    state_dict = raw["state_dict"]

                num_cls = len(FULL_WORDS)
                if isinstance(state_dict, dict):
                    for k in ["classifier.4.weight", "classifier.weight"]:
                        if k in state_dict:
                            num_cls = state_dict[k].shape[0]
                            break

                self.model = BiGRUSignClassifier(
                    input_dim=FEATURE_DIM,
                    hidden_dim=128,
                    num_layers=2,
                    num_classes=num_cls
                ).to(self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.weights_loaded = True
                self.backend = "pytorch"
                logger.info(f"Loaded PyTorch model ({num_cls} classes) from {model_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load PyTorch model checkpoint: {e}")

        # No trained weights: keep a random net only as last resort. Live inference
        # uses the geometric classifier so the UI does not silently drop every sign.
        self.model = BiGRUSignClassifier(
            input_dim=FEATURE_DIM,
            hidden_dim=128,
            num_layers=2,
            num_classes=len(FULL_WORDS)
        ).to(self.device)
        self.model.eval()
        self.weights_loaded = False
        self.backend = "geometric"
        logger.warning(
            "No trained BiGRU weights in saved_models/. Live recognition uses the geometric classifier."
        )

    def predict_window(self, window_landmarks: np.ndarray) -> Tuple[str, float]:
        """
        Classifies a single window of shape (SEQUENCE_LENGTH, FEATURE_DIM).
        Includes margin gating between top-1 and top-2 predictions to reject ambiguous transition poses.
        """
        if window_landmarks.ndim == 2:
            window_landmarks = np.expand_dims(window_landmarks, axis=0)  # (1, T, D)

        if window_landmarks.shape[-1] < FEATURE_DIM:
            pad_dim = FEATURE_DIM - window_landmarks.shape[-1]
            pad_shape = list(window_landmarks.shape)
            pad_shape[-1] = pad_dim
            window_landmarks = np.concatenate([window_landmarks, np.zeros(pad_shape, dtype=window_landmarks.dtype)], axis=-1)
            
        if self.onnx_session is not None:
            ort_inputs = {self.onnx_session.get_inputs()[0].name: window_landmarks.astype(np.float32)}
            ort_outs = self.onnx_session.run(None, ort_inputs)
            logits = ort_outs[0]  # (1, num_classes)
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
            top_indices = np.argsort(probs[0])[::-1]
            top_idx = int(top_indices[0])
            top_conf = float(probs[0, top_idx])
            second_conf = float(probs[0, top_indices[1]]) if len(top_indices) > 1 else 0.0
            
            # Margin gate: if the model is ambiguous between top 2 classes, reject
            if (top_conf - second_conf) < 0.12 and top_conf < 0.85:
                return "", 0.0
            return get_class_word(top_idx, use_extension=True), top_conf
            
        with torch.no_grad():
            tensor_x = torch.from_numpy(window_landmarks).float().to(self.device)
            out = self.model(tensor_x)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1)
            top_idx = top_indices[0, 0].item()
            top_conf = top_probs[0, 0].item()
            second_conf = top_probs[0, 1].item() if top_probs.shape[-1] > 1 else 0.0
            
            if (top_conf - second_conf) < 0.12 and top_conf < 0.85:
                return "", 0.0
            return get_class_word(top_idx), top_conf

    def segment_continuous_stream(self, landmark_stream: np.ndarray) -> List[SignPrediction]:
        """
        Sliding-window segmentation over a multi-frame continuous landmark stream.
        """
        total_frames = len(landmark_stream)
        if total_frames < self.window_size:
            # Pad to window size
            pad = np.zeros((self.window_size - total_frames, FEATURE_DIM), dtype=np.float32)
            padded_stream = np.vstack([pad, landmark_stream])
            sign, conf = self.predict_window(padded_stream)
            if sign and conf >= self.confidence_threshold:
                return [SignPrediction(sign=sign, confidence=conf, start_frame=0, end_frame=total_frames)]
            return []

        predictions: List[SignPrediction] = []
        last_sign = None

        for start in range(0, total_frames - self.window_size + 1, self.step_size):
            end = start + self.window_size
            window = landmark_stream[start:end]
            sign, conf = self.predict_window(window)
            
            if sign and conf >= self.confidence_threshold:
                # Deduplicate consecutive identical signs within short span
                if sign != last_sign or not predictions:
                    predictions.append(SignPrediction(
                        sign=sign,
                        confidence=round(conf, 4),
                        start_frame=start,
                        end_frame=end
                    ))
                    last_sign = sign

        return predictions

    def _hand_frame_count(self, landmarks_arr: np.ndarray) -> int:
        if landmarks_arr.ndim != 2:
            return 0
        return int(np.sum(np.max(np.abs(landmarks_arr), axis=1) > 1e-4))

    async def process(self, state: ConversationState) -> ConversationState:
        """
        Processes state.raw_landmarks or existing sequence to generate recognized signs.
        """
        state.metadata["sign_backend"] = self.backend
        if not state.raw_landmarks:
            return state

        landmarks_arr = np.array(state.raw_landmarks, dtype=np.float32)
        if landmarks_arr.ndim != 2:
            return state

        # Require a steady hand presence (at least 8 frames) before classifying
        if self._hand_frame_count(landmarks_arr) < 8:
            return state

        raw_landmarks_arr = None
        if "raw_landmark_window" in state.metadata and state.metadata["raw_landmark_window"]:
            raw_landmarks_arr = np.array(state.metadata["raw_landmark_window"], dtype=np.float32)

        preds: List[SignPrediction] = []
        if self.weights_loaded:
            if landmarks_arr.shape[0] >= self.window_size:
                preds = self.segment_continuous_stream(landmarks_arr)
            else:
                sign, conf = self.predict_window(landmarks_arr)
                if sign and conf >= self.confidence_threshold:
                    preds = [
                        SignPrediction(
                            sign=sign,
                            confidence=round(conf, 4),
                            start_frame=0,
                            end_frame=len(landmarks_arr),
                        )
                    ]
            # If the neural net is unsure, try geometry rather than dropping the turn
            if not preds:
                preds = geometric_classify(
                    landmarks_arr,
                    raw_landmark_stream=raw_landmarks_arr,
                    confidence_threshold=self.confidence_threshold,
                )
                if preds:
                    state.metadata["sign_backend"] = "geometric_fallback"
        else:
            preds = geometric_classify(
                landmarks_arr,
                raw_landmark_stream=raw_landmarks_arr,
                confidence_threshold=self.confidence_threshold,
            )

        if preds:
            state.recognized_signs = preds
        return state
