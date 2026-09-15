"""
Vision Agent: Landmark extraction, normalization, and rolling temporal buffer using MediaPipe.
"""
import numpy as np
import cv2
from typing import List, Optional, Tuple, Deque
from collections import deque
import logging

import sys
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None

try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_holistic = mp.solutions.holistic
    HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False

from src.agents.base_agent import BaseAgent
from src.models.schema import ConversationState, AgentState
from src.config import SEQUENCE_LENGTH, FEATURE_DIM

logger = logging.getLogger(__name__)


class VisionAgent(BaseAgent):
    """
    Vision Agent: extracts 144-dimensional landmark vectors (126 hand landmarks + 18 body-relative anchors)
    from video frames and maintains a temporal rolling buffer of landmarks.
    """
    def __init__(self, buffer_size: int = SEQUENCE_LENGTH, timeout: float = 0.5):
        super().__init__(name="VisionAgent", timeout=timeout)
        self.buffer_size = buffer_size
        self.buffer: Deque[np.ndarray] = deque(maxlen=buffer_size)
        self.raw_buffer: Deque[np.ndarray] = deque(maxlen=buffer_size)
        
        self.prev_landmarks: Optional[np.ndarray] = None
        self.smoothing_enabled: bool = True
        self.hands = None
        self.pose = None
        self.holistic = None
        if HAS_MEDIAPIPE:
            try:
                self.hands = mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.2,
                    min_tracking_confidence=0.2
                )
            except Exception as e:
                logger.warning(f"MediaPipe Hands init warning: {e}")
            try:
                self.pose = mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=0,
                    min_detection_confidence=0.2,
                    min_tracking_confidence=0.2
                )
            except Exception as e:
                logger.warning(f"MediaPipe Pose init warning: {e}")
            try:
                self.holistic = mp_holistic.Holistic(
                    static_image_mode=False,
                    model_complexity=0,
                    smooth_landmarks=True,
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3
                )
            except Exception as e:
                logger.warning(f"MediaPipe Holistic init warning: {e}")

    def smooth_features(self, current_feat: np.ndarray) -> np.ndarray:
        """
        Applies adaptive Exponential Moving Average (EMA) to eliminate high-frequency
        tracking jitter while remaining responsive during rapid hand movements.
        """
        if not self.smoothing_enabled or self.prev_landmarks is None:
            self.prev_landmarks = current_feat.copy()
            return current_feat

        curr_present = np.any(np.abs(current_feat) > 1e-4)
        prev_present = np.any(np.abs(self.prev_landmarks) > 1e-4)

        if not curr_present or not prev_present:
            self.prev_landmarks = current_feat.copy()
            return current_feat

        # Adaptive alpha based on landmark displacement velocity
        diff = np.linalg.norm(current_feat - self.prev_landmarks)
        if diff < 0.10:
            # Subtle jitter / stationary hold: smooth heavily
            alpha = 0.35
        elif diff < 0.30:
            alpha = 0.60
        else:
            # Rapid movement: prioritize responsiveness
            alpha = 0.85

        smoothed = alpha * current_feat + (1.0 - alpha) * self.prev_landmarks
        self.prev_landmarks = smoothed.copy()
        return smoothed.astype(np.float32)

    def extract_landmarks_from_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Extract 144-dim feature vector from an RGB/BGR frame:
        - 126 hand landmarks (21 left * 3 + 21 right * 3, wrist-normalized & scale-normalized)
        - 18 body-relative anchor features (wrist relative to mid-shoulder, nose, and shoulder span)
        """
        if self.hands is None and self.holistic is None and self.pose is None:
            return np.zeros(FEATURE_DIM, dtype=np.float32)
        
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        lh_raw = np.zeros((21, 3), dtype=np.float32)
        rh_raw = np.zeros((21, 3), dtype=np.float32)
        lh_detected = False
        rh_detected = False

        # 1. Hands Detection
        if self.hands is not None:
            try:
                results_hands = self.hands.process(frame_rgb)
                if results_hands.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(
                        results_hands.multi_hand_landmarks,
                        results_hands.multi_handedness or []
                    ):
                        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
                        label = handedness.classification[0].label if handedness.classification else "Right"
                        if label == "Left":
                            lh_raw = pts
                            lh_detected = True
                        else:
                            rh_raw = pts
                            rh_detected = True
                    if not lh_detected and not rh_detected and results_hands.multi_hand_landmarks:
                        rh_raw = np.array([[lm.x, lm.y, lm.z] for lm in results_hands.multi_hand_landmarks[0].landmark], dtype=np.float32)
                        rh_detected = True
            except Exception as e:
                logger.debug(f"Hands processing error: {e}")

        # 2. Pose Detection for Body Anchors
        nose = np.array([0.5, 0.3, 0.0], dtype=np.float32)
        l_sh = np.array([0.65, 0.6, 0.0], dtype=np.float32)
        r_sh = np.array([0.35, 0.6, 0.0], dtype=np.float32)
        
        if self.pose is not None:
            try:
                results_pose = self.pose.process(frame_rgb)
                if results_pose.pose_landmarks:
                    pl = results_pose.pose_landmarks.landmark
                    nose = np.array([pl[0].x, pl[0].y, pl[0].z], dtype=np.float32)
                    l_sh = np.array([pl[11].x, pl[11].y, pl[11].z], dtype=np.float32)
                    r_sh = np.array([pl[12].x, pl[12].y, pl[12].z], dtype=np.float32)
            except Exception as e:
                logger.debug(f"Pose processing error: {e}")

        # Fallback to Holistic if hands weren't detected
        if not lh_detected and not rh_detected and self.holistic is not None:
            try:
                results_hol = self.holistic.process(frame_rgb)
                if results_hol.left_hand_landmarks:
                    lh_raw = np.array([[lm.x, lm.y, lm.z] for lm in results_hol.left_hand_landmarks.landmark], dtype=np.float32)
                    lh_detected = True
                if results_hol.right_hand_landmarks:
                    rh_raw = np.array([[lm.x, lm.y, lm.z] for lm in results_hol.right_hand_landmarks.landmark], dtype=np.float32)
                    rh_detected = True
                if results_hol.pose_landmarks:
                    pl = results_hol.pose_landmarks.landmark
                    nose = np.array([pl[0].x, pl[0].y, pl[0].z], dtype=np.float32)
                    l_sh = np.array([pl[11].x, pl[11].y, pl[11].z], dtype=np.float32)
                    r_sh = np.array([pl[12].x, pl[12].y, pl[12].z], dtype=np.float32)
            except Exception as e:
                logger.debug(f"Holistic fallback processing error: {e}")

        # Compute Body Anchors
        mid_sh = (l_sh + r_sh) / 2.0
        sh_width = max(float(np.linalg.norm(r_sh - l_sh)), 0.15)

        # 18D Body-Relative Features
        lw_rel_mid = ((lh_raw[0] - mid_sh) / sh_width) if lh_detected else np.zeros(3, dtype=np.float32)
        rw_rel_mid = ((rh_raw[0] - mid_sh) / sh_width) if rh_detected else np.zeros(3, dtype=np.float32)
        lw_rel_nose = ((lh_raw[0] - nose) / sh_width) if lh_detected else np.zeros(3, dtype=np.float32)
        rw_rel_nose = ((rh_raw[0] - nose) / sh_width) if rh_detected else np.zeros(3, dtype=np.float32)
        l_sh_rel_mid = (l_sh - mid_sh) / sh_width
        r_sh_rel_mid = (r_sh - mid_sh) / sh_width

        body_rel = np.concatenate([
            lw_rel_mid, rw_rel_mid, lw_rel_nose, rw_rel_nose, l_sh_rel_mid, r_sh_rel_mid
        ]).astype(np.float32)

        # 126D Hand Normalization (Wrist-Centric & Scale Normalized)
        lh_norm = lh_raw.copy()
        if lh_detected and np.any(lh_norm):
            wrist_lh = lh_norm[0].copy()
            lh_norm = lh_norm - wrist_lh
            scale_l = np.max(np.linalg.norm(lh_norm, axis=1))
            if scale_l > 1e-4:
                lh_norm = lh_norm / scale_l

        rh_norm = rh_raw.copy()
        if rh_detected and np.any(rh_norm):
            wrist_rh = rh_norm[0].copy()
            rh_norm = rh_norm - wrist_rh
            scale_r = np.max(np.linalg.norm(rh_norm, axis=1))
            if scale_r > 1e-4:
                rh_norm = rh_norm / scale_r

        feature_vector = np.concatenate([
            lh_norm.flatten(), rh_norm.flatten(), body_rel
        ]).astype(np.float32)

        return feature_vector

    @staticmethod
    def normalize_landmarks(feature_vector: np.ndarray) -> np.ndarray:
        """
        Wrist-centric normalization for invariance to screen position and scale.
        Handles both legacy 126D and enhanced 144D feature vectors.
        """
        features = feature_vector.copy()
        if len(features) < 126:
            pad = np.zeros(FEATURE_DIM - len(features), dtype=np.float32)
            return np.concatenate([features, pad]).astype(np.float32)

        # Left hand wrist is landmark 0 (indices 0..62)
        lh = features[:63].reshape(21, 3)
        if np.any(lh):
            wrist_lh = lh[0].copy()
            lh = lh - wrist_lh
            scale = np.max(np.linalg.norm(lh, axis=1))
            if scale > 1e-4:
                lh = lh / scale
            features[:63] = lh.flatten()

        # Right hand wrist is landmark 0 (indices 63..125)
        rh = features[63:126].reshape(21, 3)
        if np.any(rh):
            wrist_rh = rh[0].copy()
            rh = rh - wrist_rh
            scale = np.max(np.linalg.norm(rh, axis=1))
            if scale > 1e-4:
                rh = rh / scale
            features[63:126] = rh.flatten()

        if len(features) == 126:
            # Pad with 18 zeros for body features if only hands are passed
            pad = np.zeros(18, dtype=np.float32)
            features = np.concatenate([features, pad]).astype(np.float32)

        return features

    def reset(self):
        """Clears buffers and previous smoothing state."""
        self.buffer.clear()
        self.raw_buffer.clear()
        self.prev_landmarks = None

    def add_frame_features(self, feature_vector: np.ndarray):
        """Append both raw and smoothed feature vectors to their respective buffers."""
        feat = feature_vector.copy()
        if len(feat) < FEATURE_DIM:
            pad = np.zeros(FEATURE_DIM - len(feat), dtype=feat.dtype)
            feat = np.concatenate([feat, pad]).astype(np.float32)
        self.raw_buffer.append(feat.copy())
        smoothed = self.smooth_features(feat)
        self.buffer.append(smoothed)

    def get_buffered_sequence(self) -> np.ndarray:
        """
        Returns a smoothed sequence of shape (SEQUENCE_LENGTH, FEATURE_DIM), padded with zeros if buffer is not full.
        """
        if len(self.buffer) == 0:
            return np.zeros((self.buffer_size, FEATURE_DIM), dtype=np.float32)
        
        current_seq = list(self.buffer)
        if len(current_seq) < self.buffer_size:
            pad_count = self.buffer_size - len(current_seq)
            pad = [np.zeros(FEATURE_DIM, dtype=np.float32)] * pad_count
            seq = pad + current_seq
        else:
            seq = current_seq[-self.buffer_size:]
            
        return np.array(seq, dtype=np.float32)

    def get_buffered_raw_sequence(self) -> np.ndarray:
        """
        Returns an unsmoothed raw sequence of shape (SEQUENCE_LENGTH, FEATURE_DIM), padded with zeros if buffer is not full.
        """
        if len(self.raw_buffer) == 0:
            return np.zeros((self.buffer_size, FEATURE_DIM), dtype=np.float32)
        
        current_seq = list(self.raw_buffer)
        if len(current_seq) < self.buffer_size:
            pad_count = self.buffer_size - len(current_seq)
            pad = [np.zeros(FEATURE_DIM, dtype=np.float32)] * pad_count
            seq = pad + current_seq
        else:
            seq = current_seq[-self.buffer_size:]
            
        return np.array(seq, dtype=np.float32)

    async def process(self, state: ConversationState) -> ConversationState:
        """
        If raw landmarks are provided in state, update buffer and set normalized landmark matrix.
        """
        if state.raw_landmarks:
            landmarks_arr = np.array(state.raw_landmarks, dtype=np.float32)
            if landmarks_arr.ndim == 1:
                norm_feat = self.normalize_landmarks(landmarks_arr)
                self.add_frame_features(norm_feat)
            elif landmarks_arr.ndim == 2:
                # Sequence of frames
                for frame_feat in landmarks_arr:
                    norm_feat = self.normalize_landmarks(frame_feat)
                    self.add_frame_features(norm_feat)
                    
        buffered = self.get_buffered_sequence()
        raw_buffered = self.get_buffered_raw_sequence()
        state.metadata["vision_buffer_len"] = len(self.buffer)
        state.metadata["vision_feature_dim"] = FEATURE_DIM
        state.metadata["raw_landmark_window"] = raw_buffered.tolist()
        state.landmark_shape = list(buffered.shape)
        # Pass the full rolling temporal sequence to downstream SignRecognitionAgent
        state.raw_landmarks = buffered.tolist()
        return state
