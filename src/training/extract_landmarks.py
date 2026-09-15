"""
Parallel Landmark Feature Extractor for ASL Citizen Dataset.
Processes dataset videos concurrently using MediaPipe Hands and saves extracted sequence archives.
"""
import sys
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import threading
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, Dict, List

import mediapipe as mp
from src.data.vocabulary import get_word_from_gloss, get_class_index, CORE_WORDS
from src.agents.vision_agent import VisionAgent
from src.config import DATASET_VIDEO_DIR, BASE_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_thread_local = threading.local()


def get_thread_detectors():
    """Returns or creates thread-local MediaPipe Hands and Pose detectors."""
    if not hasattr(_thread_local, "hands_detector"):
        _thread_local.hands_detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.15,
            min_tracking_confidence=0.15
        )
    if not hasattr(_thread_local, "pose_detector"):
        _thread_local.pose_detector = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.20,
            min_tracking_confidence=0.20
        )
    return _thread_local.hands_detector, _thread_local.pose_detector


def process_single_video(task: Tuple[str, str, int, str]) -> Optional[Dict]:
    """
    Worker task extracting 144D landmarks for a single video.
    """
    video_path, word, class_idx, video_filename = task
    if not os.path.exists(video_path):
        return None

    hands_detector, pose_detector = get_thread_detectors()
    cap = cv2.VideoCapture(video_path)
    frames_features = []
    frame_count = 0
    max_frames = 120

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lh_raw = np.zeros((21, 3), dtype=np.float32)
        rh_raw = np.zeros((21, 3), dtype=np.float32)
        lh_detected = False
        rh_detected = False

        # 1. Hands Detection
        try:
            results_hands = hands_detector.process(frame_rgb)
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
            logger.debug(f"Hands detector error on {video_filename}: {e}")

        # 2. Pose Detection
        nose = np.array([0.5, 0.3, 0.0], dtype=np.float32)
        l_sh = np.array([0.65, 0.6, 0.0], dtype=np.float32)
        r_sh = np.array([0.35, 0.6, 0.0], dtype=np.float32)

        try:
            results_pose = pose_detector.process(frame_rgb)
            if results_pose.pose_landmarks:
                pl = results_pose.pose_landmarks.landmark
                nose = np.array([pl[0].x, pl[0].y, pl[0].z], dtype=np.float32)
                l_sh = np.array([pl[11].x, pl[11].y, pl[11].z], dtype=np.float32)
                r_sh = np.array([pl[12].x, pl[12].y, pl[12].z], dtype=np.float32)
        except Exception as e:
            logger.debug(f"Pose detector error on {video_filename}: {e}")

        # Compute 18D Body Anchors
        mid_sh = (l_sh + r_sh) / 2.0
        sh_width = max(float(np.linalg.norm(r_sh - l_sh)), 0.15)

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

        feat_vector = np.concatenate([
            lh_norm.flatten(), rh_norm.flatten(), body_rel
        ]).astype(np.float32)

        frames_features.append(feat_vector)
        frame_count += 1

    cap.release()

    if len(frames_features) == 0:
        return None

    seq_arr = np.array(frames_features, dtype=np.float32)
    # Check valid hand activity (first 126 dims)
    valid_frames = int(np.sum(np.max(np.abs(seq_arr[:, :126]), axis=1) > 1e-4))
    if valid_frames < 3:
        return None

    return {
        "sequence": seq_arr,
        "label": class_idx,
        "word": word,
        "video_name": video_filename
    }


def process_dataset_split(
    csv_file: str,
    video_dir: Path,
    output_npz: Path,
    use_extension: bool = True,
    max_samples: Optional[int] = None,
    num_workers: int = 6
):
    """
    Processes a filtered CSV file, extracts landmarks for each video, and saves as a single .npz file.
    """
    df = pd.read_csv(csv_file)
    logger.info(f"Processing {csv_file}: {len(df)} total rows.")

    tasks = []
    for _, row in df.iterrows():
        if max_samples and len(tasks) >= max_samples:
            break

        video_filename = str(row["Video file"])
        gloss = str(row["Gloss"])
        word = get_word_from_gloss(gloss)
        class_idx = get_class_index(word, use_extension=use_extension)

        if class_idx == -1:
            continue

        video_path = os.path.join(video_dir, video_filename)
        if os.path.exists(video_path):
            tasks.append((video_path, word, class_idx, video_filename))

    logger.info(f"Queued {len(tasks)} valid video extraction tasks for {output_npz.name}")

    sequences: List[np.ndarray] = []
    labels: List[int] = []
    video_names: List[str] = []
    words: List[str] = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_single_video, tasks), total=len(tasks), desc=f"Extracting {output_npz.stem}"))

    for res in results:
        if res is not None:
            sequences.append(res["sequence"])
            labels.append(res["label"])
            video_names.append(res["video_name"])
            words.append(res["word"])

    logger.info(f"Successfully extracted {len(sequences)} valid samples for {output_npz.name}")
    
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        sequences=np.array(sequences, dtype=object),
        labels=np.array(labels, dtype=np.int64),
        video_names=np.array(video_names),
        words=np.array(words)
    )
    logger.info(f"Saved dataset archive to {output_npz}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract MediaPipe landmarks from dataset videos.")
    parser.add_argument("--video-dir", type=str, default=str(DATASET_VIDEO_DIR))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    v_dir = Path(args.video_dir)
    out_dir = BASE_DIR / "extracted_landmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean up obsolete val_filtered.npz
    obsolete_val = out_dir / "val_filtered.npz"
    if obsolete_val.exists():
        obsolete_val.unlink()
        logger.info("Removed obsolete val_filtered.npz")

    for split in ["train_filtered.csv", "test_filtered.csv"]:
        csv_path = BASE_DIR / split
        if csv_path.exists():
            out_file = out_dir / f"{csv_path.stem}.npz"
            process_dataset_split(str(csv_path), v_dir, out_file, max_samples=args.max_samples, num_workers=args.workers)
