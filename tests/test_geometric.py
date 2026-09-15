"""Comprehensive collision and classification unit tests for the geometric live classifier."""
import numpy as np
import pytest
from src.agents.geometric_classifier import classify_sequence, _finger_flags, _hand


def _open_right_hand() -> np.ndarray:
    feat = np.zeros(126, dtype=np.float32)
    rh = feat[63:126].reshape(21, 3)
    joints = {
        0: (0.0, 0.0, 0.0),
        1: (-0.12, -0.05, 0.0),
        2: (-0.22, -0.12, 0.0),
        3: (-0.32, -0.22, 0.0),
        4: (-0.42, -0.35, 0.0),  # thumb
        5: (0.05, -0.12, 0.0),
        6: (0.08, -0.38, 0.0),
        7: (0.10, -0.55, 0.0),
        8: (0.12, -0.78, 0.0),  # index
        9: (0.02, -0.12, 0.0),
        10: (0.02, -0.40, 0.0),
        11: (0.02, -0.58, 0.0),
        12: (0.02, -0.82, 0.0),  # middle
        13: (-0.04, -0.10, 0.0),
        14: (-0.08, -0.36, 0.0),
        15: (-0.10, -0.52, 0.0),
        16: (-0.12, -0.74, 0.0),  # ring
        17: (-0.12, -0.08, 0.0),
        18: (-0.20, -0.30, 0.0),
        19: (-0.24, -0.46, 0.0),
        20: (-0.30, -0.68, 0.0),  # pinky
    }
    for i, xyz in joints.items():
        rh[i] = xyz
    feat[63:126] = rh.flatten()
    return feat


def _bunched_hand(target_x: float = -0.05, target_y: float = -0.12, index_dx: float = 0.0) -> np.ndarray:
    feat = np.zeros(126, dtype=np.float32)
    rh = feat[63:126].reshape(21, 3)
    target = np.array([target_x, target_y, 0.0])
    joints = {
        0: (0.0, 0.0, 0.0),
        1: (-0.05, -0.05, 0.0),
        2: (-0.06, -0.08, 0.0),
        3: (-0.06, -0.10, 0.0),
        4: target,  # thumb tip
        5: (0.02, -0.05, 0.0),
        6: (0.03, -0.08, 0.0),
        7: (0.02, -0.10, 0.0),
        8: target + np.array([index_dx + 0.005, 0.005, 0.0]),  # index tip
        9: (0.00, -0.05, 0.0),
        10: (0.01, -0.08, 0.0),
        11: (0.00, -0.10, 0.0),
        12: target + np.array([0.0, 0.005, 0.0]),  # middle tip
        13: (-0.02, -0.05, 0.0),
        14: (-0.02, -0.08, 0.0),
        15: (-0.02, -0.10, 0.0),
        16: target + np.array([-0.005, 0.005, 0.0]),  # ring tip
        17: (-0.04, -0.04, 0.0),
        18: (-0.04, -0.07, 0.0),
        19: (-0.03, -0.09, 0.0),
        20: target + np.array([-0.01, 0.005, 0.0]),  # pinky tip
    }
    for i, xyz in joints.items():
        rh[i] = xyz
    feat[63:126] = rh.flatten()
    return feat


_bunched_eat_hand = _bunched_hand


def _flat_b_hand(index_y: float = -0.70) -> np.ndarray:
    feat = np.zeros(126, dtype=np.float32)
    rh = feat[63:126].reshape(21, 3)
    joints = {
        0: (0.0, 0.0, 0.0),
        1: (-0.05, -0.04, 0.0),
        2: (-0.08, -0.08, 0.0),
        3: (-0.06, -0.12, 0.0),
        4: (-0.03, -0.14, 0.0),  # thumb folded
        5: (0.04, -0.12, 0.0),
        6: (0.05, -0.32, 0.0),
        7: (0.05, -0.50, 0.0),
        8: (0.05, index_y, 0.0),  # index
        9: (0.01, -0.12, 0.0),
        10: (0.01, -0.32, 0.0),
        11: (0.01, -0.50, 0.0),
        12: (0.01, index_y - 0.02, 0.0),  # middle
        13: (-0.02, -0.12, 0.0),
        14: (-0.02, -0.32, 0.0),
        15: (-0.02, -0.50, 0.0),
        16: (-0.02, index_y - 0.04, 0.0),  # ring
        17: (-0.05, -0.10, 0.0),
        18: (-0.05, -0.28, 0.0),
        19: (-0.05, -0.45, 0.0),
        20: (-0.05, index_y - 0.06, 0.0),  # pinky
    }
    for i, xyz in joints.items():
        rh[i] = xyz
    feat[63:126] = rh.flatten()
    return feat


def _index_pointing_hand(index_x: float = 0.06, index_y: float = -0.55, index_z: float = 0.0) -> np.ndarray:
    feat = np.zeros(126, dtype=np.float32)
    rh = feat[63:126].reshape(21, 3)
    joints = {
        0: (0.0, 0.0, 0.0),
        1: (-0.08, -0.06, 0.0),
        2: (-0.12, -0.14, 0.0),
        3: (-0.10, -0.20, 0.0),
        4: (-0.06, -0.22, 0.0),  # thumb folded
        5: (0.04, -0.12, 0.0),
        6: (0.05, -0.25, 0.0),
        7: (0.05, -0.38, 0.0),
        8: (index_x, index_y, index_z),  # index extended
        9: (0.00, -0.12, 0.0),
        10: (0.00, -0.18, 0.0),
        11: (0.00, -0.14, 0.0),
        12: (0.00, -0.08, 0.0),  # middle curled
        13: (-0.04, -0.10, 0.0),
        14: (-0.04, -0.16, 0.0),
        15: (-0.04, -0.12, 0.0),
        16: (-0.04, -0.06, 0.0),  # ring curled
        17: (-0.08, -0.08, 0.0),
        18: (-0.08, -0.14, 0.0),
        19: (-0.08, -0.10, 0.0),
        20: (-0.08, -0.05, 0.0),  # pinky curled
    }
    for i, xyz in joints.items():
        rh[i] = xyz
    feat[63:126] = rh.flatten()
    return feat


def _y_shape_hand() -> np.ndarray:
    feat = np.zeros(126, dtype=np.float32)
    rh = feat[63:126].reshape(21, 3)
    joints = {
        0: (0.0, 0.0, 0.0),
        1: (-0.10, -0.08, 0.0),
        2: (-0.20, -0.18, 0.0),
        3: (-0.30, -0.28, 0.0),
        4: (-0.40, -0.38, 0.0),  # thumb extended
        5: (0.03, -0.10, 0.0),
        6: (0.03, -0.16, 0.0),
        7: (0.03, -0.10, 0.0),
        8: (0.03, -0.06, 0.0),  # index curled
        9: (0.00, -0.10, 0.0),
        10: (0.00, -0.16, 0.0),
        11: (0.00, -0.10, 0.0),
        12: (0.00, -0.06, 0.0),  # middle curled
        13: (-0.03, -0.10, 0.0),
        14: (-0.03, -0.16, 0.0),
        15: (-0.03, -0.10, 0.0),
        16: (-0.03, -0.06, 0.0),  # ring curled
        17: (-0.08, -0.08, 0.0),
        18: (-0.16, -0.20, 0.0),
        19: (-0.24, -0.35, 0.0),
        20: (-0.32, -0.50, 0.0),  # pinky extended
    }
    for i, xyz in joints.items():
        rh[i] = xyz
    feat[63:126] = rh.flatten()
    return feat


def _l_shape_hand(index_y: float = -0.55, index_x: float = 0.05) -> np.ndarray:
    feat = np.zeros(126, dtype=np.float32)
    rh = feat[63:126].reshape(21, 3)
    joints = {
        0: (0.0, 0.0, 0.0),
        1: (-0.10, -0.08, 0.0),
        2: (-0.20, -0.18, 0.0),
        3: (-0.30, -0.25, 0.0),
        4: (-0.42, -0.30, 0.0),  # thumb extended
        5: (0.04, -0.12, 0.0),
        6: (0.05, -0.25, 0.0),
        7: (0.05, -0.38, 0.0),
        8: (index_x, index_y, 0.0),  # index extended
        9: (0.00, -0.12, 0.0),
        10: (0.00, -0.18, 0.0),
        11: (0.00, -0.12, 0.0),
        12: (0.00, -0.08, 0.0),
        13: (-0.04, -0.10, 0.0),
        14: (-0.04, -0.16, 0.0),
        15: (-0.04, -0.10, 0.0),
        16: (-0.04, -0.06, 0.0),
        17: (-0.08, -0.08, 0.0),
        18: (-0.08, -0.14, 0.0),
        19: (-0.08, -0.08, 0.0),
        20: (-0.08, -0.05, 0.0),
    }
    for i, xyz in joints.items():
        rh[i] = xyz
    feat[63:126] = rh.flatten()
    return feat


def _thumbs_up_hand(thumb_x: float = -0.20, thumb_y: float = -0.38) -> np.ndarray:
    feat = np.zeros(126, dtype=np.float32)
    rh = feat[63:126].reshape(21, 3)
    joints = {
        0: (0.0, 0.0, 0.0),
        1: (-0.08, -0.10, 0.0),
        2: (-0.14, -0.20, 0.0),
        3: (-0.16, -0.28, 0.0),
        4: (thumb_x, thumb_y, 0.0),  # thumb pointing up
        5: (0.05, -0.10, 0.0),
        6: (0.06, -0.16, 0.0),
        7: (0.05, -0.10, 0.0),
        8: (0.04, -0.06, 0.0),  # index curled
        9: (0.00, -0.10, 0.0),
        10: (0.00, -0.16, 0.0),
        11: (0.00, -0.10, 0.0),
        12: (0.00, -0.06, 0.0),  # middle curled
        13: (-0.04, -0.10, 0.0),
        14: (-0.04, -0.16, 0.0),
        15: (-0.04, -0.10, 0.0),
        16: (-0.04, -0.06, 0.0),  # ring curled
        17: (-0.08, -0.08, 0.0),
        18: (-0.08, -0.14, 0.0),
        19: (-0.08, -0.08, 0.0),
        20: (-0.08, -0.05, 0.0),  # pinky curled
    }
    for i, xyz in joints.items():
        rh[i] = xyz
    feat[63:126] = rh.flatten()
    return feat


def _w_shape_hand() -> np.ndarray:
    feat = np.zeros(126, dtype=np.float32)
    rh = feat[63:126].reshape(21, 3)
    joints = {
        0: (0.0, 0.0, 0.0),
        1: (-0.06, -0.08, 0.0),
        2: (-0.08, -0.14, 0.0),
        3: (-0.08, -0.20, 0.0),
        4: (-0.06, -0.20, 0.0),  # thumb touches pinky
        5: (0.05, -0.12, 0.0),
        6: (0.08, -0.32, 0.0),
        7: (0.10, -0.48, 0.0),
        8: (0.12, -0.65, 0.0),  # index extended
        9: (0.00, -0.12, 0.0),
        10: (0.00, -0.32, 0.0),
        11: (0.00, -0.48, 0.0),
        12: (0.00, -0.68, 0.0),  # middle extended
        13: (-0.05, -0.12, 0.0),
        14: (-0.08, -0.32, 0.0),
        15: (-0.10, -0.48, 0.0),
        16: (-0.12, -0.65, 0.0),  # ring extended
        17: (-0.08, -0.08, 0.0),
        18: (-0.08, -0.14, 0.0),
        19: (-0.07, -0.18, 0.0),
        20: (-0.06, -0.20, 0.0),  # pinky curled
    }
    for i, xyz in joints.items():
        rh[i] = xyz
    feat[63:126] = rh.flatten()
    return feat


def _u_shape_hand() -> np.ndarray:
    feat = np.zeros(126, dtype=np.float32)
    rh = feat[63:126].reshape(21, 3)
    joints = {
        0: (0.0, 0.0, 0.0),
        1: (-0.08, -0.06, 0.0),
        2: (-0.12, -0.12, 0.0),
        3: (-0.10, -0.18, 0.0),
        4: (-0.06, -0.20, 0.0),  # thumb curled
        5: (0.02, -0.12, 0.0),
        6: (0.03, -0.30, 0.0),
        7: (0.03, -0.45, 0.0),
        8: (0.03, -0.62, 0.0),  # index extended
        9: (-0.02, -0.12, 0.0),
        10: (-0.01, -0.30, 0.0),
        11: (-0.01, -0.45, 0.0),
        12: (-0.01, -0.62, 0.0),  # middle extended next to index
        13: (-0.05, -0.10, 0.0),
        14: (-0.05, -0.16, 0.0),
        15: (-0.05, -0.10, 0.0),
        16: (-0.05, -0.06, 0.0),  # ring curled
        17: (-0.08, -0.08, 0.0),
        18: (-0.08, -0.14, 0.0),
        19: (-0.08, -0.08, 0.0),
        20: (-0.08, -0.05, 0.0),  # pinky curled
    }
    for i, xyz in joints.items():
        rh[i] = xyz
    feat[63:126] = rh.flatten()
    return feat


def _claw_hand(index_y: float = -0.45, index_z: float = 0.0) -> np.ndarray:
    feat = np.zeros(126, dtype=np.float32)
    rh = feat[63:126].reshape(21, 3)
    joints = {
        0: (0.0, 0.0, 0.0),
        1: (-0.06, -0.06, 0.0),
        2: (-0.10, -0.12, 0.0),
        3: (-0.12, -0.18, 0.0),
        4: (-0.14, -0.22, 0.0),  # thumb curved
        5: (0.04, -0.12, 0.0),
        6: (0.06, -0.26, 0.0),
        7: (0.06, -0.36, 0.0),
        8: (0.05, index_y, index_z),  # index clawed
        9: (0.00, -0.12, 0.0),
        10: (0.01, -0.26, 0.0),
        11: (0.01, -0.36, 0.0),
        12: (0.00, index_y, index_z),  # middle clawed
        13: (-0.04, -0.10, 0.0),
        14: (-0.04, -0.24, 0.0),
        15: (-0.04, -0.34, 0.0),
        16: (-0.04, index_y, index_z),  # ring clawed
        17: (-0.08, -0.08, 0.0),
        18: (-0.08, -0.20, 0.0),
        19: (-0.08, -0.30, 0.0),
        20: (-0.08, index_y, index_z),  # pinky clawed
    }
    for i, xyz in joints.items():
        rh[i] = xyz
    feat[63:126] = rh.flatten()
    return feat


# --- Tests ---

def test_open_hand_flags():
    feat = _open_right_hand()
    flags = _finger_flags(_hand(feat, "right"))
    assert sum(flags) >= 4


def test_wave_classifies_hello():
    frames = []
    base = _open_right_hand()
    for i in range(16):
        f = base.copy()
        f[63 + 8 * 3] += 0.25 * ((-1) ** i)
        frames.append(f)
    preds = classify_sequence(np.stack(frames), confidence_threshold=0.5)
    assert preds, "expected a geometric prediction for a waving open hand"
    assert preds[0].sign in {"HELLO", "GOODBYE", "FINISH", "WHAT"}


def test_eat_classifies_correctly():
    frames = [_bunched_hand(target_x=-0.02, target_y=-0.12) for _ in range(12)]
    preds = classify_sequence(np.stack(frames), confidence_threshold=0.5)
    assert preds, "expected a geometric prediction for EAT"
    assert preds[0].sign == "EAT"


def test_home_classifies_correctly_and_does_not_collide_with_eat():
    # HOME: Bunched hand near upper cheek (index_y < -0.18, index_x > 0.06)
    frames = [_bunched_hand(target_x=0.08, target_y=-0.22) for _ in range(12)]
    preds = classify_sequence(np.stack(frames), confidence_threshold=0.5)
    assert preds, "expected a geometric prediction for HOME"
    assert preds[0].sign == "HOME"


def test_shop_and_money_classify_correctly():
    # SHOP / MONEY: Bunched hand positioned lower (index_y >= -0.06)
    frames = [_bunched_hand(target_x=0.0, target_y=-0.02) for _ in range(12)]
    preds = classify_sequence(np.stack(frames), confidence_threshold=0.5)
    assert preds, "expected a geometric prediction for SHOP"
    assert preds[0].sign in {"SHOP", "MONEY"}


def test_flat_b_know_versus_thankyou():
    # KNOW: Flat B high up near forehead (index_y < -0.60)
    frames_know = [_flat_b_hand(index_y=-0.75) for _ in range(12)]
    preds_know = classify_sequence(np.stack(frames_know), confidence_threshold=0.5)
    assert preds_know, "expected prediction for KNOW"
    assert preds_know[0].sign == "KNOW"

    # THANKYOU / PLEASE: Flat B chest level
    frames_thanks = [_flat_b_hand(index_y=-0.55) for _ in range(12)]
    preds_thanks = classify_sequence(np.stack(frames_thanks), confidence_threshold=0.5)
    assert preds_thanks, "expected prediction for THANKYOU"
    assert preds_thanks[0].sign in {"THANKYOU", "PLEASE"}


def test_y_shape_phone_versus_why():
    frames_phone = [_y_shape_hand() for _ in range(12)]
    preds_phone = classify_sequence(np.stack(frames_phone), confidence_threshold=0.5)
    assert preds_phone, "expected prediction for PHONE"
    assert preds_phone[0].sign == "PHONE"

    frames_why = []
    base = _y_shape_hand()
    for i in range(14):
        f = base.copy()
        f[63 + 8 * 3 + 1] += 0.12 * ((-1) ** i)
        frames_why.append(f)
    preds_why = classify_sequence(np.stack(frames_why), confidence_threshold=0.5)
    assert preds_why, "expected prediction for WHY"
    assert preds_why[0].sign == "WHY"


def test_l_shape_later_versus_who():
    frames_later = [_l_shape_hand(index_y=-0.55, index_x=0.20) for _ in range(12)]
    preds_later = classify_sequence(np.stack(frames_later), confidence_threshold=0.5)
    assert preds_later, "expected prediction for LATER"
    assert preds_later[0].sign == "LATER"

    frames_who = [_l_shape_hand(index_y=-0.55, index_x=0.02) for _ in range(12)]
    preds_who = classify_sequence(np.stack(frames_who), confidence_threshold=0.5)
    assert preds_who, "expected prediction for WHO"
    assert preds_who[0].sign == "WHO"


def test_thumbs_up_good_versus_tomorrow():
    frames_good = [_thumbs_up_hand(thumb_x=-0.20, thumb_y=-0.38) for _ in range(12)]
    preds_good = classify_sequence(np.stack(frames_good), confidence_threshold=0.5)
    assert preds_good, "expected prediction for GOOD"
    assert preds_good[0].sign == "GOOD"

    frames_tomorrow = [_thumbs_up_hand(thumb_x=0.12, thumb_y=-0.35) for _ in range(12)]
    preds_tomorrow = classify_sequence(np.stack(frames_tomorrow), confidence_threshold=0.5)
    assert preds_tomorrow, "expected prediction for TOMORROW"
    assert preds_tomorrow[0].sign == "TOMORROW"


def test_w_shape_water():
    frames = [_w_shape_hand() for _ in range(12)]
    preds = classify_sequence(np.stack(frames), confidence_threshold=0.5)
    assert preds, "expected prediction for WATER"
    assert preds[0].sign == "WATER"


def test_u_shape_have():
    frames = [_u_shape_hand() for _ in range(12)]
    preds = classify_sequence(np.stack(frames), confidence_threshold=0.5)
    assert preds, "expected prediction for HAVE"
    assert preds[0].sign == "HAVE"


def test_claw_want_versus_tired():
    frames_want = [_claw_hand(index_y=-0.45) for _ in range(12)]
    preds_want = classify_sequence(np.stack(frames_want), confidence_threshold=0.5)
    assert preds_want, "expected prediction for WANT"
    assert preds_want[0].sign == "WANT"

    frames_tired = [_claw_hand(index_y=-0.15, index_z=-0.25) for _ in range(12)]
    preds_tired = classify_sequence(np.stack(frames_tired), confidence_threshold=0.5)
    assert preds_tired, "expected prediction for TIRED"
    assert preds_tired[0].sign == "TIRED"


def test_index_understand_where_he_she_they():
    # UNDERSTAND: Pointing high (index_y < -0.55)
    frames_und = [_index_pointing_hand(index_y=-0.65) for _ in range(12)]
    preds_und = classify_sequence(np.stack(frames_und), confidence_threshold=0.5)
    assert preds_und, "expected prediction for UNDERSTAND"
    assert preds_und[0].sign == "UNDERSTAND"

    # HE_SHE: Pointing laterally to the side (index_x > 0.15)
    frames_heshe = [_index_pointing_hand(index_x=0.45, index_y=-0.12) for _ in range(12)]
    preds_heshe = classify_sequence(np.stack(frames_heshe), confidence_threshold=0.5)
    assert preds_heshe, "expected prediction for HE_SHE"
    assert preds_heshe[0].sign == "HE_SHE"

    # THEY: Sweeping laterally (std_x > 0.08)
    frames_they = []
    base = _index_pointing_hand(index_x=0.0, index_y=-0.45)
    for i in range(14):
        f = base.copy()
        f[63 + 8 * 3] += 0.50 * (i / 14.0)  # linear lateral sweep
        frames_they.append(f)
    preds_they = classify_sequence(np.stack(frames_they), confidence_threshold=0.5)
    assert preds_they, "expected prediction for THEY"
    assert preds_they[0].sign == "THEY"


def test_no_wag_raw_and_smoothed_pipeline():
    from src.agents.vision_agent import VisionAgent
    
    # Construct an oscillating index finger wag (NO)
    frames = []
    base = _index_pointing_hand(index_x=0.0, index_y=-0.45, index_z=0.0)
    for i in range(16):
        f = base.copy()
        # Fast oscillation on index tip x: reverses direction every frame
        f[63 + 8 * 3] += 0.18 * ((-1) ** i)
        frames.append(f)
    
    raw_seq = np.stack(frames)
    
    # 1. Direct classification with raw sequence provided
    preds_raw = classify_sequence(raw_seq, raw_landmark_stream=raw_seq, confidence_threshold=0.5)
    assert preds_raw, "expected NO prediction on raw sequence"
    assert preds_raw[0].sign == "NO"

    # 2. Pipeline test: pass through VisionAgent to verify raw/smoothed buffer split survives EMA smoothing
    agent = VisionAgent()
    for frame in frames:
        agent.add_frame_features(frame)
    
    smoothed_window = agent.get_buffered_sequence()
    raw_window = agent.get_buffered_raw_sequence()
    
    preds_pipeline = classify_sequence(smoothed_window, raw_landmark_stream=raw_window, confidence_threshold=0.5)
    assert preds_pipeline, "expected NO prediction through VisionAgent buffered pipeline"
    assert preds_pipeline[0].sign == "NO"


def test_me_vs_you_trajectory_disambiguation():
    # 1. YOU: Outward pointing motion (index tip z moves closer to camera, decreasing z)
    frames_you = []
    base = _index_pointing_hand(index_x=0.06, index_y=-0.50, index_z=0.0)
    for i in range(16):
        f = base.copy()
        # Finger extends towards camera: z goes from 0.0 to -0.12
        f[63 + 8 * 3 + 2] = -0.12 * (i / 15.0)
        frames_you.append(f)
    
    seq_you = np.stack(frames_you)
    preds_you = classify_sequence(seq_you, raw_landmark_stream=seq_you, confidence_threshold=0.5)
    assert preds_you, "expected prediction for YOU (outward pointing trajectory)"
    assert preds_you[0].sign == "YOU"

    # 2. ME: Self-referential pointing motion (index tip stationary or near body, z >= 0.0)
    frames_me = []
    base_me = _index_pointing_hand(index_x=0.06, index_y=-0.50, index_z=0.0)
    for i in range(16):
        f = base_me.copy()
        f[63 + 8 * 3 + 2] = 0.02 * (i / 15.0)  # flat/inward near chest
        frames_me.append(f)
    
    seq_me = np.stack(frames_me)
    preds_me = classify_sequence(seq_me, raw_landmark_stream=seq_me, confidence_threshold=0.5)
    assert preds_me, "expected prediction for ME (inward/chest-level point)"
    assert preds_me[0].sign == "ME"

    # 3. Backward compatibility: when raw_landmark_stream is omitted, single-frame z fallback works
    frame_static_you = _index_pointing_hand(index_x=0.06, index_y=-0.50, index_z=-0.10)
    seq_compat_you = np.stack([frame_static_you] * 12)
    preds_compat_you = classify_sequence(seq_compat_you, confidence_threshold=0.5)
    assert preds_compat_you and preds_compat_you[0].sign == "YOU"

    frame_static_me = _index_pointing_hand(index_x=0.06, index_y=-0.50, index_z=0.0)
    seq_compat_me = np.stack([frame_static_me] * 12)
    preds_compat_me = classify_sequence(seq_compat_me, confidence_threshold=0.5)
    assert preds_compat_me and preds_compat_me[0].sign == "ME"


