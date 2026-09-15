"""
Geometric / kinematic ASL gloss classifier used when no trained BiGRU weights exist.

Uses wrist-normalized 21-point hand landmarks (126-D: left + right) plus short-term
motion of the index fingertip across the rolling buffer.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.models.schema import SignPrediction

WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20
FINGER_TIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)


def _hand(feat: np.ndarray, which: str) -> np.ndarray:
    block = feat[:63] if which == "left" else feat[63:126]
    return block.reshape(21, 3)


def _present(hand: np.ndarray) -> bool:
    return float(np.max(np.abs(hand))) > 1e-4


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _finger_extended(hand: np.ndarray, tip_i: int, pip_i: int, mcp_i: int, thumb: bool = False) -> bool:
    tip, pip, mcp, wrist = hand[tip_i], hand[pip_i], hand[mcp_i], hand[WRIST]
    if thumb:
        return _dist(tip, mcp) > _dist(pip, mcp) * 1.15 and _dist(tip, wrist) > 0.28
    return _dist(tip, wrist) > _dist(pip, wrist) * 1.12 and _dist(tip, mcp) > 0.32


def _finger_flags(hand: np.ndarray) -> Tuple[bool, bool, bool, bool, bool]:
    t = _finger_extended(hand, THUMB_TIP, THUMB_IP, THUMB_MCP, thumb=True)
    i = _finger_extended(hand, INDEX_TIP, INDEX_PIP, INDEX_MCP)
    m = _finger_extended(hand, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
    r = _finger_extended(hand, RING_TIP, RING_PIP, RING_MCP)
    p = _finger_extended(hand, PINKY_TIP, PINKY_PIP, PINKY_MCP)
    return t, i, m, r, p


def _curl_mean(hand: np.ndarray) -> float:
    curls = []
    for tip_i, mcp_i in (
        (INDEX_TIP, INDEX_MCP),
        (MIDDLE_TIP, MIDDLE_MCP),
        (RING_TIP, RING_MCP),
        (PINKY_TIP, PINKY_MCP),
    ):
        ext = _dist(hand[tip_i], hand[mcp_i])
        curls.append(1.0 - min(1.0, ext / 0.75))
    return float(np.mean(curls))


def _spread(hand: np.ndarray) -> float:
    return _dist(hand[INDEX_TIP], hand[PINKY_TIP])


def _tips_cluster(hand: np.ndarray) -> float:
    tips = hand[list(FINGER_TIPS[1:]), :]
    return float(np.mean(np.linalg.norm(tips - tips.mean(axis=0), axis=1)))


def _dominant(feat: np.ndarray) -> Tuple[np.ndarray, str, Optional[np.ndarray]]:
    lh, rh = _hand(feat, "left"), _hand(feat, "right")
    lp, rp = _present(lh), _present(rh)
    if rp and not lp:
        return rh, "right", None
    if lp and not rp:
        return lh, "left", None
    if lp and rp:
        if float(np.max(np.linalg.norm(rh, axis=1))) >= float(np.max(np.linalg.norm(lh, axis=1))):
            return rh, "right", lh
        return lh, "left", rh
    return rh, "none", None


def _index_tip_motion(seq: np.ndarray) -> Dict[str, float]:
    xs, ys, zs = [], [], []
    for frame in seq:
        for which in ("right", "left"):
            h = _hand(frame, which)
            if _present(h):
                xs.append(float(h[INDEX_TIP, 0]))
                ys.append(float(h[INDEX_TIP, 1]))
                zs.append(float(h[INDEX_TIP, 2]))
                break
    empty = {
        "std_x": 0.0,
        "std_y": 0.0,
        "osc_x": 0,
        "osc_y": 0,
        "n": len(xs),
        "z_trend": 0.0,
        "z_avg": 0.0,
    }
    if len(xs) < 4:
        return empty
    xa, ya, za = np.array(xs), np.array(ys), np.array(zs)
    dx, dy = np.diff(xa), np.diff(ya)
    third = max(1, len(za) // 3)
    z_trend = float(np.mean(za[-third:]) - np.mean(za[:third]))
    return {
        "std_x": float(np.std(xa)),
        "std_y": float(np.std(ya)),
        "osc_x": int(np.sum((dx[1:] * dx[:-1]) < 0)) if len(dx) > 1 else 0,
        "osc_y": int(np.sum((dy[1:] * dy[:-1]) < 0)) if len(dy) > 1 else 0,
        "n": len(xs),
        "z_trend": z_trend,
        "z_avg": float(np.mean(za[-third:])),
    }


def classify_frame_scores(feat: np.ndarray, motion: Dict[str, float]) -> Dict[str, float]:
    hand, which, other_hand = _dominant(feat)
    if which == "none":
        return {}

    t, i, m, r, p = _finger_flags(hand)
    n_ext = sum((t, i, m, r, p))
    curl = _curl_mean(hand)
    spread = _spread(hand)
    cluster = _tips_cluster(hand)
    two_hands = other_hand is not None and _present(other_hand)

    osc_x, osc_y = motion.get("osc_x", 0), motion.get("osc_y", 0)
    std_x, std_y = motion.get("std_x", 0.0), motion.get("std_y", 0.0)
    wave = (osc_x >= 2 and std_x > 0.08) or std_x > 0.14
    nod = (osc_y >= 2 and std_y > 0.07) or std_y > 0.12
    shake = (osc_x >= 2 and std_x > 0.07)

    # Primary handshape feature flags
    open5 = n_ext >= 4 and curl < 0.32
    flat_b = (not t) and i and m and r and p and curl < 0.45

    # Bunched / Flattened-O handshape (EAT / FOOD): all fingertips converge directly onto thumb tip
    thumb_index_dist = _dist(hand[INDEX_TIP], hand[THUMB_TIP])
    thumb_mid_dist = _dist(hand[MIDDLE_TIP], hand[THUMB_TIP])
    thumb_ring_dist = _dist(hand[RING_TIP], hand[THUMB_TIP])
    thumb_pinky_dist = _dist(hand[PINKY_TIP], hand[THUMB_TIP])
    
    bunched = (
        thumb_index_dist < 0.15
        and thumb_mid_dist < 0.17
        and thumb_ring_dist < 0.22
        and thumb_pinky_dist < 0.25
        and cluster < 0.18
        and curl > 0.25
        and not open5
    )

    # ASL NO: 3-finger snap/pinch (index + middle tips touching thumb tip with ring & pinky curled)
    no_pinch = (
        thumb_index_dist < 0.15
        and thumb_mid_dist < 0.17
        and not r and not p
        and not bunched
    )

    # Hand shape definitions
    fist = n_ext <= 1 and curl > 0.72 and not bunched and not no_pinch
    index_only = (not t) and i and not m and not r and not p and curl > 0.35 and not no_pinch
    w_shape = i and m and r and not p and (not t or _dist(hand[THUMB_TIP], hand[PINKY_TIP]) < 0.20)
    y_shape = t and p and not i and not m and not r and curl > 0.40
    l_shape = t and i and not m and not r and not p and curl > 0.35
    u_shape = (not t) and i and m and not r and not p and _dist(hand[INDEX_TIP], hand[MIDDLE_TIP]) < 0.15
    claw = 0.25 < curl <= 0.72 and not flat_b and not open5 and not bunched and not no_pinch

    # Thumbs up / down: Thumb extended while other 4 fingers are curled in a fist AND thumb is NOT pinched with fingers
    thumbs_up = (
        t and not i and not m and not r and not p
        and hand[THUMB_TIP, 1] < -0.10
        and curl > 0.55
        and thumb_index_dist > 0.22
        and thumb_mid_dist > 0.22
        and not claw
    )
    thumbs_down = (
        t and not i and not m and not r and not p
        and hand[THUMB_TIP, 1] > 0.15
        and curl > 0.55
        and thumb_index_dist > 0.22
        and thumb_mid_dist > 0.22
        and not claw
    )

    # Neutral / resting relaxed hand detection (suppress false positives on idle hands)
    if 0.35 <= curl <= 0.60 and 1 <= n_ext <= 3 and not bunched and not claw:
        if not (t and p) and not (t and i and not m) and not (w_shape or u_shape or thumbs_up or thumbs_down) and not two_hands:
            if not wave and not nod and not shake:
                return {}

    scores: Dict[str, float] = {}

    # Open 5 / Wave gestures (HELLO, GOODBYE)
    # Open hand extended outward, fingertips far from thumb, no bunched pinch
    if (open5 or flat_b) and not bunched and thumb_index_dist > 0.16:
        if wave:
            scores["HELLO"] = 0.95
            scores["GOODBYE"] = 0.82
        elif spread > 0.14 or curl < 0.22:
            scores["HELLO"] = 0.80

    # Fist / Nod gestures (require deliberate nod for affirmative YES)
    if fist and nod:
        scores["YES"] = 0.94
    elif fist and not nod and curl > 0.75:
        scores["YES"] = 0.70

    no_wag = index_only and (osc_x >= 3 and std_x > 0.10) and hand[INDEX_TIP, 1] < -0.30

    # Index finger pointing (ME vs YOU)
    if no_pinch:
        scores["NO"] = 0.94
    elif no_wag:
        scores["NO"] = 0.90
    elif index_only and not no_pinch:
        z_trend = motion.get("z_trend", 0.0)
        z_avg = motion.get("z_avg", float(hand[INDEX_TIP, 2]))
        # Outward point (toward addressee) = finger moved closer to camera over window, or is stably close
        if z_trend < -0.04 or z_avg < -0.08:
            scores["YOU"] = 0.92
            scores["ME"] = 0.70
        else:
            scores["ME"] = 0.94
            scores["YOU"] = 0.70

    # Letter / Specialized ASL Handshapes
    if w_shape:
        scores["WATER"] = 0.92

    if y_shape:
        if nod or std_y > 0.08:
            scores["WHY"] = 0.94
            scores["PHONE"] = 0.75
        else:
            scores["PHONE"] = 0.94
            scores["NOW"] = 0.74

    if l_shape:
        if hand[INDEX_TIP, 1] < -0.10 and abs(hand[INDEX_TIP, 0]) < 0.15:
            scores["WHO"] = 0.94
            scores["LATER"] = 0.80
        else:
            scores["LATER"] = 0.94
            scores["WHO"] = 0.80

    if thumbs_up:
        if hand[THUMB_TIP, 0] > 0.08 and hand[THUMB_TIP, 1] < -0.08:
            scores["TOMORROW"] = 0.96
            scores["GOOD"] = 0.82
        else:
            scores["GOOD"] = 0.94

    if thumbs_down:
        scores["BAD"] = 0.94

    # Flat B handshape (THANKYOU / PLEASE / KNOW / HAPPY / FINISH)
    if flat_b and not wave and not bunched:
        if hand[INDEX_TIP, 1] < -0.60:
            scores["KNOW"] = 0.94
            scores["THANKYOU"] = 0.78
        elif nod or std_y > 0.08:
            scores["HAPPY"] = 0.94
            scores["THANKYOU"] = 0.78
        else:
            scores["THANKYOU"] = 0.90
            scores["PLEASE"] = 0.80
    elif (flat_b or open5) and wave:
        scores["FINISH"] = 0.90

    # Bunched handshape (EAT / FOOD / HOME / SHOP / MONEY)
    if bunched and not open5 and not wave:
        if hand[INDEX_TIP, 1] < -0.18 and hand[INDEX_TIP, 0] > 0.06:
            scores["HOME"] = 0.94
            scores["EAT"] = 0.80
        elif hand[INDEX_TIP, 1] < -0.06:
            scores["EAT"] = 0.94
        elif hand[INDEX_TIP, 1] >= -0.06:
            scores["SHOP"] = 0.92
            scores["MONEY"] = 0.82

    # Claw handshape (WANT / NEED / TIRED / HUNGRY)
    if claw and not two_hands:
        if hand[INDEX_TIP, 1] > -0.25:
            scores["TIRED"] = 0.94
            scores["WANT"] = 0.75
        else:
            scores["WANT"] = 0.92
            scores["NEED"] = 0.82

    if u_shape:
        scores["HAVE"] = 0.94

    if t and m and not i and not r:
        scores["LIKE"] = 0.84

    # Index finger actions (GO, WHERE, WHO, UNDERSTAND, HE_SHE, THEY)
    if index_only and not no_pinch and not no_wag:
        if std_x > 0.08:
            scores["THEY"] = 0.96
        elif osc_x >= 2 and hand[INDEX_TIP, 1] < -0.18:
            scores["WHERE"] = 0.96
        elif hand[INDEX_TIP, 1] < -0.55:
            scores["UNDERSTAND"] = 0.96
        elif hand[INDEX_TIP, 0] > 0.15:
            scores["HE_SHE"] = 0.96
        elif hand[INDEX_TIP, 2] < -0.05 and std_y > 0.06:
            scores["GO"] = 0.96

    if t and n_ext <= 2 and hand[THUMB_TIP, 1] < -0.12 and not thumbs_up:
        scores["NOT"] = 0.82

    c_shape = t and i and not p and 0.25 < curl < 0.65
    if c_shape:
        if hand[INDEX_TIP, 1] < -0.08:
            scores["DRINK"] = 0.86
        else:
            scores["HUNGRY"] = 0.84

    if fist and t and not thumbs_up and not thumbs_down:
        scores["BATHROOM"] = 0.80

    # Middle finger bent (SICK)
    if m and not i and not r and not p and hand[MIDDLE_TIP, 1] < -0.10:
        scores["SICK"] = 0.84

    # Two-handed gestures
    if two_hands and other_hand is not None:
        ot, oi, om, o_r, op = _finger_flags(other_hand)
        other_open = sum((ot, oi, om, o_r, op)) >= 4
        other_fist = sum((ot, oi, om, o_r, op)) <= 1
        other_bunched = _tips_cluster(other_hand) < 0.20
        other_curl = _curl_mean(other_hand)

        if bunched and other_bunched:
            scores["MORE"] = 0.94
            scores["SHOP"] = 0.86
        elif open5 and other_open:
            if wave or std_x > 0.10:
                scores["WHAT"] = 0.90
                scores["HOW"] = 0.82
                scores["FINISH"] = 0.86
            else:
                scores["WE"] = 0.80
        elif flat_b and other_open:
            scores["SCHOOL"] = 0.90
        elif fist and other_fist:
            if osc_x >= 2 or osc_y >= 2:
                scores["CAR"] = 0.90
            elif curl > 0.70 and other_curl > 0.70 and abs(hand[WRIST, 0] - other_hand[WRIST, 0]) < 0.25:
                scores["LOVE"] = 0.90
            else:
                scores["WORK"] = 0.90
        elif (flat_b or open5) and other_fist:
            scores["HELP"] = 0.92
        elif claw and 0.30 < _curl_mean(other_hand) < 0.75:
            scores["WANT"] = 0.92
            scores["NEED"] = 0.82
        elif y_shape:
            scores["TODAY"] = 0.88
            scores["NOW"] = 0.86

    return scores


def classify_sequence(
    landmark_stream: np.ndarray,
    raw_landmark_stream: Optional[np.ndarray] = None,
    confidence_threshold: float = 0.70,
) -> List[SignPrediction]:
    """Classify a (T, 126) or (T, 144) wrist-normalized landmark window."""
    if landmark_stream.ndim != 2 or landmark_stream.shape[-1] < 126:
        return []

    present_idx = [
        i
        for i, row in enumerate(landmark_stream)
        if _present(_hand(row, "left")) or _present(_hand(row, "right"))
    ]
    if len(present_idx) < 6:
        return []

    recent = landmark_stream[present_idx[-20:]]

    # Motion features come from the raw (unsmoothed) window when available
    motion_source = raw_landmark_stream if raw_landmark_stream is not None else landmark_stream
    motion_present_idx = [
        i
        for i, row in enumerate(motion_source)
        if _present(_hand(row, "left")) or _present(_hand(row, "right"))
    ]
    motion_recent = motion_source[motion_present_idx[-20:]] if motion_present_idx else recent
    motion = _index_tip_motion(motion_recent)

    scores = classify_frame_scores(recent[-1], motion)
    if not scores:
        return []

    # Sort scores to check margin
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    sign, conf = ranked[0]

    # If confidence is below threshold, prune
    if conf < confidence_threshold:
        return []

    # If top two candidates are ambiguous (within 0.05 margin) and confidence is moderate, reject false positive
    if len(ranked) > 1 and (ranked[0][1] - ranked[1][1] < 0.05) and conf < 0.90:
        return []

    return [
        SignPrediction(
            sign=sign,
            confidence=round(float(conf), 4),
            start_frame=0,
            end_frame=int(len(landmark_stream)),
        )
    ]
