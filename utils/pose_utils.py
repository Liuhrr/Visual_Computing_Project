import numpy as np

# COCO keypoint indices (17 points)
KEYPOINT_NAMES = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}

# Joint triplets for angle calculation: (a, b, c) where b is the joint angle vertex
ANGLE_TRIPLETS = [
    (5, 7, 9),  # left elbow: left_shoulder -> left_elbow -> left_wrist
    (6, 8, 10),  # right elbow
    (5, 11, 13),  # left knee (shoulder-hip-knee)
    (6, 12, 14),  # right knee
    (11, 13, 15),  # left ankle (hip-knee-ankle)
    (12, 14, 16),  # right ankle
    (5, 6, 0),  # neck tilt (left_shoulder, right_shoulder, nose)
]


def extract_keypoints_from_result(result, width, height, conf_threshold=0.5):
    if result.keypoints is None:
        return None
    keypoints_xy = result.keypoints.xy[0].cpu().numpy()  # shape (17, 2)
    keypoints_conf = result.keypoints.conf[0].cpu().numpy() if result.keypoints.conf is not None else np.ones(17)

    keypoints = []
    for (x, y), conf in zip(keypoints_xy, keypoints_conf):
        if conf < conf_threshold:
            keypoints.append((None, None))
        else:
            keypoints.append((int(x), int(y)))
    return keypoints


def calculate_angle(a, b, c):
    if None in (a, b, c):
        return None
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return None
    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)


def extract_angle_vector(keypoints):
    angles = []
    for i, j, k in ANGLE_TRIPLETS:
        if i < len(keypoints) and j < len(keypoints) and k < len(keypoints):
            a = keypoints[i]
            b = keypoints[j]
            c = keypoints[k]
            angle = calculate_angle(a, b, c)
            angles.append(angle if angle is not None else 0.0)
        else:
            angles.append(0.0)
    return np.array(angles, dtype=np.float32)