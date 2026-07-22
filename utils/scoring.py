import numpy as np

def compute_angle_similarity(angles_ref, angles_user):
    if len(angles_ref) != len(angles_user):
        return 0.0
    # Mean absolute difference, normalized by 180 degrees
    diff = np.abs(np.array(angles_ref) - np.array(angles_user))
    mean_diff = np.mean(diff)
    # Map mean_diff (0-180) to score (1-0)
    score = max(0.0, 1.0 - (mean_diff / 180.0))
    return score

def map_score_to_feedback(score):
    if score >= 0.90:
        return "Perfect!", (0, 255, 0)      # Green
    elif score >= 0.75:
        return "Super!", (0, 200, 200)      # Cyan
    elif score >= 0.60:
        return "Good", (255, 165, 0)        # Orange
    else:
        return "Miss", (0, 0, 255)          # Red