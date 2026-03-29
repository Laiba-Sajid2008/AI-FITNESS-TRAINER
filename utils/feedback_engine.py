"""
Feedback Engine — Posture & form analysis with real-time coaching.
Analyses joint angles and returns a form score (0–100) plus feedback messages.
"""

import math
import time


class FeedbackEngine:
    """
    Analyses pose landmarks against ideal ranges for each exercise and
    produces a form score and human-readable feedback messages.
    """

    # ── Ideal angle ranges per exercise ──────────────────────────────────────
    # Each entry: { angle_key: (ideal_min, ideal_max, weight, label) }
    IDEAL_RANGES = {
        "Push-Ups": {
            "left_elbow_angle":  (80,  100, 2.0, "Elbow bend"),
            "right_elbow_angle": (80,  100, 2.0, "Elbow bend"),
            "left_hip_angle":    (160, 180, 1.5, "Core/body alignment"),
            "right_hip_angle":   (160, 180, 1.5, "Core/body alignment"),
        },
        "Squats": {
            "left_knee_angle":   (80,  100, 2.0, "Knee depth"),
            "right_knee_angle":  (80,  100, 2.0, "Knee depth"),
            "left_hip_angle":    (80,  110, 1.5, "Hip hinge"),
            "right_hip_angle":   (80,  110, 1.5, "Hip hinge"),
        },
        "Bicep Curls": {
            "left_elbow_angle":  (30,  50,  2.0, "Left curl depth"),
            "right_elbow_angle": (30,  50,  2.0, "Right curl depth"),
            "left_shoulder_angle":  (150, 180, 1.0, "Left elbow position"),
            "right_shoulder_angle": (150, 180, 1.0, "Right elbow position"),
        },
        "Shoulder Press": {
            "left_elbow_angle":  (150, 180, 2.0, "Left arm extension"),
            "right_elbow_angle": (150, 180, 2.0, "Right arm extension"),
            "left_shoulder_angle":  (60, 90, 1.5, "Left arm raise"),
            "right_shoulder_angle": (60, 90, 1.5, "Right arm raise"),
        },
        "Lunges": {
            "right_knee_angle":  (80, 100, 2.0, "Front knee depth"),
            "left_knee_angle":   (80, 100, 2.0, "Back knee depth"),
            "right_hip_angle":   (80, 110, 1.0, "Hip alignment"),
        },
    }

    # ── Coaching messages ─────────────────────────────────────────────────────
    COACHING = {
        "Push-Ups": {
            "Elbow bend_low":  ("Keep elbows closer to body — flare less!", "warn"),
            "Elbow bend_high": ("Lower your chest closer to the floor!", "bad"),
            "Core/body alignment_low": ("Lift your hips — keep body straight!", "bad"),
            "Core/body alignment_high": ("Drop hips slightly — avoid sagging!", "warn"),
            "good": ("Great push-up form! Keep it up!", "good"),
        },
        "Squats": {
            "Knee depth_low":  ("Go deeper — thighs should be parallel to floor!", "warn"),
            "Knee depth_high": ("Excellent squat depth!", "good"),
            "Hip hinge_low":   ("Hinge hips back more, keep chest up!", "warn"),
            "good": ("Perfect squat! Knees tracking over toes.", "good"),
        },
        "Bicep Curls": {
            "Left curl depth_high":  ("Curl higher — bring fist to shoulder!", "warn"),
            "Right curl depth_high": ("Curl higher — bring fist to shoulder!", "warn"),
            "Left elbow position_low":  ("Keep left elbow tucked to your side!", "warn"),
            "Right elbow position_low": ("Keep right elbow tucked to your side!", "warn"),
            "good": ("Solid curl! Controlled movement.", "good"),
        },
        "Shoulder Press": {
            "Left arm extension_low":  ("Press all the way up — extend fully!", "warn"),
            "Right arm extension_low": ("Press all the way up — extend fully!", "warn"),
            "good": ("Excellent press! Full overhead extension.", "good"),
        },
        "Lunges": {
            "Front knee depth_low":  ("Lunge lower — front knee to 90°!", "warn"),
            "Back knee depth_low":   ("Lower back knee toward the floor!", "warn"),
            "good": ("Great lunge depth and balance!", "good"),
        },
    }

    def __init__(self, exercise: str, difficulty: str = "Beginner"):
        self.exercise = exercise
        self.difficulty = difficulty
        self.ranges = self.IDEAL_RANGES.get(exercise, {})
        self.coaching = self.COACHING.get(exercise, {})
        self._last_score = 100
        self._score_history = []

    def analyze(self, landmarks: dict) -> tuple:
        """
        Analyse landmarks and return (score: int, feedbacks: list[dict]).
        feedbacks items: {"msg": str, "type": "good"|"warn"|"bad"}
        """
        feedbacks = []
        scores = []
        total_weight = 0

        for angle_key, (ideal_min, ideal_max, weight, label) in self.ranges.items():
            angle = self._compute_angle(landmarks, angle_key)
            if angle is None:
                continue

            total_weight += weight
            deviation = 0
            direction = None

            if angle < ideal_min:
                deviation = ideal_min - angle
                direction = "low"
            elif angle > ideal_max:
                deviation = angle - ideal_max
                direction = "high"

            # Score for this joint: 100 if perfect, decreasing with deviation
            max_dev = 60.0   # full deviation = 0 score
            joint_score = max(0, 100 * (1 - deviation / max_dev))
            scores.append(joint_score * weight)

            # Add coaching feedback for notable deviations
            if deviation > 15 and direction:
                key = f"{label}_{direction}"
                if key in self.coaching:
                    msg, mtype = self.coaching[key]
                    feedbacks.append({"msg": msg, "type": mtype})

        # Overall form score
        if total_weight > 0 and scores:
            raw_score = sum(scores) / total_weight
        else:
            raw_score = 75  # neutral when no data

        # Smooth over time
        self._score_history.append(raw_score)
        if len(self._score_history) > 10:
            self._score_history.pop(0)
        score = int(sum(self._score_history) / len(self._score_history))

        # Add positive feedback if form is good
        if not feedbacks and "good" in self.coaching:
            msg, mtype = self.coaching["good"]
            feedbacks.append({"msg": msg, "type": mtype})

        # Difficulty-based tip
        tip = self._difficulty_tip()
        if tip:
            feedbacks.append(tip)

        self._last_score = score
        return score, feedbacks

    # ── Private ───────────────────────────────────────────────────────────────

    def _compute_angle(self, landmarks: dict, joint_key: str):
        """Compute a joint angle from landmark (x, y, vis) tuples."""
        TRIPLETS = {
            "left_elbow_angle":      ("left_shoulder",  "left_elbow",   "left_wrist"),
            "right_elbow_angle":     ("right_shoulder", "right_elbow",  "right_wrist"),
            "left_knee_angle":       ("left_hip",       "left_knee",    "left_ankle"),
            "right_knee_angle":      ("right_hip",      "right_knee",   "right_ankle"),
            "left_hip_angle":        ("left_shoulder",  "left_hip",     "left_knee"),
            "right_hip_angle":       ("right_shoulder", "right_hip",    "right_knee"),
            "left_shoulder_angle":   ("left_elbow",     "left_shoulder","left_hip"),
            "right_shoulder_angle":  ("right_elbow",    "right_shoulder","right_hip"),
        }
        triplet = TRIPLETS.get(joint_key)
        if not triplet:
            return None
        a_name, b_name, c_name = triplet
        if not all(n in landmarks for n in (a_name, b_name, c_name)):
            return None
        if landmarks[b_name][2] < 0.4:
            return None
        a = landmarks[a_name][:2]
        b = landmarks[b_name][:2]
        c = landmarks[c_name][:2]
        try:
            radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
            angle = abs(math.degrees(radians))
            return round(360 - angle if angle > 180 else angle, 1)
        except Exception:
            return None

    def _difficulty_tip(self):
        """Return an extra tip based on difficulty level."""
        tips = {
            "Beginner":     {"msg": "Focus on form over speed — slow reps build strength.", "type": "warn"},
            "Intermediate": {"msg": "Control the eccentric (lowering) phase for max gains.", "type": "warn"},
            "Advanced":     {"msg": "Add time under tension — 3s down, 1s pause, 1s up.", "type": "warn"},
        }
        import random
        # Only occasionally show difficulty tip (1 in 5 chance)
        if random.random() < 0.2:
            return tips.get(self.difficulty)
        return None
