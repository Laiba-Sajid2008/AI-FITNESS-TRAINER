"""
Exercise Tracker — Rep counting & set management
Uses joint angles from PoseDetector to count reps for each exercise.
"""

import time


class ExerciseTracker:
    """
    Tracks repetitions and sets for a given exercise using angle-based
    state machines.

    State machine:
        "down" → angle passes UP threshold → "up"  → rep counted
        "up"   → angle passes DOWN threshold → "down"
    """

    # Exercise configurations:
    # joint       : angle key from PoseDetector
    # down_thresh : angle <= this means "down" position (start/end)
    # up_thresh   : angle >= this means "up" position (peak)
    # use_avg     : True = average left+right angles
    EXERCISE_CONFIG = {
        "Push-Ups": {
            "joint": "left_elbow_angle",
            "joint_r": "right_elbow_angle",
            "use_avg": True,
            "down_thresh": 90,
            "up_thresh": 160,
            "description": "Lower chest to floor, push back up",
        },
        "Squats": {
            "joint": "left_knee_angle",
            "joint_r": "right_knee_angle",
            "use_avg": True,
            "down_thresh": 90,
            "up_thresh": 160,
            "description": "Lower hips until knees at 90°, stand back up",
        },
        "Bicep Curls": {
            "joint": "right_elbow_angle",
            "joint_r": "left_elbow_angle",
            "use_avg": False,
            "down_thresh": 40,
            "up_thresh": 150,
            "description": "Curl wrist to shoulder, lower slowly",
            "invert": True,   # rep on DOWN movement (curl up = angle decreasing)
        },
        "Shoulder Press": {
            "joint": "right_elbow_angle",
            "joint_r": "left_elbow_angle",
            "use_avg": True,
            "down_thresh": 90,
            "up_thresh": 160,
            "description": "Press weights overhead, lower to shoulder height",
        },
        "Lunges": {
            "joint": "right_knee_angle",
            "joint_r": "left_knee_angle",
            "use_avg": False,
            "down_thresh": 85,
            "up_thresh": 155,
            "description": "Step forward, lower back knee toward floor, return",
        },
    }

    def __init__(self, exercise: str, target_reps: int = 10, target_sets: int = 3):
        self.exercise = exercise
        self.target_reps = target_reps
        self.target_sets = target_sets
        self.config = self.EXERCISE_CONFIG.get(exercise, self.EXERCISE_CONFIG["Squats"])

        self.reps = 0
        self.sets = 0
        self.stage = "down"   # "down" | "up"
        self.last_rep_time = 0
        self.rep_cooldown = 0.5   # seconds between reps (prevent double-counting)

        self.angle_history = []   # for smoothing

    def track(self, landmarks: dict) -> tuple:
        """
        Given a landmark dict from PoseDetector, update rep/set counters.
        Returns (reps, sets, stage).
        """
        angle = self._get_angle(landmarks)
        if angle is None:
            return self.reps, self.sets, self.stage

        # Smooth angle over 5 frames
        self.angle_history.append(angle)
        if len(self.angle_history) > 5:
            self.angle_history.pop(0)
        smooth_angle = sum(self.angle_history) / len(self.angle_history)

        invert = self.config.get("invert", False)
        now = time.time()

        if not invert:
            # Normal: rep = going UP then back DOWN
            if smooth_angle <= self.config["down_thresh"]:
                self.stage = "down"
            if smooth_angle >= self.config["up_thresh"] and self.stage == "down":
                if now - self.last_rep_time > self.rep_cooldown:
                    self.stage = "up"
                    self.reps += 1
                    self.last_rep_time = now
        else:
            # Inverted (e.g. bicep curl): rep = going DOWN then back UP
            if smooth_angle >= self.config["up_thresh"]:
                self.stage = "up"
            if smooth_angle <= self.config["down_thresh"] and self.stage == "up":
                if now - self.last_rep_time > self.rep_cooldown:
                    self.stage = "down"
                    self.reps += 1
                    self.last_rep_time = now

        # Auto-advance sets
        if self.reps >= self.target_reps:
            self.sets += 1
            self.reps = 0

        return self.reps, self.sets, self.stage

    def reset(self):
        self.reps = 0
        self.sets = 0
        self.stage = "down"
        self.angle_history = []

    # ── Private ───────────────────────────────────────────────────────────────

    def _get_angle(self, landmarks: dict):
        """Extract angle value from landmark dict based on config."""
        joint = self.config["joint"]
        joint_r = self.config.get("joint_r")

        # landmarks dict maps name → (x, y, vis); we need computed angles
        # PoseDetector stores computed angles in self.angles, but we receive
        # the landmark dict here. We'll compute angle from (x,y) coords.
        angle_l = self._compute_from_landmarks(landmarks, joint)
        if self.config.get("use_avg") and joint_r:
            angle_r = self._compute_from_landmarks(landmarks, joint_r)
            if angle_l is not None and angle_r is not None:
                return (angle_l + angle_r) / 2
        return angle_l

    def _compute_from_landmarks(self, landmarks: dict, joint_key: str):
        """
        Map joint_key (e.g. 'left_elbow_angle') to its three landmark names
        and compute the angle.
        """
        import math

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
        if a_name not in landmarks or b_name not in landmarks or c_name not in landmarks:
            return None
        a = landmarks[a_name][:2]
        b = landmarks[b_name][:2]
        c = landmarks[c_name][:2]
        if landmarks[b_name][2] < 0.4:   # low visibility
            return None
        try:
            radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
            angle = abs(math.degrees(radians))
            if angle > 180:
                angle = 360 - angle
            return round(angle, 1)
        except Exception:
            return None
