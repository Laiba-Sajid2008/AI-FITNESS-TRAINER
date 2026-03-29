"""
Pose Detector — MediaPipe-based human pose estimation
Detects 33 body landmarks and draws skeleton overlay.
"""

import cv2
import numpy as np
import math

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class PoseDetector:
    """
    Wraps MediaPipe Pose to detect body landmarks from a video frame.
    Falls back to a stub if MediaPipe is not installed.
    """

    # MediaPipe landmark indices
    LANDMARKS = {
        "nose": 0,
        "left_shoulder": 11, "right_shoulder": 12,
        "left_elbow": 13,    "right_elbow": 14,
        "left_wrist": 15,    "right_wrist": 16,
        "left_hip": 23,      "right_hip": 24,
        "left_knee": 25,     "right_knee": 26,
        "left_ankle": 27,    "right_ankle": 28,
    }

    # Joint triplets for angle calculation: (point_a, vertex, point_b)
    ANGLE_JOINTS = {
        "left_elbow_angle":  ("left_shoulder",  "left_elbow",  "left_wrist"),
        "right_elbow_angle": ("right_shoulder", "right_elbow", "right_wrist"),
        "left_knee_angle":   ("left_hip",       "left_knee",   "left_ankle"),
        "right_knee_angle":  ("right_hip",       "right_knee",  "right_ankle"),
        "left_hip_angle":    ("left_shoulder",  "left_hip",    "left_knee"),
        "right_hip_angle":   ("right_shoulder", "right_hip",   "right_knee"),
        "left_shoulder_angle":  ("left_elbow",  "left_shoulder",  "left_hip"),
        "right_shoulder_angle": ("right_elbow", "right_shoulder", "right_hip"),
    }

    def __init__(self, min_detection_confidence=0.6, min_tracking_confidence=0.6):
        self.angles = {}
        self.lm_dict = {}

        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                model_complexity=1,
            )
        else:
            self.pose = None

    # ── Public API ────────────────────────────────────────────────────────────

    def find_pose(self, frame: np.ndarray, draw: bool = True):
        """
        Process a BGR frame, detect pose landmarks.
        Returns (annotated_frame, landmark_dict) where landmark_dict maps
        name → (x_px, y_px, visibility).
        """
        if self.pose is None:
            return frame, self._stub_landmarks(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)
        rgb.flags.writeable = True

        self.lm_dict = {}
        if not results.pose_landmarks:
            return frame, {}

        h, w = frame.shape[:2]
        for name, idx in self.LANDMARKS.items():
            lm = results.pose_landmarks.landmark[idx]
            self.lm_dict[name] = (int(lm.x * w), int(lm.y * h), lm.visibility)

        # Compute all joint angles
        self.angles = self._compute_angles()

        if draw:
            self._draw_custom_skeleton(frame, results.pose_landmarks, w, h)

        return frame, self.lm_dict

    def draw_angles(self, frame: np.ndarray, landmarks: dict, exercise: str):
        """Overlay relevant joint angles on the frame based on exercise type."""
        relevant = {
            "Push-Ups":       ["left_elbow_angle", "right_elbow_angle", "left_hip_angle"],
            "Squats":         ["left_knee_angle", "right_knee_angle", "left_hip_angle"],
            "Bicep Curls":    ["left_elbow_angle", "right_elbow_angle"],
            "Shoulder Press": ["left_shoulder_angle", "right_shoulder_angle", "left_elbow_angle"],
            "Lunges":         ["left_knee_angle", "right_knee_angle"],
        }.get(exercise, list(self.ANGLE_JOINTS.keys())[:3])

        for joint_name in relevant:
            if joint_name not in self.angles:
                continue
            angle = self.angles[joint_name]
            # Find the vertex landmark position
            triplet = self.ANGLE_JOINTS[joint_name]
            vertex = triplet[1]
            if vertex in self.lm_dict:
                x, y, vis = self.lm_dict[vertex]
                if vis > 0.5:
                    self._draw_angle_badge(frame, x, y, angle)

    def get_angle(self, joint_name: str) -> float:
        """Return the computed angle for a named joint (degrees)."""
        return self.angles.get(joint_name, 0.0)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_angles(self) -> dict:
        angles = {}
        for angle_name, (a, b, c) in self.ANGLE_JOINTS.items():
            if a in self.lm_dict and b in self.lm_dict and c in self.lm_dict:
                angles[angle_name] = self._calc_angle(
                    self.lm_dict[a][:2],
                    self.lm_dict[b][:2],
                    self.lm_dict[c][:2],
                )
        return angles

    @staticmethod
    def _calc_angle(a, b, c) -> float:
        """Calculate angle at vertex b given three (x,y) points."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = math.atan2(c[1] - b[1], c[0] - b[0]) - \
                  math.atan2(a[1] - b[1], a[0] - b[0])
        angle = abs(math.degrees(radians))
        if angle > 180:
            angle = 360 - angle
        return round(angle, 1)

    def _draw_custom_skeleton(self, frame, pose_landmarks, w, h):
        """Draw a sleek neon-style skeleton overlay."""
        lms = pose_landmarks.landmark

        def pt(idx):
            lm = lms[idx]
            return (int(lm.x * w), int(lm.y * h))

        def visible(idx, threshold=0.5):
            return lms[idx].visibility > threshold

        # Connections: (idx_a, idx_b, color_BGR)
        connections = [
            # Torso
            (11, 12, (0, 212, 255)),
            (11, 23, (0, 212, 255)),
            (12, 24, (0, 212, 255)),
            (23, 24, (0, 212, 255)),
            # Left arm
            (11, 13, (123, 47, 255)),
            (13, 15, (123, 47, 255)),
            # Right arm
            (12, 14, (123, 47, 255)),
            (14, 16, (123, 47, 255)),
            # Left leg
            (23, 25, (0, 180, 100)),
            (25, 27, (0, 180, 100)),
            # Right leg
            (24, 26, (0, 180, 100)),
            (26, 28, (0, 180, 100)),
        ]

        for a, b, color in connections:
            if visible(a) and visible(b):
                cv2.line(frame, pt(a), pt(b), color, 2, cv2.LINE_AA)

        # Draw joint dots
        key_points = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        for idx in key_points:
            if visible(idx):
                cv2.circle(frame, pt(idx), 5, (255, 255, 255), -1)
                cv2.circle(frame, pt(idx), 7, (0, 212, 255), 1)

    @staticmethod
    def _draw_angle_badge(frame, x, y, angle):
        """Draw a small angle label near a joint."""
        label = f"{angle:.0f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        pad = 4
        rx, ry = x + 10, y - 10
        cv2.rectangle(frame, (rx - pad, ry - th - pad), (rx + tw + pad, ry + pad),
                      (10, 12, 20), -1)
        cv2.rectangle(frame, (rx - pad, ry - th - pad), (rx + tw + pad, ry + pad),
                      (0, 212, 255), 1)
        cv2.putText(frame, label, (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 212, 255), 1, cv2.LINE_AA)

    @staticmethod
    def _stub_landmarks(frame):
        """Return fake landmarks when MediaPipe is unavailable (for UI testing)."""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        return {
            "nose":             (cx,        cy - 160, 1.0),
            "left_shoulder":    (cx - 80,   cy - 100, 1.0),
            "right_shoulder":   (cx + 80,   cy - 100, 1.0),
            "left_elbow":       (cx - 110,  cy - 20,  1.0),
            "right_elbow":      (cx + 110,  cy - 20,  1.0),
            "left_wrist":       (cx - 100,  cy + 60,  1.0),
            "right_wrist":      (cx + 100,  cy + 60,  1.0),
            "left_hip":         (cx - 60,   cy + 40,  1.0),
            "right_hip":        (cx + 60,   cy + 40,  1.0),
            "left_knee":        (cx - 70,   cy + 130, 1.0),
            "right_knee":       (cx + 70,   cy + 130, 1.0),
            "left_ankle":       (cx - 70,   cy + 220, 1.0),
            "right_ankle":      (cx + 70,   cy + 220, 1.0),
        }
