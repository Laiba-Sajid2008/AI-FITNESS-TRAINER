import cv2
import mediapipe as mp
import math

class PoseDetector:
    def __init__(self, mode=False, up_body=False, smooth=True, detection_con=0.5, track_con=0.5):
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=mode,
            model_complexity=1,
            smooth_landmarks=smooth,
            min_detection_confidence=detection_con,
            min_tracking_confidence=track_con
        )
        self.results = None  # Safe initialization
        self.lm_list = []

    def findPose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(
                    img,
                    self.results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )
        return img

    def findPosition(self, img, draw=True):
        self.lm_list = []
        if self.results and self.results.pose_landmarks:  # None check added
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lm_list.append([id, cx, cy])
        return self.lm_list

    def findAngle(self, img, p1, p2, p3, draw=True):
        # Safety check - landmarks available hain?
        if len(self.lm_list) <= max(p1, p2, p3):
            return 0

        # Coordinates nikalna
        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        x3, y3 = self.lm_list[p3][1:]

        # Angle calculate karna
        angle = math.degrees(
            math.atan2(y3 - y2, x3 - x2) -
            math.atan2(y1 - y2, x1 - x2)
        )
        if angle < 0:
            angle += 360
        if angle > 180:
            angle = 360 - angle

        # Draw lines and points
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 242, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 10, (0, 242, 255), cv2.FILLED)

        return angle
