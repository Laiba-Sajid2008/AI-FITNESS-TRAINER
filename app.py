import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from utils.pose_detector import PoseDetector

# 1. Page Configuration (Ye sab se upar hona chahiye)
st.set_page_config(page_title="AI Fitness Trainer", layout="wide")

st.title("🏋️ AI Fitness Trainer - Live Feed")

# 2. Sidebar Settings
st.sidebar.header("Settings")
exercise = st.sidebar.selectbox("Choose Exercise", ["Squats"])

# 3. Initialize Detector in Session State (Ye error ko rokta hai)
if 'detector' not in st.session_state:
    try:
        st.session_state.detector = PoseDetector()
    except Exception as e:
        st.error(f"Detector load nahi ho saka: {e}")

# Counter State initialize karna
if 'count' not in st.session_state:
    st.session_state.count = 0
    st.session_state.dir = 0

# 4. Layout for Video and Stats
col1, col2 = st.columns([3, 1])

with col1:
    # Live Video ke liye placeholder
    frame_placeholder = st.empty()

with col2:
    # Stats display
    st.metric("Reps Completed", int(st.session_state.count))
    feedback_placeholder = st.empty()

# 5. Live Camera Logic
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        st.warning("Camera access nahi mil raha. Browser permission check karein.")
        break

    # Processing (BGR to RGB convert karna Streamlit ke liye zaroori hai)
    img = cv2.flip(img, 1)
    
    # Detector use karna jo session state mein hai
    if 'detector' in st.session_state:
        img = st.session_state.detector.findPose(img)
        lmList = st.session_state.detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            # Squat Logic: Hip (23), Knee (25), Ankle (27)
            angle = st.session_state.detector.findAngle(img, 23, 25, 27)
            
            # Percentage aur Counting
            per = np.interp(angle, (90, 160), (100, 0))
            if per == 100:
                if st.session_state.dir == 0:
                    st.session_state.count += 0.5
                    st.session_state.dir = 1
            if per == 0:
                if st.session_state.dir == 1:
                    st.session_state.count += 0.5
                    st.session_state.dir = 0
            
            # Feedback Display
            if per > 90:
                feedback_placeholder.success("Good Form!")
            else:
                feedback_placeholder.info("Go Lower!")

    # Live frame ko screen par dikhana
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(img_rgb, channels="RGB", use_container_width=True)

    # Stop button check
    if st.sidebar.button("Stop Training"):
        break

cap.release()
