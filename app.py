import streamlit as st
import cv2
import numpy as np
import time
import mediapipe as mp
from utils.pose_detector import PoseDetector
from utils.exercise_tracker import ExerciseTracker
from utils.feedback_engine import FeedbackEngine

# -- Page Configuration --
st.set_page_config(page_title="AI Fitness Tracker", layout="wide", initial_sidebar_state="expanded")

# -- CSS to match your Video's Exact "Session Steps" Style --
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    .stApp { background-color: #0E1117; color: white; font-family: 'Inter', sans-serif; }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] { background-color: #11141B !important; border-right: 1px solid #1E293B; }

    /* Right Side Metric Boxes (Session Stats) */
    .metric-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-label { font-size: 0.65rem; color: #888; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #00D4FF; margin: 2px 0; line-height: 1; }
    .metric-sub { font-size: 0.7rem; color: #444; }

    /* AI Feedback & Progress Boxes (Corner Boxes) */
    .section-header { font-size: 0.75rem; color: #555; font-weight: 700; text-transform: uppercase; margin-top: 15px; margin-bottom: 8px; }
    .feedback-box { 
        background: rgba(255, 255, 255, 0.02); 
        border-radius: 6px; padding: 12px; 
        border-left: 3px solid #00D4FF; color: #aaa; font-size: 0.8rem; min-height: 50px;
    }
</style>
""", unsafe_allow_html=True)

# -- State Management --
if 'running' not in st.session_state: st.session_state.running = False
if 'reps' not in st.session_state: st.session_state.reps = 0
if 'sets' not in st.session_state: st.session_state.sets = 0
if 'form_score' not in st.session_state: st.session_state.form_score = 0

# -- Sidebar (Matches Video precisely) --
with st.sidebar:
    st.markdown("### AI FITNESS\nTRACKER V1.0")
    st.divider()
    exercise = st.selectbox("EXERCISE MODE", ["Push-Ups", "Squats", "Bicep Curls", "Lunges"])
    difficulty = st.select_slider("DIFFICULTY", options=["Beginner", "Intermediate", "Advanced"])
    target_reps = st.slider("TARGET REPS PER SET", 5, 50, 10)
    target_sets = st.slider("TARGET SETS", 1, 10, 3)
    
    st.markdown("### CAMERA SETTINGS")
    cam_id = st.number_input("CAMERA SOURCE ID", 0, 5, 0)
    
    st.divider()
    col1, col2 = st.columns(2)
    if col1.button("START", use_container_width=True, type="primary"):
        st.session_state.running = True
        st.session_state.start_time = time.time()
        st.session_state.detector = PoseDetector()
        st.session_state.tracker = ExerciseTracker(exercise, target_reps, target_sets)
        st.session_state.fb = FeedbackEngine(exercise, difficulty)
    
    if col2.button("STOP", use_container_width=True):
        st.session_state.running = False

# -- Dashboard UI --
st.title("AI FITNESS TRACKER")
st.caption("COMPUTER VISION • POSE DETECTION • REAL-TIME ANALYSIS")

main_col, side_col = st.columns([2.3, 1])

with main_col:
    st.markdown("<div class='section-header'>● LIVE FEED</div>", unsafe_allow_html=True)
    video_placeholder = st.empty()

with side_col:
    st.markdown("<div class='section-header'>SESSION STATS</div>", unsafe_allow_html=True)
    
    # Pre-building the "Steps" Boxes as they are in the video
    c1, c2 = st.columns(2)
    rep_ph = c1.empty()
    set_ph = c2.empty()
    score_ph = st.empty()
    
    c3, c4 = st.columns(2)
    time_ph = c3.empty()
    kcal_ph = c4.empty()

    # Corner Boxes
    st.markdown("<div class='section-header'>🤖 AI FEEDBACK</div>", unsafe_allow_html=True)
    feedback_ph = st.empty()
    
    st.markdown("<div class='section-header'>📊 PROGRESS</div>", unsafe_allow_html=True)
    progress_ph = st.empty()

    # Initialize placeholders with 0 values so they appear immediately
    rep_ph.markdown(f"<div class='metric-card'><div class='metric-label'>REPS</div><div class='metric-value'>0</div><div class='metric-sub'>/ {target_reps}</div></div>", unsafe_allow_html=True)
    set_ph.markdown(f"<div class='metric-card'><div class='metric-label'>SETS</div><div class='metric-value'>0</div><div class='metric-sub'>/ {target_sets}</div></div>", unsafe_allow_html=True)
    score_ph.markdown(f"<div class='metric-card'><div class='metric-label'>FORM SCORE</div><div class='metric-value'>0%</div></div>", unsafe_allow_html=True)
    time_ph.markdown(f"<div class='metric-card'><div class='metric-label'>DURATION</div><div class='metric-value'>00:00</div></div>", unsafe_allow_html=True)
    kcal_ph.markdown(f"<div class='metric-card'><div class='metric-label'>EST. CALORIES</div><div class='metric-value'>0.0</div></div>", unsafe_allow_html=True)
    feedback_ph.markdown(f"<div class='feedback-box'>Waiting for session start...</div>", unsafe_allow_html=True)
    progress_ph.progress(0.0)

# -- Main Loop --
if st.session_state.running:
    # Camera debug: Trying DSHOW first
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_id)

    while cap.isOpened() and st.session_state.running:
        ret, frame = cap.read()
        if not ret: break

        frame, landmarks = st.session_state.detector.find_pose(frame)
        
        msg = "Start your workout!"
        if landmarks:
            reps, sets, _ = st.session_state.tracker.track(landmarks)
            score, fb_list = st.session_state.fb.analyze(landmarks)
            st.session_state.reps, st.session_state.sets, st.session_state.form_score = reps, sets, score
            msg = fb_list[0] if fb_list else "Form is good!"

        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Update Session Steps Boxes
        elapsed = int(time.time() - st.session_state.start_time)
        m, s = divmod(elapsed, 60)
        kcal = round((0.15 * st.session_state.reps), 1)

        rep_ph.markdown(f"<div class='metric-card'><div class='metric-label'>REPS</div><div class='metric-value'>{st.session_state.reps}</div><div class='metric-sub'>/ {target_reps}</div></div>", unsafe_allow_html=True)
        set_ph.markdown(f"<div class='metric-card'><div class='metric-label'>SETS</div><div class='metric-value'>{st.session_state.sets}</div><div class='metric-sub'>/ {target_sets}</div></div>", unsafe_allow_html=True)
        score_ph.markdown(f"<div class='metric-card'><div class='metric-label'>FORM SCORE</div><div class='metric-value'>{st.session_state.form_score}%</div></div>", unsafe_allow_html=True)
        time_ph.markdown(f"<div class='metric-card'><div class='metric-label'>DURATION</div><div class='metric-value'>{m:02d}:{s:02d}</div></div>", unsafe_allow_html=True)
        kcal_ph.markdown(f"<div class='metric-card'><div class='metric-label'>EST. CALORIES</div><div class='metric-value'>{kcal}</div></div>", unsafe_allow_html=True)
        feedback_ph.markdown(f"<div class='feedback-box'>💡 {msg}</div>", unsafe_allow_html=True)
        progress_ph.progress(min(st.session_state.reps / target_reps, 1.0))

    cap.release()
else:
    # Standby Screen
    off_img = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(0, 640, 40): cv2.line(off_img, (i, 0), (i, 480), (20, 20, 20), 1)
    for i in range(0, 480, 40): cv2.line(off_img, (0, i), (640, i), (20, 20, 20), 1)
    cv2.putText(off_img, "CAMERA OFFLINE", (220, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
    video_placeholder.image(off_img, channels="BGR", use_container_width=True)