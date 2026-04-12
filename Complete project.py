
import streamlit as st
import cv2
import numpy as np
import time
from utils.pose_detector import PoseDetector
from utils.exercise_tracker import ExerciseTracker
from utils.feedback_engine import FeedbackEngine
# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Fitness Tracker",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject Custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0D0F14;
    color: #E8EAF0;
}

/* Hide Streamlit branding */
#MainMenu, footer, header {visibility: hidden;}

/* App background */
.stApp {
    background: linear-gradient(135deg, #0D0F14 0%, #111520 50%, #0D0F14 100%);
}

/* Title */
.main-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    background: linear-gradient(90deg, #00D4FF, #7B2FFF, #FF6B35);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    padding: 0;
}

.subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: #5A6080;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-top: 2px;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #161A24, #1C2033);
    border: 1px solid #252A3A;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00D4FF, #7B2FFF);
}
.metric-label {
    font-size: 0.7rem;
    color: #5A6080;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    color: #00D4FF;
    line-height: 1;
}
.metric-unit {
    font-size: 0.75rem;
    color: #5A6080;
    margin-top: 4px;
}

/* Feedback box */
.feedback-box {
    border-radius: 12px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.85rem;
    font-weight: 500;
    border-left: 3px solid;
}
.feedback-good {
    background: rgba(0, 212, 100, 0.08);
    border-color: #00D464;
    color: #00D464;
}
.feedback-warn {
    background: rgba(255, 193, 7, 0.08);
    border-color: #FFC107;
    color: #FFC107;
}
.feedback-bad {
    background: rgba(255, 75, 75, 0.08);
    border-color: #FF4B4B;
    color: #FF4B4B;
}

/* Score ring */
.score-ring {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    text-align: center;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0D0F14 !important;
    border-right: 1px solid #1C2033;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
    color: #8890A8 !important;
    font-size: 0.78rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00D4FF22, #7B2FFF22);
    border: 1px solid #7B2FFF;
    color: #E8EAF0;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-radius: 8px;
    padding: 10px 24px;
    width: 100%;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00D4FF44, #7B2FFF44);
    border-color: #00D4FF;
    transform: translateY(-1px);
}

/* Camera feed container */
.camera-container {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid #252A3A;
    background: #080A0F;
}

/* Status badge */
.status-live {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0, 212, 100, 0.1);
    border: 1px solid #00D464;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.72rem;
    color: #00D464;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.status-offline {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(90, 96, 128, 0.1);
    border: 1px solid #3A4060;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.72rem;
    color: #5A6080;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Dividers */
hr {
    border: none;
    border-top: 1px solid #1C2033;
    margin: 16px 0;
}

/* Progress bar override */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00D4FF, #7B2FFF) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session State Init ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "running": False,
        "reps": 0,
        "sets": 0,
        "form_score": 0,
        "feedback_list": [],
        "calories": 0.0,
        "session_time": 0,
        "start_time": None,
        "tracker": None,
        "detector": None,
        "feedback_engine": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 20px 0 10px'>
        <div class='main-title' style='font-size:1.6rem'>AI Fitness</div>
        <div class='subtitle'>Tracker v1.0</div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    exercise = st.selectbox(
        "Exercise Mode",
        ["Push-Ups", "Squats", "Bicep Curls", "Shoulder Press", "Lunges"],
        help="Choose the exercise you want to perform"
    )

    difficulty = st.select_slider(
        "Difficulty",
        options=["Beginner", "Intermediate", "Advanced"],
        value="Beginner"
    )

    target_reps = st.slider("Target Reps per Set", 5, 30, 10)
    target_sets = st.slider("Target Sets", 1, 5, 3)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle' style='margin-bottom:12px'>Camera Settings</div>", unsafe_allow_html=True)

    camera_index = st.selectbox("Camera Source", [0, 1, 2], index=0)
    show_skeleton = st.toggle("Show Skeleton", value=True)
    show_angles = st.toggle("Show Joint Angles", value=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        start_btn = st.button("▶  START", use_container_width=True)
    with col_s2:
        stop_btn = st.button("■  STOP", use_container_width=True)

    reset_btn = st.button("↺  RESET SESSION", use_container_width=True)

    if start_btn and not st.session_state.running:
        st.session_state.running = True
        st.session_state.start_time = time.time()
        st.session_state.tracker = ExerciseTracker(exercise, target_reps, target_sets)
        st.session_state.detector = PoseDetector()
        st.session_state.feedback_engine = FeedbackEngine(exercise, difficulty)

    if stop_btn:
        st.session_state.running = False

    if reset_btn:
        for k in ["reps", "sets", "form_score", "feedback_list", "calories", "session_time", "start_time", "tracker", "detector", "feedback_engine"]:
            if k in ["reps", "sets", "form_score", "session_time", "calories"]:
                st.session_state[k] = 0
            elif k == "feedback_list":
                st.session_state[k] = []
            else:
                st.session_state[k] = None
        st.session_state.running = False


# ── Header ────────────────────────────────────────────────────────────────────
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown("""
    <div class='main-title'>AI Fitness Tracker</div>
    <div class='subtitle'>Computer Vision · Pose Detection · Real-Time Analysis</div>
    """, unsafe_allow_html=True)
with col_status:
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    if st.session_state.running:
        st.markdown("<div class='status-live'>● LIVE</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='status-offline'>○ OFFLINE</div>", unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── Main Layout ───────────────────────────────────────────────────────────────
cam_col, stats_col = st.columns([3, 2])

with cam_col:
    st.markdown("<div style='font-size:0.72rem;color:#5A6080;letter-spacing:3px;text-transform:uppercase;margin-bottom:8px'>📹 Live Feed</div>", unsafe_allow_html=True)
    frame_placeholder = st.empty()

    # Default placeholder image
    if not st.session_state.running:
        placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Grid lines for aesthetic
        for i in range(0, 640, 80):
            cv2.line(placeholder_img, (i, 0), (i, 480), (20, 25, 35), 1)
        for i in range(0, 480, 80):
            cv2.line(placeholder_img, (0, i), (640, i), (20, 25, 35), 1)
        # Center text
        cv2.putText(placeholder_img, "CAMERA OFFLINE", (180, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 60, 90), 2)
        cv2.putText(placeholder_img, "Press START to begin", (185, 265),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 50, 75), 1)
        frame_placeholder.image(placeholder_img, channels="BGR", use_container_width=True)

with stats_col:
    # Metrics
    st.markdown("<div style='font-size:0.72rem;color:#5A6080;letter-spacing:3px;text-transform:uppercase;margin-bottom:8px'>📊 Session Stats</div>", unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    reps_ph = m1.empty()
    sets_ph = m2.empty()
    score_ph = st.empty()
    time_ph = st.empty()
    cal_ph = st.empty()

    def render_metrics():
        elapsed = 0
        if st.session_state.start_time:
            elapsed = int(time.time() - st.session_state.start_time)
        mins, secs = divmod(elapsed, 60)

        reps_ph.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Reps</div>
            <div class='metric-value' style='color:#00D4FF'>{st.session_state.reps}</div>
            <div class='metric-unit'>/ {target_reps} target</div>
        </div>""", unsafe_allow_html=True)

        sets_ph.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Sets</div>
            <div class='metric-value' style='color:#7B2FFF'>{st.session_state.sets}</div>
            <div class='metric-unit'>/ {target_sets} target</div>
        </div>""", unsafe_allow_html=True)

        score_color = "#00D464" if st.session_state.form_score >= 75 else "#FFC107" if st.session_state.form_score >= 50 else "#FF4B4B"
        score_ph.markdown(f"""
        <div class='metric-card' style='margin-top:8px'>
            <div class='metric-label'>Form Score</div>
            <div class='metric-value' style='color:{score_color}'>{st.session_state.form_score}<span style='font-size:1.2rem'>%</span></div>
        </div>""", unsafe_allow_html=True)

        time_ph.markdown(f"""
        <div class='metric-card' style='margin-top:8px'>
            <div class='metric-label'>Duration</div>
            <div class='metric-value' style='font-size:2rem;color:#FF6B35'>{mins:02d}:{secs:02d}</div>
        </div>""", unsafe_allow_html=True)

        cal_ph.markdown(f"""
        <div class='metric-card' style='margin-top:8px'>
            <div class='metric-label'>Est. Calories</div>
            <div class='metric-value' style='font-size:2rem;color:#FFB347'>{st.session_state.calories:.1f}</div>
            <div class='metric-unit'>kcal</div>
        </div>""", unsafe_allow_html=True)

    render_metrics()

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.72rem;color:#5A6080;letter-spacing:3px;text-transform:uppercase;margin-bottom:8px'>💬 AI Feedback</div>", unsafe_allow_html=True)
    feedback_ph = st.empty()

    def render_feedback():
        if not st.session_state.feedback_list:
            feedback_ph.markdown("""
            <div class='feedback-box' style='background:rgba(90,96,128,0.05);border-color:#3A4060;color:#5A6080'>
                🤖 Start exercising to receive AI coaching feedback...
            </div>""", unsafe_allow_html=True)
            return
        html = ""
        for fb in st.session_state.feedback_list[-4:]:
            cls = "feedback-good" if fb["type"] == "good" else "feedback-warn" if fb["type"] == "warn" else "feedback-bad"
            icon = "✅" if fb["type"] == "good" else "⚠️" if fb["type"] == "warn" else "❌"
            html += f"<div class='feedback-box {cls}'>{icon} {fb['msg']}</div>"
        feedback_ph.markdown(html, unsafe_allow_html=True)

    render_feedback()

    # Progress bars
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.72rem;color:#5A6080;letter-spacing:3px;text-transform:uppercase;margin-bottom:4px'>Progress</div>", unsafe_allow_html=True)
    progress_ph = st.empty()

    def render_progress():
        reps_pct = min(st.session_state.reps / target_reps, 1.0) if target_reps > 0 else 0
        sets_pct = min(st.session_state.sets / target_sets, 1.0) if target_sets > 0 else 0
        progress_ph.markdown(f"""
        <div style='margin-bottom:6px'>
            <div style='font-size:0.7rem;color:#5A6080;margin-bottom:3px'>Reps {st.session_state.reps}/{target_reps}</div>
            <div style='background:#1C2033;border-radius:4px;height:6px'>
                <div style='width:{reps_pct*100:.0f}%;background:linear-gradient(90deg,#00D4FF,#7B2FFF);height:6px;border-radius:4px;transition:width 0.3s'></div>
            </div>
        </div>
        <div>
            <div style='font-size:0.7rem;color:#5A6080;margin-bottom:3px'>Sets {st.session_state.sets}/{target_sets}</div>
            <div style='background:#1C2033;border-radius:4px;height:6px'>
                <div style='width:{sets_pct*100:.0f}%;background:linear-gradient(90deg,#7B2FFF,#FF6B35);height:6px;border-radius:4px;transition:width 0.3s'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    render_progress()


# ── Main Loop ─────────────────────────────────────────────────────────────────
if st.session_state.running:
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        st.error("❌ Could not open camera. Check your camera index in the sidebar.")
        st.session_state.running = False
    else:
        detector = st.session_state.detector
        tracker = st.session_state.tracker
        fb_engine = st.session_state.feedback_engine

        frame_count = 0

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Camera feed lost.")
                break

            # Pose detection
            frame, landmarks = detector.find_pose(frame, draw=show_skeleton)

            if landmarks:
                # Exercise tracking
                reps, sets, stage = tracker.track(landmarks)
                st.session_state.reps = reps
                st.session_state.sets = sets

                # Form analysis
                score, feedbacks = fb_engine.analyze(landmarks)
                st.session_state.form_score = score

                # Update feedback (throttle)
                if frame_count % 15 == 0 and feedbacks:
                    st.session_state.feedback_list = feedbacks

                # Draw angles on frame if enabled
                if show_angles:
                    detector.draw_angles(frame, landmarks, exercise)

          #Draw HUD ovrlay 
        frame = draw_hud(frame, reps, sets, score, stage, exercise, target_reps, target_sets) # Calories (rough estimate) elapsed = time.time() - st.session_state.start_time st.session_state.calories = estimate_calories(exercise, reps, sets, elapsed)
        frame_placeholder.image(frame, channels="BGR", use_container_width=True)
        render_metrics()
        render_feedback()
        render_progress()

        frame_count += 1

            # Check if session complete
        if st.session_state.sets >= target_sets and target_sets > 0:
                st.session_state.running = False
                st.balloons()
                st.success(f"🎉 Session Complete! {target_sets} sets of {target_reps} {exercise} done!")
                

        cap.release()


def draw_hud(frame, reps, sets, score, stage, exercise, target_reps, target_sets):
    """Draw heads-up display overlay on the camera frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Top bar
    cv2.rectangle(overlay, (0, 0), (w, 60), (10, 12, 20), -1)

    # Exercise name
    cv2.putText(overlay, exercise.upper(), (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 212, 255), 2)

    # Stage indicator
    stage_color = (0, 212, 100) if stage == "up" else (123, 47, 255)
    cv2.putText(overlay, f"STAGE: {stage.upper()}", (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, stage_color, 1)

    # Reps counter (top right)
    rep_str = f"{reps}/{target_reps}"
    cv2.putText(overlay, "REPS", (w - 120, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 96, 128), 1)
    cv2.putText(overlay, rep_str, (w - 120, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 212, 255), 2)

    # Sets counter
    set_str = f"SET {sets+1}/{target_sets}"
    cv2.putText(overlay, set_str, (w//2 - 50, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 179, 71), 1)

    # Form score bar (bottom)
    bar_h = 50
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (10, 12, 20), -1)
    bar_w = int((score / 100) * (w - 40))
    score_color = (0, 212, 100) if score >= 75 else (255, 193, 7) if score >= 50 else (255, 75, 75)
    cv2.rectangle(overlay, (20, h - bar_h + 18), (20 + bar_w, h - bar_h + 32), score_color, -1)
    cv2.rectangle(overlay, (20, h - bar_h + 18), (w - 20, h - bar_h + 32), (30, 35, 50), 1)
    cv2.putText(overlay, f"FORM: {score}%", (20, h - bar_h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, score_color, 1)

    # Blend overlay
    result = cv2.addWeighted(overlay, 0.9, frame, 0.1, 0)
    return result


def estimate_calories(exercise, reps, sets, elapsed_seconds):
    """Rough MET-based calorie estimate."""
    met = {"Push-Ups": 8.0, "Squats": 5.5, "Bicep Curls": 3.5,
           "Shoulder Press": 4.0, "Lunges": 6.0}.get(exercise, 5.0)
    weight_kg = 70  # default body weight
    hours = elapsed_seconds / 3600
    return round(met * weight_kg * hours, 2)
