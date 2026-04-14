import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
from datetime import date

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Fitness Tracker",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    background-color: #0a0a0f;
    color: #e8e8e8;
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 2px; }
.stApp { background: linear-gradient(135deg, #0a0a0f 0%, #0f1a2e 100%); }

.metric-card {
    background: linear-gradient(135deg, #111827, #1f2937);
    border: 1px solid #00f5a020;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 0 20px #00f5a010;
}
.metric-val {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.8rem;
    color: #00f5a0;
    line-height: 1;
}
.metric-label {
    font-size: 0.75rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 4px;
}
.exercise-badge {
    background: #00f5a015;
    border: 1px solid #00f5a040;
    border-radius: 8px;
    padding: 6px 12px;
    display: inline-block;
    font-size: 0.82rem;
    color: #00f5a0;
    margin: 3px;
}
.feedback-box {
    background: #111827;
    border-left: 4px solid #00f5a0;
    border-radius: 0 12px 12px 0;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.88rem;
}
.stButton > button {
    background: linear-gradient(135deg, #00f5a0, #00d4aa);
    color: #0a0a0f;
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 2px;
    font-size: 1rem;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    width: 100%;
}
.stage-pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.stage-up   { background: #00f5a020; color: #00f5a0; border: 1px solid #00f5a040; }
.stage-down { background: #f59e0b20; color: #f59e0b; border: 1px solid #f59e0b40; }
div[data-testid="stSidebar"] { background: #060610 !important; border-right: 1px solid #1f2937; }
.sidebar-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.4rem;
    color: #00f5a0;
    letter-spacing: 3px;
    border-bottom: 1px solid #00f5a030;
    padding-bottom: 8px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ─── MediaPipe ────────────────────────────────────────────────────────────────
mp_pose           = mp.solutions.pose
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ─── Session State ────────────────────────────────────────────────────────────
for k, v in {
    "reps": 0, "stage": None, "calories": 0.0,
    "feedback": [], "exercise": "Bicep Curl",
    "workout_log": [], "total_reps_session": 0,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Helpers ──────────────────────────────────────────────────────────────────
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(rad * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

EXERCISE_MET = {
    "Bicep Curl": 3.5, "Squat": 5.0, "Push-up": 4.5,
    "Shoulder Press": 4.0, "Lateral Raise": 3.0,
}

def calories_per_rep(exercise, weight_kg=70):
    return EXERCISE_MET.get(exercise, 3.5) * weight_kg * (3 / 3600)

def lm(landmarks, part):
    l = landmarks[part.value]
    return [l.x, l.y]

# ─── Exercise Processors ──────────────────────────────────────────────────────
def process_bicep_curl(landmarks, stage):
    angle = calculate_angle(
        lm(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER),
        lm(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW),
        lm(landmarks, mp_pose.PoseLandmark.LEFT_WRIST),
    )
    rep_delta, feedback, new_stage = 0, [], stage
    if angle > 160:
        new_stage = "down"
    elif angle < 40 and stage == "down":
        new_stage, rep_delta = "up", 1
        feedback.append("💪 Rep counted!")
    if 40 <= angle <= 90 and stage == "down":
        feedback.append("⬆️ Keep curling higher")
    return rep_delta, new_stage, feedback, angle

def process_squat(landmarks, stage):
    angle = calculate_angle(
        lm(landmarks, mp_pose.PoseLandmark.LEFT_HIP),
        lm(landmarks, mp_pose.PoseLandmark.LEFT_KNEE),
        lm(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE),
    )
    rep_delta, feedback, new_stage = 0, [], stage
    if angle > 160:
        new_stage = "up"
    elif angle < 90 and stage == "up":
        new_stage, rep_delta = "down", 1
        feedback.append("💪 Rep counted!")
        if angle < 70:
            feedback.append("⚠️ Careful — don't go too deep")
    if stage == "up" and 90 < angle < 160:
        feedback.append("⬇️ Lower to parallel")
    return rep_delta, new_stage, feedback, angle

def process_pushup(landmarks, stage):
    angle = calculate_angle(
        lm(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER),
        lm(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW),
        lm(landmarks, mp_pose.PoseLandmark.LEFT_WRIST),
    )
    rep_delta, feedback, new_stage = 0, [], stage
    if angle > 160:
        new_stage = "up"
    elif angle < 70 and stage == "up":
        new_stage, rep_delta = "down", 1
        feedback.append("💪 Rep counted!")
    return rep_delta, new_stage, feedback, angle

def process_shoulder_press(landmarks, stage):
    angle = calculate_angle(
        lm(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW),
        lm(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER),
        lm(landmarks, mp_pose.PoseLandmark.LEFT_WRIST),
    )
    rep_delta, feedback, new_stage = 0, [], stage
    if angle < 50:
        new_stage = "down"
    elif angle > 150 and stage == "down":
        new_stage, rep_delta = "up", 1
        feedback.append("💪 Rep counted!")
    return rep_delta, new_stage, feedback, angle

def process_lateral_raise(landmarks, stage):
    angle = calculate_angle(
        lm(landmarks, mp_pose.PoseLandmark.LEFT_HIP),
        lm(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER),
        lm(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW),
    )
    rep_delta, feedback, new_stage = 0, [], stage
    if angle < 20:
        new_stage = "down"
    elif angle > 80 and stage == "down":
        new_stage, rep_delta = "up", 1
        feedback.append("💪 Rep counted!")
        if angle > 100:
            feedback.append("⚠️ Don't raise above shoulder level")
    return rep_delta, new_stage, feedback, angle

PROCESSORS = {
    "Bicep Curl":     process_bicep_curl,
    "Squat":          process_squat,
    "Push-up":        process_pushup,
    "Shoulder Press": process_shoulder_press,
    "Lateral Raise":  process_lateral_raise,
}

# ─── Video Processor ──────────────────────────────────────────────────────────
class FitnessProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img     = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            lmks      = results.pose_landmarks.landmark
            exercise  = st.session_state.get("exercise", "Bicep Curl")
            stage     = st.session_state.get("stage")
            weight_kg = st.session_state.get("weight_kg", 70)

            proc = PROCESSORS.get(exercise, process_bicep_curl)
            rep_delta, new_stage, feedback, angle = proc(lmks, stage)

            if rep_delta > 0:
                st.session_state["reps"]               += rep_delta
                st.session_state["total_reps_session"] += rep_delta
                st.session_state["calories"]           += calories_per_rep(exercise, weight_kg)

            st.session_state["stage"]    = new_stage
            st.session_state["feedback"] = feedback

            # Draw skeleton
            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )

            # HUD panel
            h, w = img.shape[:2]
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (230, 150), (8, 8, 18), -1)
            cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
            cv2.putText(img, f"REPS: {st.session_state['reps']}",
                        (12, 48), cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 245, 160), 2)
            cv2.putText(img, f"STAGE: {(new_stage or 'ready').upper()}",
                        (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(img, f"ANGLE: {int(angle)} deg",
                        (12, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (100, 200, 255), 1)
            cv2.putText(img, f"CAL: {st.session_state['calories']:.1f} kcal",
                        (12, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 245, 160), 1)

            # Angle arc (top-right)
            cx, cy, r = w - 65, 65, 50
            cv2.circle(img, (cx, cy), r, (25, 25, 45), -1)
            cv2.circle(img, (cx, cy), r, (0, 245, 160), 2)
            cv2.ellipse(img, (cx, cy), (r-8, r-8), -90, 0, int(angle), (0, 245, 160), 3)
            cv2.putText(img, f"{int(angle)}", (cx-20, cy+9),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚡ CONTROLS</div>', unsafe_allow_html=True)

    exercise = st.selectbox("Exercise", list(PROCESSORS.keys()))
    if exercise != st.session_state.get("exercise"):
        st.session_state.update({"exercise": exercise, "reps": 0, "stage": None,
                                  "calories": 0.0, "feedback": []})

    weight_kg = st.number_input("Your Weight (kg)", 30, 200, 70, 1)
    st.session_state["weight_kg"] = weight_kg

    target_reps = st.slider("Target Reps", 5, 50, 15, 5)

    st.markdown("---")
    if st.button("🔄  Reset Counter"):
        st.session_state.update({"reps": 0, "stage": None, "calories": 0.0, "feedback": []})

    if st.button("💾  Save to Log"):
        if st.session_state["reps"] > 0:
            st.session_state["workout_log"].append({
                "date":     date.today().strftime("%Y-%m-%d"),
                "exercise": st.session_state["exercise"],
                "reps":     st.session_state["reps"],
                "calories": round(st.session_state["calories"], 1),
            })
            st.success(f"Saved {st.session_state['reps']} reps!")
        else:
            st.warning("No reps to save yet.")

    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("""
- Allow camera access  
- Stand so body is fully visible  
- Pick your exercise  
- Move slowly & controlled  
- AI counts reps automatically ✅
""")

# ─── Main UI ──────────────────────────────────────────────────────────────────
st.markdown("# 🏋️ AI FITNESS TRACKER")
st.markdown("**Real-time pose detection & rep counting — powered by MediaPipe + OpenCV**")

# Metric cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{st.session_state["reps"]}</div><div class="metric-label">Reps</div></div>', unsafe_allow_html=True)
with c2:
    pct = min(int(st.session_state["reps"] / max(target_reps, 1) * 100), 100)
    st.markdown(f'<div class="metric-card"><div class="metric-val">{pct}%</div><div class="metric-label">Target Progress</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{st.session_state["calories"]:.1f}</div><div class="metric-label">Calories Burned</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{st.session_state["total_reps_session"]}</div><div class="metric-label">Session Total</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.progress(min(st.session_state["reps"] / max(target_reps, 1), 1.0),
            text=f"{st.session_state['reps']} / {target_reps} reps toward goal")
st.markdown("<br>", unsafe_allow_html=True)

# Camera + feedback
cam_col, info_col = st.columns([3, 1])

with cam_col:
    webrtc_streamer(
        key="fitness-tracker",
        video_processor_factory=FitnessProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}, {"urls": ["stun:stun1.l.google.com:19302"]}, {"urls": ["stun:stun2.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with info_col:
    st.markdown("### Live Feedback")
    stage = st.session_state.get("stage")
    if stage:
        pill = "stage-up" if stage == "up" else "stage-down"
        st.markdown(f'<span class="stage-pill {pill}">{stage.upper()}</span>', unsafe_allow_html=True)
    for fb in st.session_state.get("feedback", []):
        st.markdown(f'<div class="feedback-box">{fb}</div>', unsafe_allow_html=True)
    if not st.session_state.get("feedback"):
        st.markdown('<div class="feedback-box">👁️ Waiting for movement…</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Exercises")
    for ex in PROCESSORS:
        active = "✅ " if ex == st.session_state.get("exercise") else ""
        st.markdown(f'<span class="exercise-badge">{active}{ex}</span>', unsafe_allow_html=True)

# ─── Workout Log ──────────────────────────────────────────────────────────────
if st.session_state["workout_log"]:
    st.markdown("---")
    st.markdown("## 📋 Workout Log")
    import pandas as pd
    df = pd.DataFrame(st.session_state["workout_log"])
    st.dataframe(df, use_container_width=True, hide_index=True,
        column_config={
            "date":     st.column_config.TextColumn("Date"),
            "exercise": st.column_config.TextColumn("Exercise"),
            "reps":     st.column_config.NumberColumn("Reps", format="%d"),
            "calories": st.column_config.NumberColumn("Calories", format="%.1f kcal"),
        })
    lc1, lc2 = st.columns(2)
    lc1.metric("Total Reps",     sum(e["reps"]     for e in st.session_state["workout_log"]))
    lc2.metric("Total Calories", f"{sum(e['calories'] for e in st.session_state['workout_log']):.1f} kcal")
