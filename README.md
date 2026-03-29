# 🏋️ AI Fitness Tracker

> Real-time exercise detection, rep counting, and posture analysis using Computer Vision & AI.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the App](#running-the-app)
- [How It Works](#how-it-works)
- [Supported Exercises](#supported-exercises)
- [Technologies Used](#technologies-used)

---

## Overview

AI Fitness Tracker uses **MediaPipe Pose** and **OpenCV** to detect your body in real-time via camera, count exercise repetitions automatically, and provide AI-based coaching feedback on your form.

---

## Features

| Feature | Description |
|---|---|
| 🎯 Real-time Detection | Detects body pose at 30 FPS via webcam |
| 🔢 Rep Counting | Automatic counting using joint angle state machines |
| 💯 Form Score | Live 0–100% posture quality score |
| 💬 AI Coaching | Context-aware feedback messages for each exercise |
| 🔥 Calorie Estimate | MET-based calorie burn estimation |
| 📊 Progress Bars | Visual rep and set progress tracking |
| 🎨 Dark UI | Sleek Streamlit interface with custom CSS |

---

## Project Structure

```
ai_fitness_tracker/
│
├── app.py                    # Main Streamlit application
│
├── utils/
│   ├── __init__.py
│   ├── pose_detector.py      # MediaPipe pose estimation + skeleton drawing
│   ├── exercise_tracker.py   # Rep counting & set management
│   └── feedback_engine.py    # Form analysis & coaching feedback
│
├── .streamlit/
│   └── config.toml           # Streamlit theme (dark mode)
│
├── .vscode/
│   ├── launch.json           # VS Code run configuration
│   └── settings.json         # Python/linting settings
│
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### 1. Clone / Open in VS Code
Open the `ai_fitness_tracker/` folder in VS Code.

### 2. Create a Virtual Environment (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note:** MediaPipe requires Python 3.8–3.11. OpenCV may need:
> ```bash
> pip install opencv-python-headless   # if opencv-python fails
> ```

---

## Running the App

### Option A — VS Code (recommended)
1. Open the project in VS Code
2. Press **F5** or go to **Run → Start Debugging**
3. Select **"Run AI Fitness Tracker"**
4. Browser opens at `http://localhost:8501`

### Option B — Terminal
```bash
streamlit run app.py
```

### Option C — VS Code Terminal shortcut
```bash
# In VS Code integrated terminal (Ctrl + `)
cd ai_fitness_tracker
streamlit run app.py
```

---

## How It Works

```
Camera Frame
     │
     ▼
PoseDetector (MediaPipe)
     │  → 33 body landmarks (x, y, visibility)
     │  → Custom neon skeleton overlay
     ▼
ExerciseTracker
     │  → Extracts relevant joint angles
     │  → State machine: "down" → "up" → rep counted
     │  → Manages sets automatically
     ▼
FeedbackEngine
     │  → Compares angles to ideal ranges
     │  → Calculates weighted form score
     │  → Generates coaching messages
     ▼
Streamlit UI
     → Live camera feed with HUD overlay
     → Real-time metrics (reps, sets, score, time, calories)
     → Feedback panel with color-coded messages
```

---

## Supported Exercises

| Exercise | Primary Joints Tracked | Rep Trigger |
|---|---|---|
| Push-Ups | Elbow angle, Hip alignment | Elbow extends > 160° |
| Squats | Knee angle, Hip hinge | Knee extends > 160° |
| Bicep Curls | Elbow angle, Shoulder stability | Elbow curls < 40° |
| Shoulder Press | Elbow extension, Shoulder raise | Elbow extends > 160° |
| Lunges | Front & back knee angles | Knee extends > 155° |

---

## Technologies Used

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.8–3.11 | Core language |
| Streamlit | ≥ 1.32 | Web UI framework |
| MediaPipe | ≥ 0.10 | Pose landmark detection |
| OpenCV | ≥ 4.9 | Camera access & frame processing |
| NumPy | ≥ 1.24 | Array operations |

---

## Tips

- Stand **2–3 metres** from camera for full body visibility
- Ensure **good lighting** — avoid strong backlighting
- Wear **fitted clothing** so joints are clearly visible
- Use the **"Show Joint Angles"** toggle to see live angle values
- Adjust **Target Reps** and **Sets** in the sidebar before starting

---

*Built with ❤️ using Python, MediaPipe & Streamlit*
