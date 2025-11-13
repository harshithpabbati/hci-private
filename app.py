"""
Driver Drowsiness Detection System - Streamlit Web Interface

This module provides a modern web-based interface for real-time driver monitoring.
Features multi-view monitoring, alert history, and interactive controls.

Features:
- Front View: Face and eye analysis with drowsiness detection
- Side View: Posture monitoring and classification
- Crash Detection: Motion-based crash detection using optical flow
- Yawn Detection: Fatigue monitoring through yawn detection
- Blink Rate Monitoring: Abnormal blink pattern detection
- Alerts Gallery: Historical view of captured alert events
- Real-time FPS counter and metrics display
- Screenshot capture for alert events
"""

import streamlit as st
import time
import pygame
import os
import glob
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageGrab
from attention_scorer import AttentionScorer as AttScorer
from eye_detector import EyeDetector as EyeDet
from pose_estimation import HeadPoseEstimator as HeadPoseEst
from utils import get_landmarks, load_camera_parameters
from arg_parser import get_args
from posture import classify_posture
from crash_detector import CrashDetector
from yawn_detector import YawnDetector
from blink_rate_monitor import BlinkRateMonitor

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize audio system
try:
    pygame.mixer.init()
except pygame.error as e:
    print(f"Warning: Audio system initialization failed: {e}")
    print("The application will run without sound alerts.\n")

sound = None

# Screenshot directory configuration
SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="Driver Drowsiness & Posture Monitor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced custom CSS styling
st.markdown("""
<style>
    /* Main background and font */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    /* Headers */
    .block-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 1.5rem;
        padding: 1rem;
        background: rgba(0, 0, 0, 0.3);
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    
    /* Metric boxes */
    .metric-box {
        padding: 1.5rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-box:hover {
        transform: translateY(-5px);
    }
    
    /* Alert messages */
    .alert-message {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        font-weight: bold;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 4px 8px rgba(255, 0, 0, 0.3);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Success message */
    .success-message {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.75rem;
        transition: all 0.3s ease;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
    }
    
    /* Video frame styling */
    .stImage {
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        overflow: hidden;
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.markdown("### 🚗 Driver Safety Monitor")
st.sidebar.markdown("---")
menu = st.sidebar.radio(
    "📍 Navigation",
    ["🎥 Front View (Face)", "🧍 Side View (Posture)", "📸 Alerts Gallery", "⚙️ Settings"],
    index=0
)
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class='info-box'>
    <h4>ℹ️ About</h4>
    <p>Real-time AI-powered driver safety monitoring system.</p>
    <p><b>Features:</b></p>
    <ul>
        <li>👁️ Drowsiness Detection</li>
        <li>🎯 Distraction Alert</li>
        <li>🥱 Yawn Detection</li>
        <li>👀 Blink Rate Monitoring</li>
        <li>💥 Crash Detection</li>
        <li>🧍 Posture Monitoring</li>
        <li>📸 Auto Screenshot</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Facial landmark groups for visualization
LANDMARK_GROUPS = {
    "left_eye": [33, 160, 158, 133, 153, 144, 33],
    "right_eye": [362, 385, 387, 263, 373, 380, 362],
    "left_eyebrow": [70, 63, 105, 66, 107],
    "right_eyebrow": [336, 296, 334, 293, 300],
    "nose_bridge": [168, 6, 197, 195, 5],
    "outer_lips": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 61],
}


@st.cache_resource
def load_models(camera_params_path: str = None):
    """
    Load and cache detection models for efficient reuse.
    
    Args:
        camera_params_path: Optional path to camera calibration parameters
        
    Returns:
        Tuple of (face_mesh_detector, eye_detector, head_pose_estimator)
    """
    if camera_params_path:
        camera_matrix, dist_coeffs = load_camera_parameters(camera_params_path)
    else:
        camera_matrix, dist_coeffs = None, None
    
    detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True,
    )
    
    return detector, EyeDet(), HeadPoseEst(camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)


def play_alert():
    """Play alert sound if available."""
    if sound:
        sound.play(loops=0)


def capture_screenshot(tag: str):
    """
    Capture and save a screenshot with timestamp.
    
    Args:
        tag: Label for the screenshot (e.g., "ASLEEP", "DISTRACTED")
    """
    global last_screenshot_time
    current_time = time.time()
    
    # Prevent too frequent screenshots
    if current_time - last_screenshot_time < screenshot_interval:
        return
    
    try:
        image = ImageGrab.grab()
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SCREENSHOT_DIR, f"{tag}_{ts}.png")
        image.save(path)
        last_screenshot_time = current_time
        st.toast(f"📸 Screenshot saved: {tag}", icon="✅")
    except Exception as e:
        st.error(f"Screenshot error: {e}")


# Screenshot timing control
last_screenshot_time = 0
screenshot_interval = 2  # Minimum seconds between screenshots


def draw_selected_landmarks(image: np.ndarray, landmarks: np.ndarray, frame_size: tuple):
    """
    Draw facial landmark connections on the image.
    
    Args:
        image: BGR image array
        landmarks: Facial landmark coordinates
        frame_size: Tuple of (width, height)
    """
    h, w = frame_size[1], frame_size[0]
    for group in LANDMARK_GROUPS.values():
        for i in range(len(group) - 1):
            pt1 = tuple((landmarks[group[i], :2] * [w, h]).astype(int))
            pt2 = tuple((landmarks[group[i + 1], :2] * [w, h]).astype(int))
            cv2.line(image, pt1, pt2, (0, 255, 255), 2)


def draw_pupil_centers(frame: np.ndarray, landmarks: np.ndarray, frame_size: tuple):
    """
    Draw pupil center points on the frame.
    
    Args:
        frame: BGR image array
        landmarks: Facial landmark coordinates
        frame_size: Tuple of (width, height)
    """
    h, w = frame_size[1], frame_size[0]
    left_pts = np.array([landmarks[i][:2] * [w, h] for i in [474, 475, 476, 477]], dtype=np.float32)
    right_pts = np.array([landmarks[i][:2] * [w, h] for i in [469, 470, 471, 472]], dtype=np.float32)
    left_center = tuple(np.mean(left_pts, axis=0).astype(int))
    right_center = tuple(np.mean(right_pts, axis=0).astype(int))
    cv2.circle(frame, left_center, 3, (255, 0, 255), -1)
    cv2.circle(frame, right_center, 3, (255, 0, 255), -1)


def process_frame(frame: np.ndarray, detector, eye_det, head_pose, scorer, 
                  frame_size: tuple, yawn_det=None, blink_monitor=None, crash_det=None, t_now=None):
    """
    Process a single frame for drowsiness detection with advanced features.
    
    Args:
        frame: BGR image array
        detector: MediaPipe face mesh detector
        eye_det: Eye detector instance
        head_pose: Head pose estimator instance
        scorer: Attention scorer instance
        frame_size: Tuple of (width, height)
        yawn_det: Optional yawn detector instance
        blink_monitor: Optional blink rate monitor instance
        crash_det: Optional crash detector instance
        t_now: Current timestamp
        
    Returns:
        Tuple of (processed_frame, list_of_alerts, dict_of_metrics)
    """
    alerts = []
    metrics = {}
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.repeat(gray[..., np.newaxis], 3, axis=-1)
    lms = detector.process(gray).multi_face_landmarks
    roll = pitch = yaw = None
    
    if t_now is None:
        t_now = time.perf_counter()
    
    if lms:
        landmarks = get_landmarks(lms)
        draw_selected_landmarks(frame, landmarks, frame_size)
        draw_pupil_centers(frame, landmarks, frame_size)
        
        ear = eye_det.get_EAR(landmarks)
        tired, _ = scorer.get_rolling_PERCLOS(t_now, ear)
        gaze = eye_det.get_Gaze_Score(frame, landmarks, frame_size)
        _, roll, pitch, yaw = head_pose.get_pose(frame, landmarks, frame_size)
        asleep, look_away, distracted = scorer.eval_scores(
            t_now, ear, gaze, roll, pitch, yaw
        )
        
        # Check yawning
        is_yawning = False
        yawn_count = 0
        if yawn_det is not None:
            is_yawning, mar, yawn_count = yawn_det.detect_yawn(landmarks, t_now)
            metrics['yawn_count'] = yawn_count
        
        # Check blink rate
        blink_rate = 0.0
        abnormal_blink = False
        if blink_monitor is not None:
            blink_rate, abnormal_blink, total_blinks = blink_monitor.update(ear, t_now)
            metrics['blink_rate'] = blink_rate
            metrics['total_blinks'] = total_blinks
        
        # Collect alerts
        for flag, label in zip(
            [tired, asleep, look_away, distracted, is_yawning, abnormal_blink],
            ["DROWSY", "ASLEEP", "LOOK AWAY", "DISTRACTED", "YAWNING", "ABNORMAL BLINK"]
        ):
            if flag:
                alerts.append(label)
                play_alert()
                capture_screenshot(label)
        
        # Draw pose information
        if roll is not None:
            cv2.putText(frame, f"Roll: {roll[0]:.1f}°", (10, frame_size[1] - 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if pitch is not None:
            cv2.putText(frame, f"Pitch: {pitch[0]:.1f}°", (10, frame_size[1] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if yaw is not None:
            cv2.putText(frame, f"Yaw: {yaw[0]:.1f}°", (10, frame_size[1] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Run crash detection (works regardless of face detection)
    if crash_det is not None:
        crash_detected, motion_mag, frame = crash_det.detect_crash(frame, t_now)
        metrics['motion'] = motion_mag
        if crash_detected:
            alerts.append("CRASH DETECTED")
            play_alert()
            capture_screenshot("CRASH")
    
    return frame, alerts, metrics


def dashboard(detector, eye_det, head_pose, scorer, args, yawn_det=None, blink_monitor=None, crash_det=None):
    """
    Front view dashboard for face and drowsiness monitoring with advanced features.
    
    Args:
        detector: MediaPipe face mesh detector
        eye_det: Eye detector instance
        head_pose: Head pose estimator instance
        scorer: Attention scorer instance
        args: Command-line arguments
        yawn_det: Optional yawn detector instance
        blink_monitor: Optional blink rate monitor instance
        crash_det: Optional crash detector instance
    """
    st.markdown(
        "<div class='block-header'>🧠 Advanced Driver Monitoring</div>",
        unsafe_allow_html=True
    )
    
    # Create columns for controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        start_button = st.button("🎥 Start Camera", use_container_width=True)
    with col2:
        stop_button = st.button("⏹️ Stop Camera", use_container_width=True)
    with col3:
        features_text = "👁️ Eyes • 🥱 Yawn • 👀 Blink • 💥 Crash • 🎯 Pose"
        st.markdown(f"<div class='info-box'>{features_text}</div>", unsafe_allow_html=True)
    
    # Status and video placeholders
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    video_placeholder = st.empty()
    alert_placeholder = st.empty()
    
    cap = None
    prev_time = time.time()

    if start_button:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            st.error("❌ Failed to open camera. Please check camera connection.")
            return
        status_placeholder.success("✅ Camera started successfully!")

    if stop_button:
        if cap:
            cap.release()
        video_placeholder.empty()
        status_placeholder.info("⏹️ Camera stopped.")
        return

    if cap and cap.isOpened():
        frame_count = 0
        display_frame_count = 0
        last_display_time = time.time()
        display_interval = 0.1  # Update display every 100ms (10 FPS max for display)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Failed to read from camera.")
                break
            
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (640, 480))
            frame_size = frame.shape[1], frame.shape[0]
            
            t_now = time.time()
            
            # Process frame with all features (process every frame)
            processed, messages, metrics = process_frame(
                frame, detector, eye_det, head_pose, scorer, frame_size,
                yawn_det=yawn_det, blink_monitor=blink_monitor, crash_det=crash_det, t_now=t_now
            )
            
            # Display alerts on frame
            y_offset = 40
            for msg in messages:
                cv2.putText(processed, msg, (10, y_offset),
                           cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
                y_offset += 40
            
            # Calculate FPS (based on actual processing)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            
            # Display FPS on frame
            cv2.putText(processed, f"FPS: {fps:.1f}", (10, frame_size[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Only update display at throttled rate to avoid overwhelming Streamlit
            if (curr_time - last_display_time) >= display_interval:
                # Display metrics
                with metrics_placeholder.container():
                    met1, met2, met3, met4, met5 = st.columns(5)
                    with met1:
                        st.metric("📊 FPS", f"{fps:.1f}")
                    with met2:
                        st.metric("🖼️ Processed", frame_count)
                    with met3:
                        st.metric("⚠️ Alerts", len(messages))
                    with met4:
                        if 'yawn_count' in metrics:
                            st.metric("🥱 Yawns", metrics['yawn_count'])
                        else:
                            st.metric("🥱 Yawns", "N/A")
                    with met5:
                        if 'blink_rate' in metrics and metrics['blink_rate'] > 0:
                            st.metric("👀 Blink/min", f"{metrics['blink_rate']:.1f}")
                        else:
                            st.metric("👀 Blink/min", "N/A")
                
                # Display video (throttled)
                video_placeholder.image(processed, channels="BGR", use_container_width=True)
                
                # Display alert messages
                if messages:
                    alert_html = "<div class='alert-message'>🚨 " + " | ".join(messages) + "</div>"
                    alert_placeholder.markdown(alert_html, unsafe_allow_html=True)
                else:
                    alert_placeholder.markdown(
                        "<div class='success-message'>✅ All Clear - Driver Alert</div>",
                        unsafe_allow_html=True
                    )
                
                last_display_time = curr_time
                display_frame_count += 1
            
            frame_count += 1
            time.sleep(0.01)  # Small delay to prevent overwhelming the UI


def side_posture_monitor():
    """Side view dashboard for posture monitoring."""
    st.markdown(
        "<div class='block-header'>🧍 Side View - Posture Detection</div>",
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        start_button = st.button("🎥 Start Posture Cam", use_container_width=True)
    with col2:
        stop_button = st.button("⏹️ Stop Camera", use_container_width=True)
    with col3:
        st.markdown("<div class='info-box'>📐 Monitoring: Back Angle, Arm Position</div>", unsafe_allow_html=True)
    
    status_placeholder = st.empty()
    video_placeholder = st.empty()
    posture_status = st.empty()
    
    cap = None

    if start_button:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Failed to open camera. Please check camera connection.")
            return
        status_placeholder.success("✅ Posture camera started!")

    if stop_button:
        if cap:
            cap.release()
        video_placeholder.empty()
        status_placeholder.info("⏹️ Camera stopped.")
        return

    if cap and cap.isOpened():
        last_display_time = time.time()
        display_interval = 0.1  # Update display every 100ms (10 FPS max)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Failed to read from camera.")
                break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_model.process(rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
                )
                posture, color = classify_posture(results.pose_landmarks.landmark)
            else:
                posture, color = "No person detected", (200, 200, 200)

            # Display posture status
            cv2.putText(frame, posture, (20, 60),
                       cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 3)
            
            # Only update display at throttled rate
            curr_time = time.time()
            if (curr_time - last_display_time) >= display_interval:
                video_placeholder.image(frame, channels="BGR", use_container_width=True)
                
                # Show posture status
                if "Right" in posture:
                    posture_status.markdown(
                        "<div class='success-message'>✅ " + posture + "</div>",
                        unsafe_allow_html=True
                    )
                else:
                    posture_status.markdown(
                        "<div class='alert-message'>⚠️ " + posture + "</div>",
                        unsafe_allow_html=True
                    )
                
                last_display_time = curr_time
            
            time.sleep(0.01)


def gallery():
    """Display gallery of captured alert screenshots."""
    st.markdown(
        "<div class='block-header'>📸 Alert Screenshot Gallery</div>",
        unsafe_allow_html=True
    )
    
    imgs = sorted(glob.glob(os.path.join(SCREENSHOT_DIR, '*.png')),
                 key=os.path.getmtime, reverse=True)
    
    if not imgs:
        st.info("📭 No alert screenshots captured yet. Start monitoring to capture events!")
        return
    
    st.success(f"📊 Total Screenshots: {len(imgs)}")
    
    # Display images in a grid
    cols = st.columns(3)
    for i, img_path in enumerate(imgs):
        with cols[i % 3]:
            try:
                img = Image.open(img_path)
                filename = os.path.basename(img_path)
                st.image(img, caption=filename, use_container_width=True)
                
                # Add delete button
                if st.button(f"🗑️ Delete", key=f"del_{i}"):
                    os.remove(img_path)
                    st.rerun()
            except Exception as e:
                st.error(f"Error loading image: {e}")


def settings_page():
    """Display settings and configuration page."""
    st.markdown(
        "<div class='block-header'>⚙️ System Settings</div>",
        unsafe_allow_html=True
    )
    
    st.markdown("### 🎛️ Detection Thresholds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👁️ Eye Monitoring")
        ear_thresh = st.slider("EAR Threshold", 0.1, 0.3, 0.15, 0.01,
                               help="Lower values = more sensitive drowsiness detection")
        ear_time = st.slider("EAR Time Threshold (s)", 1.0, 5.0, 2.0, 0.5,
                            help="Time before triggering asleep alert")
        
        st.markdown("#### 👀 Gaze Monitoring")
        gaze_thresh = st.slider("Gaze Threshold", 0.01, 0.05, 0.015, 0.001,
                               help="Gaze deviation threshold")
        gaze_time = st.slider("Gaze Time Threshold (s)", 1.0, 5.0, 2.0, 0.5,
                             help="Time before triggering look away alert")
    
    with col2:
        st.markdown("#### 🎯 Head Pose Monitoring")
        roll_thresh = st.slider("Roll Threshold (°)", 10.0, 40.0, 20.0, 5.0,
                               help="Head tilt angle threshold")
        pitch_thresh = st.slider("Pitch Threshold (°)", 10.0, 40.0, 20.0, 5.0,
                                help="Head nod angle threshold")
        yaw_thresh = st.slider("Yaw Threshold (°)", 10.0, 40.0, 20.0, 5.0,
                              help="Head turn angle threshold")
        pose_time = st.slider("Pose Time Threshold (s)", 1.0, 5.0, 2.5, 0.5,
                             help="Time before triggering distraction alert")
    
    st.markdown("### 📊 System Information")
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.info(f"📁 Screenshots: {len(glob.glob(os.path.join(SCREENSHOT_DIR, '*.png')))}")
    with info_col2:
        st.info(f"🎥 Camera: {st.session_state.get('camera', 0)}")
    with info_col3:
        st.info(f"🔊 Sound: {'✅ Enabled' if sound else '❌ Disabled'}")
    
    if st.button("🗑️ Clear All Screenshots", use_container_width=True):
        for img in glob.glob(os.path.join(SCREENSHOT_DIR, '*.png')):
            os.remove(img)
        st.success("✅ All screenshots cleared!")
        st.rerun()

def main():
    """Main entry point for the Streamlit application."""
    global sound
    args = get_args()
    
    # Initialize session state
    if 'camera' not in st.session_state:
        st.session_state.camera = args.camera
    
    # Load sound
    try:
        sound = pygame.mixer.Sound('assets/alarm.mp3')
    except Exception as e:
        st.sidebar.warning(f"⚠️ Alert sound not loaded: {e}")
    
    # Load models
    detector, eye_det, head_pose = load_models(args.camera_params)
    
    # Initialize attention scorer
    scorer = AttScorer(
        t_now=time.perf_counter(),
        ear_thresh=args.ear_thresh,
        gaze_time_thresh=args.gaze_time_thresh,
        roll_thresh=args.roll_thresh,
        pitch_thresh=args.pitch_thresh,
        yaw_thresh=args.yaw_thresh,
        ear_time_thresh=args.ear_time_thresh,
        gaze_thresh=args.gaze_thresh,
        pose_time_thresh=args.pose_time_thresh,
        verbose=getattr(args, 'verbose', False)
    )
    
    # Initialize additional feature detectors
    yawn_det = None
    blink_monitor = None
    crash_det = None
    
    if getattr(args, 'enable_yawn_detection', True):
        yawn_det = YawnDetector(
            mar_thresh=getattr(args, 'yawn_thresh', 0.6),
            verbose=getattr(args, 'verbose', False)
        )
    
    if getattr(args, 'enable_blink_rate', True):
        blink_monitor = BlinkRateMonitor(
            ear_blink_thresh=args.ear_thresh,
            verbose=getattr(args, 'verbose', False)
        )
    
    if getattr(args, 'enable_crash_detection', True):
        crash_det = CrashDetector(
            motion_threshold=getattr(args, 'crash_motion_thresh', 15.0),
            verbose=getattr(args, 'verbose', False)
        )
    
    # Route to appropriate page based on menu selection
    if menu == "🎥 Front View (Face)":
        dashboard(detector, eye_det, head_pose, scorer, args, yawn_det, blink_monitor, crash_det)
    elif menu == "🧍 Side View (Posture)":
        side_posture_monitor()
    elif menu == "📸 Alerts Gallery":
        gallery()
    elif menu == "⚙️ Settings":
        settings_page()


if __name__ == "__main__":
    main()
