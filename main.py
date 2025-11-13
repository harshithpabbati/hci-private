"""
Driver Drowsiness Detection System - CLI Version

This module provides a command-line interface for real-time driver drowsiness
and distraction detection using computer vision and MediaPipe face mesh.

Features:
- Eye Aspect Ratio (EAR) monitoring for drowsiness detection
- Gaze tracking to detect looking away from the road
- Head pose estimation to detect distraction
- Crash detection using motion analysis
- Yawn detection for fatigue monitoring
- Blink rate monitoring
- Real-time visual feedback with alerts
- Audio alarm system for critical alerts
"""

import time
import pprint
import pygame
import cv2
import mediapipe as mp
import numpy as np

from attention_scorer import AttentionScorer as AttScorer
from eye_detector import EyeDetector as EyeDet
from arg_parser import get_args
from pose_estimation import HeadPoseEstimator as HeadPoseEst
from utils import get_landmarks, load_camera_parameters
from crash_detector import CrashDetector
from yawn_detector import YawnDetector
from blink_rate_monitor import BlinkRateMonitor

# Audio configuration
try:
    pygame.mixer.init()
except pygame.error as e:
    print(f"Warning: Audio system initialization failed: {e}")
    print("The application will run without sound alerts.\n")
    
sound = None

# Display constants
ALERT_FONT = cv2.FONT_HERSHEY_DUPLEX
ALERT_FONT_SCALE = 0.7
ALERT_FONT_THICKNESS = 2
ALERT_COLOR = (0, 0, 255)  # Red for alerts

POSE_FONT = cv2.FONT_HERSHEY_PLAIN
POSE_FONT_SCALE = 1.5
POSE_FONT_THICKNESS = 1
POSE_COLOR = (0, 255, 0)  # Green for pose info


def load_sound(filepath: str) -> pygame.mixer.Sound:
    """
    Load an audio file for alert notifications.
    
    Args:
        filepath: Path to the audio file (MP3, WAV, OGG supported)
        
    Returns:
        pygame.mixer.Sound object or None if loading fails
    """
    try:
        return pygame.mixer.Sound(filepath)
    except pygame.error as e:
        print(f"Error loading sound file '{filepath}': {e}")
        return None

def main():
    """
    Main entry point for the driver drowsiness detection system.
    
    Initializes camera, detection models, and runs the main detection loop.
    Displays real-time video feed with overlay information and triggers
    alerts when drowsiness or distraction is detected.
    """
    args = get_args()
    global sound
    sound = load_sound(r'assets/alarm.mp3')

    # Enable OpenCV optimization if available
    if not cv2.useOptimized():
        try:
            cv2.setUseOptimized(True)
        except Exception as e:
            print(
                f"OpenCV optimization could not be set to True, the script may be slower than expected.\nError: {e}"
            )

    # Load camera calibration parameters if provided
    if args.camera_params:
        camera_matrix, dist_coeffs = load_camera_parameters(args.camera_params)
    else:
        camera_matrix, dist_coeffs = None, None

    # Print configuration if verbose mode is enabled
    if args.verbose:
        print("Arguments and Parameters used:\n")
        pprint.pp(vars(args), indent=4)
        print("\nCamera Matrix:")
        pprint.pp(camera_matrix, indent=4)
        print("\nDistortion Coefficients:")
        pprint.pp(dist_coeffs, indent=4)
        print("\n")

    # Initialize MediaPipe Face Mesh detector
    # Returns 478 landmarks: 468 for face + 10 for irises
    Detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True,
    )

    # Initialize detection and estimation modules
    Eye_det = EyeDet(show_processing=args.show_eye_proc)

    Head_pose = HeadPoseEst(
        show_axis=args.show_axis, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs
    )

    # Initialize timing variables for FPS calculation
    prev_time = time.perf_counter()
    fps = 0.0

    t_now = time.perf_counter()

    # Initialize attention scorer with configured thresholds
    Scorer = AttScorer(
        t_now=t_now,
        ear_thresh=args.ear_thresh,
        gaze_time_thresh=args.gaze_time_thresh,
        roll_thresh=args.roll_thresh,
        pitch_thresh=args.pitch_thresh,
        yaw_thresh=args.yaw_thresh,
        ear_time_thresh=args.ear_time_thresh,
        gaze_thresh=args.gaze_thresh,
        pose_time_thresh=args.pose_time_thresh,
        verbose=args.verbose,
    )

    # Initialize additional feature detectors
    crash_det = None
    yawn_det = None
    blink_monitor = None
    
    if args.enable_crash_detection:
        crash_det = CrashDetector(
            motion_threshold=args.crash_motion_thresh,
            verbose=args.verbose
        )
        print("Crash detection enabled")
    
    if args.enable_yawn_detection:
        yawn_det = YawnDetector(
            mar_thresh=args.yawn_thresh,
            verbose=args.verbose
        )
        print("Yawn detection enabled")
    
    if args.enable_blink_rate:
        blink_monitor = BlinkRateMonitor(
            ear_blink_thresh=args.ear_thresh,
            verbose=args.verbose
        )
        print("Blink rate monitoring enabled")

    # Initialize video capture
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Driver Drowsiness Detection System started.")
    print("Press 'q' to quit.\n")

    # Main detection loop
    while True:
        t_now = time.perf_counter()

        # Calculate FPS
        elapsed_time = t_now - prev_time
        prev_time = t_now

        if elapsed_time > 0:
            fps = np.round(1 / elapsed_time, 3)

        # Capture frame
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame from camera/stream end")
            break

        # Flip frame horizontally if using webcam
        if args.camera == 0:
            frame = cv2.flip(frame, 2)

        e1 = cv2.getTickCount()

        # Convert to grayscale and prepare for MediaPipe processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_size = frame.shape[1], frame.shape[0]
        gray = np.expand_dims(gray, axis=2)
        gray = np.concatenate([gray, gray, gray], axis=2)

        # Detect facial landmarks
        lms = Detector.process(gray).multi_face_landmarks

        alert_messages = []
        if lms:
            landmarks = get_landmarks(lms)
            
            # Draw eye keypoints on frame
            Eye_det.show_eye_keypoints(
                color_frame=frame, landmarks=landmarks, frame_size=frame_size
            )

            # Calculate Eye Aspect Ratio (EAR)
            ear = Eye_det.get_EAR(landmarks=landmarks)

            # Compute PERCLOS score for tiredness detection
            tired, perclos_score = Scorer.get_rolling_PERCLOS(t_now, ear)

            # Compute Gaze Score
            gaze = Eye_det.get_Gaze_Score(
                frame=gray, landmarks=landmarks, frame_size=frame_size
            )

            # Compute head pose (roll, pitch, yaw)
            frame_det, roll, pitch, yaw = Head_pose.get_pose(
                frame=frame, landmarks=landmarks, frame_size=frame_size
            )

            # Evaluate all scores and determine driver state
            asleep, looking_away, distracted = Scorer.eval_scores(
                t_now=t_now,
                ear_score=ear,
                gaze_score=gaze,
                head_roll=roll,
                head_pitch=pitch,
                head_yaw=yaw,
            )

            # Update frame if head pose estimation was successful
            if frame_det is not None:
                frame = frame_det

            # Run additional feature detections
            is_yawning = False
            yawn_count = 0
            if yawn_det is not None:
                is_yawning, mar, yawn_count = yawn_det.detect_yawn(landmarks, t_now)
            
            blink_rate = 0.0
            abnormal_blink = False
            if blink_monitor is not None:
                blink_rate, abnormal_blink, total_blinks = blink_monitor.update(ear, t_now)

            # Collect and display alert messages
            if tired:
                alert_messages.append("TIRED!")
            if asleep:
                alert_messages.append("ASLEEP!")
                if sound:
                    sound.play(loops=0)
            if looking_away:
                alert_messages.append("LOOKING AWAY!")
                if sound:
                    sound.play(loops=0)
            if distracted:
                alert_messages.append("DISTRACTED!")
                if sound:
                    sound.play(loops=0)
            if is_yawning:
                alert_messages.append("YAWNING!")
            if abnormal_blink:
                alert_messages.append("ABNORMAL BLINK RATE!")

            # Display alert messages at top left in red
            y_position = 30
            for message in alert_messages:
                cv2.putText(
                    frame,
                    message,
                    (10, y_position),
                    ALERT_FONT,
                    ALERT_FONT_SCALE,
                    ALERT_COLOR,
                    ALERT_FONT_THICKNESS,
                    cv2.LINE_AA,
                )
                y_position += 30

            # Display additional metrics at bottom left
            metrics_y = frame.shape[0] - 80
            if yawn_det is not None:
                cv2.putText(
                    frame,
                    f"Yawns: {yawn_count}",
                    (10, metrics_y),
                    POSE_FONT,
                    POSE_FONT_SCALE,
                    POSE_COLOR,
                    POSE_FONT_THICKNESS,
                    cv2.LINE_AA,
                )
                metrics_y += 25
            
            if blink_monitor is not None and blink_rate > 0:
                cv2.putText(
                    frame,
                    f"Blink Rate: {blink_rate:.1f}/min",
                    (10, metrics_y),
                    POSE_FONT,
                    POSE_FONT_SCALE,
                    POSE_COLOR,
                    POSE_FONT_THICKNESS,
                    cv2.LINE_AA,
                )

            # Display head pose angles at top right in green
            x_position_pose = frame.shape[1] - 150
            y_position_pose = 40
            if roll is not None:
                cv2.putText(
                    frame,
                    f"Roll:{roll.round(1)[0]}",
                    (x_position_pose, y_position_pose),
                    POSE_FONT,
                    POSE_FONT_SCALE,
                    POSE_COLOR,
                    POSE_FONT_THICKNESS,
                    cv2.LINE_AA,
                )
                y_position_pose += 30
            if pitch is not None:
                cv2.putText(
                    frame,
                    f"Pitch:{pitch.round(1)[0]}",
                    (x_position_pose, y_position_pose),
                    POSE_FONT,
                    POSE_FONT_SCALE,
                    POSE_COLOR,
                    POSE_FONT_THICKNESS,
                    cv2.LINE_AA,
                )
                y_position_pose += 30
            if yaw is not None:
                cv2.putText(
                    frame,
                    f"Yaw:{yaw.round(1)[0]}",
                    (x_position_pose, y_position_pose),
                    POSE_FONT,
                    POSE_FONT_SCALE,
                    POSE_COLOR,
                    POSE_FONT_THICKNESS,
                    cv2.LINE_AA,
                )

        # Run crash detection (works regardless of face detection)
        crash_detected = False
        if crash_det is not None:
            crash_detected, motion_mag, frame = crash_det.detect_crash(frame, t_now)
            if crash_detected:
                alert_messages.append("CRASH DETECTED!")
                if sound:
                    sound.play(loops=0)

        e2 = cv2.getTickCount()
        proc_time_frame_ms = ((e2 - e1) / cv2.getTickFrequency()) * 1000

        # Display the frame
        cv2.imshow("Driver Drowsiness Detection - Press 'q' to quit", frame)

        # Check for quit command
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    return

if __name__ == "__main__":
    main()
