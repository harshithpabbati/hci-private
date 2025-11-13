"""
Posture Classification Module

This module provides posture classification for driver monitoring.
It uses MediaPipe Pose landmarks to analyze body positioning and
detect poor posture or unsafe sitting positions.

Functions:
    calculate_angle: Calculate angle between three body landmarks
    classify_posture: Classify driver posture as correct or problematic
"""

import numpy as np
import mediapipe as mp

# Initialize pose landmarks
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c) -> float:
    """
    Calculate the angle at point b given three points.
    
    This is used for analyzing back posture by calculating angles
    between shoulder, hip, and knee landmarks.
    
    Args:
        a: First landmark (e.g., shoulder)
        b: Middle landmark (e.g., hip) - the angle is measured here
        c: Third landmark (e.g., knee)
        
    Returns:
        Angle in degrees (0-180)
    """
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle


def classify_posture(landmarks) -> tuple:
    """
    Classify driver posture from side-view using body landmarks.
    
    Analyzes back angle and arm extension to determine if the driver
    is sitting in a safe, upright position or exhibiting poor posture.
    
    Args:
        landmarks: MediaPipe pose landmarks
        
    Returns:
        Tuple of (posture_label, color) for display purposes
        - posture_label: String describing the posture
        - color: BGR color tuple for visualization
    """
    try:
        # Extract key body landmarks
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

        # Calculate back posture angle (shoulder-hip-knee)
        back_angle = calculate_angle(left_shoulder, left_hip, left_knee)

        # Measure arm extension (distance between elbow and wrist)
        arm_extension = abs(left_elbow.x - left_wrist.x)

        # Classify posture based on thresholds
        if back_angle < 140:
            # Driver is reclined or slouching
            return "Reclined Posture", (0, 0, 255)  # Red
        elif arm_extension > 0.25:
            # Arm is overextended (possibly reaching or improper position)
            return "Overextended Arm", (0, 0, 255)  # Red
        else:
            # Posture is within acceptable range
            return "Right Posture", (0, 255, 0)  # Green
    except Exception as e:
        # Landmark detection failed
        return "No person detected", (255, 255, 255)  # White