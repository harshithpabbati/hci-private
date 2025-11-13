"""
Crash Detection Module

This module implements crash detection using optical flow analysis to detect
sudden deceleration or impact events. It monitors motion patterns in the video
feed and triggers alerts when abrupt changes indicate a potential crash.

Classes:
    CrashDetector: Main class for detecting sudden deceleration/crash events
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class CrashDetector:
    """
    Crash detector using optical flow analysis.
    
    This class monitors frame-to-frame motion using optical flow to detect
    sudden changes in motion that could indicate a crash or sudden stop.
    It tracks the magnitude and direction of motion vectors and triggers
    an alert when a sudden deceleration is detected.
    
    Attributes:
        motion_threshold (float): Threshold for detecting significant motion change
        crash_time_thresh (float): Time threshold for crash alert (seconds)
        smoothing_factor (float): Smoothing factor for motion magnitude (0-1)
        verbose (bool): Enable detailed debug output
    """
    
    def __init__(
        self,
        motion_threshold: float = 15.0,
        crash_time_thresh: float = 0.5,
        smoothing_factor: float = 0.7,
        verbose: bool = False
    ):
        """
        Initialize the crash detector with specified thresholds.
        
        Args:
            motion_threshold: Threshold for detecting significant motion change
            crash_time_thresh: Time before triggering crash alert (seconds)
            smoothing_factor: Smoothing factor for motion (0-1)
            verbose: Enable debug output
        """
        self.motion_threshold = motion_threshold
        self.crash_time_thresh = crash_time_thresh
        self.smoothing_factor = smoothing_factor
        self.verbose = verbose
        
        # Initialize optical flow parameters
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # State variables
        self.prev_gray = None
        self.prev_points = None
        self.prev_motion_magnitude = 0.0
        self.smoothed_motion = 0.0
        self.crash_accumulator = 0.0
        self.last_update_time = None
        
    def reset(self):
        """Reset the detector state."""
        self.prev_gray = None
        self.prev_points = None
        self.prev_motion_magnitude = 0.0
        self.smoothed_motion = 0.0
        self.crash_accumulator = 0.0
        self.last_update_time = None
        
    def detect_crash(
        self, 
        frame: np.ndarray, 
        t_now: float
    ) -> Tuple[bool, float, Optional[np.ndarray]]:
        """
        Detect sudden deceleration/crash using optical flow.
        
        Args:
            frame: Current BGR frame
            t_now: Current timestamp
            
        Returns:
            Tuple of (crash_detected, motion_magnitude, visualization_frame)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        crash_detected = False
        motion_magnitude = 0.0
        vis_frame = frame.copy()
        
        # Initialize on first frame
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            self.last_update_time = t_now
            return False, 0.0, vis_frame
        
        # Calculate time elapsed
        elapsed = t_now - self.last_update_time if self.last_update_time else 0.0
        self.last_update_time = t_now
        
        # Get feature points if needed
        if self.prev_points is None or len(self.prev_points) < 10:
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            
        if self.prev_points is not None and len(self.prev_points) > 0:
            # Calculate optical flow
            curr_points, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_points, None, **self.lk_params
            )
            
            if curr_points is not None:
                # Select good points
                good_prev = self.prev_points[status == 1]
                good_curr = curr_points[status == 1]
                
                if len(good_prev) > 5:
                    # Calculate motion vectors
                    motion_vectors = good_curr - good_prev
                    motion_magnitudes = np.sqrt(
                        motion_vectors[:, 0]**2 + motion_vectors[:, 1]**2
                    )
                    
                    # Average motion magnitude
                    motion_magnitude = np.mean(motion_magnitudes)
                    
                    # Smooth the motion
                    self.smoothed_motion = (
                        self.smoothing_factor * self.smoothed_motion +
                        (1 - self.smoothing_factor) * motion_magnitude
                    )
                    
                    # Detect sudden deceleration (sudden drop in motion)
                    motion_change = self.prev_motion_magnitude - motion_magnitude
                    
                    # If there was significant motion and it suddenly decreased
                    if (self.prev_motion_magnitude > 5.0 and 
                        motion_change > self.motion_threshold):
                        self.crash_accumulator += elapsed
                    else:
                        # Decay the accumulator
                        self.crash_accumulator = max(0, self.crash_accumulator - elapsed * 2)
                    
                    # Trigger crash alert
                    if self.crash_accumulator >= self.crash_time_thresh:
                        crash_detected = True
                    
                    # Draw motion vectors on visualization frame
                    for i, (prev_pt, curr_pt) in enumerate(zip(good_prev, good_curr)):
                        a, b = prev_pt.ravel()
                        c, d = curr_pt.ravel()
                        
                        # Draw motion vector
                        color = (0, 255, 0) if not crash_detected else (0, 0, 255)
                        cv2.line(vis_frame, (int(a), int(b)), (int(c), int(d)), 
                                color, 2)
                        cv2.circle(vis_frame, (int(c), int(d)), 3, color, -1)
                    
                    # Draw motion info
                    cv2.putText(vis_frame, f"Motion: {motion_magnitude:.1f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(vis_frame, f"Smoothed: {self.smoothed_motion:.1f}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    if crash_detected:
                        cv2.putText(vis_frame, "CRASH DETECTED!", 
                                   (10, frame.shape[0] - 30),
                                   cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
                    
                    # Update state
                    self.prev_motion_magnitude = motion_magnitude
                    self.prev_points = good_curr.reshape(-1, 1, 2)
                    
                    if self.verbose:
                        print(f"Motion: {motion_magnitude:.2f} | "
                              f"Smoothed: {self.smoothed_motion:.2f} | "
                              f"Change: {motion_change:.2f} | "
                              f"Accumulator: {self.crash_accumulator:.2f}")
        
        # Update previous frame
        self.prev_gray = gray
        
        return crash_detected, motion_magnitude, vis_frame
