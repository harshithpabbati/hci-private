"""
Yawn Detection Module

This module implements yawn detection by monitoring mouth aspect ratio (MAR).
Frequent yawning is a strong indicator of driver fatigue and drowsiness.

Classes:
    YawnDetector: Main class for detecting yawning behavior
"""

import numpy as np
from typing import Tuple


class YawnDetector:
    """
    Yawn detector using Mouth Aspect Ratio (MAR).
    
    This class monitors the mouth opening by calculating the mouth aspect ratio
    from facial landmarks. Prolonged mouth opening indicates yawning, which is
    a strong fatigue indicator.
    
    Attributes:
        mar_thresh (float): Threshold for detecting mouth opening (yawn)
        yawn_time_thresh (float): Time threshold for yawn confirmation (seconds)
        verbose (bool): Enable detailed debug output
    """
    
    # Landmark indices for mouth (MediaPipe face mesh)
    MOUTH_OUTER_TOP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
    MOUTH_OUTER_BOTTOM = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
    MOUTH_INNER_TOP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    MOUTH_INNER_BOTTOM = [95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
    
    def __init__(
        self,
        mar_thresh: float = 0.6,
        yawn_time_thresh: float = 1.5,
        verbose: bool = False
    ):
        """
        Initialize the yawn detector with specified thresholds.
        
        Args:
            mar_thresh: Mouth Aspect Ratio threshold for yawn detection
            yawn_time_thresh: Time before confirming yawn (seconds)
            verbose: Enable debug output
        """
        self.mar_thresh = mar_thresh
        self.yawn_time_thresh = yawn_time_thresh
        self.verbose = verbose
        
        # State variables
        self.yawn_accumulator = 0.0
        self.yawn_count = 0
        self.last_update_time = None
        
    def calculate_MAR(self, landmarks: np.ndarray) -> float:
        """
        Calculate Mouth Aspect Ratio (MAR).
        
        The MAR is calculated using vertical and horizontal distances
        of mouth landmarks:
        MAR = (vertical_distance_1 + vertical_distance_2) / (2 * horizontal_distance)
        
        Args:
            landmarks: Facial landmarks array (478 landmarks, normalized)
            
        Returns:
            Mouth Aspect Ratio value
        """
        # Extract key mouth points
        # Top center
        top_center = landmarks[13]
        # Bottom center
        bottom_center = landmarks[14]
        # Left corner
        left_corner = landmarks[61]
        # Right corner
        right_corner = landmarks[291]
        
        # Calculate vertical distance (mouth height)
        vertical_dist = np.linalg.norm(top_center - bottom_center)
        
        # Calculate horizontal distance (mouth width)
        horizontal_dist = np.linalg.norm(left_corner - right_corner)
        
        # Calculate MAR
        if horizontal_dist > 0:
            mar = vertical_dist / horizontal_dist
        else:
            mar = 0.0
            
        return mar
    
    def detect_yawn(
        self,
        landmarks: np.ndarray,
        t_now: float
    ) -> Tuple[bool, float, int]:
        """
        Detect yawning behavior.
        
        Args:
            landmarks: Facial landmarks array
            t_now: Current timestamp
            
        Returns:
            Tuple of (is_yawning, mar_value, total_yawn_count)
        """
        # Initialize on first call
        if self.last_update_time is None:
            self.last_update_time = t_now
            
        elapsed = t_now - self.last_update_time
        self.last_update_time = t_now
        
        # Calculate MAR
        mar = self.calculate_MAR(landmarks)
        
        # Check if mouth is open (potential yawn)
        is_yawning = False
        if mar > self.mar_thresh:
            self.yawn_accumulator += elapsed
            
            # Confirm yawn if mouth stays open long enough
            if self.yawn_accumulator >= self.yawn_time_thresh:
                is_yawning = True
        else:
            # Reset accumulator if mouth closes
            if self.yawn_accumulator >= self.yawn_time_thresh:
                # Yawn completed, increment counter
                self.yawn_count += 1
            self.yawn_accumulator = 0.0
        
        if self.verbose:
            print(f"MAR: {mar:.3f} | Yawn Acc: {self.yawn_accumulator:.2f}s | "
                  f"Yawn Count: {self.yawn_count}")
        
        return is_yawning, mar, self.yawn_count
    
    def reset(self):
        """Reset the detector state."""
        self.yawn_accumulator = 0.0
        self.yawn_count = 0
        self.last_update_time = None
