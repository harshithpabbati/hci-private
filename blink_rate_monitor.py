"""
Blink Rate Monitor Module

This module monitors blink rate and patterns to detect driver fatigue.
Abnormal blink rates (too frequent or too infrequent) can indicate drowsiness.

Classes:
    BlinkRateMonitor: Main class for monitoring blink patterns
"""

import numpy as np
from collections import deque
from typing import Tuple


class BlinkRateMonitor:
    """
    Blink rate monitor for fatigue detection.
    
    This class monitors the frequency and pattern of eye blinks. Normal blink
    rate is 15-20 blinks per minute. Abnormally high or low rates can indicate
    fatigue or drowsiness.
    
    Attributes:
        ear_blink_thresh (float): EAR threshold for detecting a blink
        min_blink_duration (float): Minimum blink duration (seconds)
        max_blink_duration (float): Maximum blink duration (seconds)
        window_size (float): Time window for rate calculation (seconds)
        verbose (bool): Enable detailed debug output
    """
    
    def __init__(
        self,
        ear_blink_thresh: float = 0.2,
        min_blink_duration: float = 0.1,
        max_blink_duration: float = 0.4,
        window_size: float = 60.0,
        verbose: bool = False
    ):
        """
        Initialize the blink rate monitor.
        
        Args:
            ear_blink_thresh: EAR threshold for blink detection
            min_blink_duration: Minimum valid blink duration
            max_blink_duration: Maximum valid blink duration
            window_size: Time window for rate calculation (seconds)
            verbose: Enable debug output
        """
        self.ear_blink_thresh = ear_blink_thresh
        self.min_blink_duration = min_blink_duration
        self.max_blink_duration = max_blink_duration
        self.window_size = window_size
        self.verbose = verbose
        
        # State variables
        self.blink_timestamps = deque()
        self.eyes_closed = False
        self.blink_start_time = None
        self.total_blinks = 0
        
        # Normal blink rate range (blinks per minute)
        self.normal_min_rate = 12
        self.normal_max_rate = 25
        
    def update(self, ear: float, t_now: float) -> Tuple[float, bool, int]:
        """
        Update blink detection and calculate current blink rate.
        
        Args:
            ear: Current Eye Aspect Ratio value
            t_now: Current timestamp
            
        Returns:
            Tuple of (blink_rate, abnormal_rate, total_blinks)
        """
        # Detect blink start
        if ear <= self.ear_blink_thresh and not self.eyes_closed:
            self.eyes_closed = True
            self.blink_start_time = t_now
            
        # Detect blink end
        elif ear > self.ear_blink_thresh and self.eyes_closed:
            self.eyes_closed = False
            
            if self.blink_start_time is not None:
                blink_duration = t_now - self.blink_start_time
                
                # Validate blink duration
                if self.min_blink_duration <= blink_duration <= self.max_blink_duration:
                    # Valid blink detected
                    self.blink_timestamps.append(t_now)
                    self.total_blinks += 1
                    
                    if self.verbose:
                        print(f"Blink detected! Duration: {blink_duration:.3f}s")
                
                self.blink_start_time = None
        
        # Remove old blink timestamps outside the window
        while self.blink_timestamps and self.blink_timestamps[0] < t_now - self.window_size:
            self.blink_timestamps.popleft()
        
        # Calculate blink rate (blinks per minute)
        if len(self.blink_timestamps) > 0:
            time_span = min(t_now - self.blink_timestamps[0], self.window_size)
            if time_span > 0:
                blink_rate = (len(self.blink_timestamps) / time_span) * 60.0
            else:
                blink_rate = 0.0
        else:
            blink_rate = 0.0
        
        # Check if rate is abnormal
        abnormal = False
        if len(self.blink_timestamps) >= 3:  # Need at least 3 blinks for reliable rate
            if blink_rate < self.normal_min_rate or blink_rate > self.normal_max_rate:
                abnormal = True
        
        if self.verbose and len(self.blink_timestamps) > 0:
            print(f"Blink Rate: {blink_rate:.1f} bpm | "
                  f"Count in window: {len(self.blink_timestamps)} | "
                  f"Total: {self.total_blinks}")
        
        return blink_rate, abnormal, self.total_blinks
    
    def reset(self):
        """Reset the monitor state."""
        self.blink_timestamps.clear()
        self.eyes_closed = False
        self.blink_start_time = None
        self.total_blinks = 0
