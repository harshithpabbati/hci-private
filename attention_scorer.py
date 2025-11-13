"""
Attention Scoring Module

This module implements the attention scoring system for driver drowsiness detection.
It tracks multiple metrics over time and triggers alerts when thresholds are exceeded.

Classes:
    AttentionScorer: Main class for scoring driver attention and alertness
"""

import numpy as np


class AttentionScorer:
    """
    Attention scorer for driver drowsiness and distraction detection.
    
    This class tracks multiple metrics including eye closure (EAR), gaze direction,
    and head pose over time. It uses time-based accumulation with decay to smooth
    out momentary fluctuations and provide stable alert triggers.
    
    Attributes:
        ear_thresh (float): Eye Aspect Ratio threshold for detecting closed eyes
        gaze_thresh (float): Gaze deviation threshold for detecting looking away
        perclos_thresh (float): PERCLOS threshold for tiredness detection
        roll_thresh (float): Head roll angle threshold (degrees)
        pitch_thresh (float): Head pitch angle threshold (degrees)
        yaw_thresh (float): Head yaw angle threshold (degrees)
        ear_time_thresh (float): Time threshold for asleep alert (seconds)
        gaze_time_thresh (float): Time threshold for looking away alert (seconds)
        pose_time_thresh (float): Time threshold for distraction alert (seconds)
        decay_factor (float): Decay factor for smoothing metrics (0-1)
        verbose (bool): Enable detailed debug output
    """

    def __init__(
        self,
        t_now: float,
        ear_thresh: float,
        gaze_thresh: float,
        perclos_thresh: float = 0.2,
        roll_thresh: float = 60,
        pitch_thresh: float = 20,
        yaw_thresh: float = 30,
        ear_time_thresh: float = 4.0,
        gaze_time_thresh: float = 2.0,
        pose_time_thresh: float = 4.0,
        decay_factor: float = 0.9,
        verbose: bool = False,
    ):
        """
        Initialize the attention scorer with specified thresholds.
        
        Args:
            t_now: Current timestamp
            ear_thresh: Eye Aspect Ratio threshold
            gaze_thresh: Gaze deviation threshold
            perclos_thresh: PERCLOS percentage threshold (default: 0.2 = 20%)
            roll_thresh: Head roll angle threshold in degrees
            pitch_thresh: Head pitch angle threshold in degrees
            yaw_thresh: Head yaw angle threshold in degrees
            ear_time_thresh: Time before triggering asleep alert
            gaze_time_thresh: Time before triggering looking away alert
            pose_time_thresh: Time before triggering distraction alert
            decay_factor: Smoothing decay factor (0-1)
            verbose: Enable debug output
        """
        # Thresholds and configuration
        self.ear_thresh = ear_thresh
        self.gaze_thresh = gaze_thresh
        self.perclos_thresh = perclos_thresh
        self.roll_thresh = roll_thresh
        self.pitch_thresh = pitch_thresh
        self.yaw_thresh = yaw_thresh
        self.ear_time_thresh = ear_time_thresh
        self.gaze_time_thresh = gaze_time_thresh
        self.pose_time_thresh = pose_time_thresh
        self.decay_factor = decay_factor
        self.verbose = verbose

        # Initialize timers for smoothing the metrics
        self.last_eval_time = t_now
        self.closure_time = 0.0
        self.not_look_ahead_time = 0.0
        self.distracted_time = 0.0

        # PERCLOS parameters
        self.PERCLOS_TIME_PERIOD = 60
        self.timestamps = np.empty((0,), dtype=np.float64)
        self.closed_flags = np.empty((0,), dtype=bool)
        self.eye_closure_counter = 0
        self.prev_time = t_now

    def _update_metric(self, metric_value: float, condition: bool, elapsed: float) -> float:
        """
        Update a metric value based on condition and elapsed time.
        
        If condition is True, accumulate time. Otherwise, apply decay.
        
        Args:
            metric_value: Current metric value
            condition: Whether the condition is met
            elapsed: Time elapsed since last update
            
        Returns:
            Updated metric value
        """
        if condition:
            return metric_value + elapsed
        else:
            return metric_value * self.decay_factor

    def eval_scores(
        self, t_now: float, ear_score: float, gaze_score: float, 
        head_roll: np.ndarray, head_pitch: np.ndarray, head_yaw: np.ndarray
    ) -> tuple:
        """
        Evaluate all scores and determine driver state.
        
        Args:
            t_now: Current timestamp
            ear_score: Eye Aspect Ratio score
            gaze_score: Gaze deviation score
            head_roll: Head roll angle (degrees)
            head_pitch: Head pitch angle (degrees)
            head_yaw: Head yaw angle (degrees)
            
        Returns:
            Tuple of (asleep, looking_away, distracted) boolean flags
        """
        # Calculate the time elapsed since the last evaluation
        elapsed = t_now - self.last_eval_time
        self.last_eval_time = t_now

        # Update the eye closure metric
        self.closure_time = self._update_metric(
            self.closure_time,
            (ear_score is not None and ear_score <= self.ear_thresh),
            elapsed,
        )

        # Update the gaze metric
        self.not_look_ahead_time = self._update_metric(
            self.not_look_ahead_time,
            (gaze_score is not None and gaze_score > self.gaze_thresh),
            elapsed,
        )

        # Update the head pose metric: check if any head angle exceeds its threshold
        head_condition = (
            (head_roll is not None and abs(head_roll) > self.roll_thresh)
            or (head_pitch is not None and abs(head_pitch) > self.pitch_thresh)
            or (head_yaw is not None and abs(head_yaw) > self.yaw_thresh)
        )
        self.distracted_time = self._update_metric(
            self.distracted_time, head_condition, elapsed
        )

        # Determine driver state based on thresholds
        asleep = self.closure_time >= self.ear_time_thresh
        looking_away = self.not_look_ahead_time >= self.gaze_time_thresh
        distracted = self.distracted_time >= self.pose_time_thresh

        if self.verbose:
            print(
                f"Closure Time: {self.closure_time:.2f}s | "
                f"Not Look Ahead Time: {self.not_look_ahead_time:.2f}s | "
                f"Distracted Time: {self.distracted_time:.2f}s"
            )

        return asleep, looking_away, distracted

    def get_PERCLOS(self, t_now: float, fps: float, ear_score: float) -> tuple:
        """
        Calculate PERCLOS (Percentage of Eye Closure) using frame-based method.
        
        This method counts frames where eyes are closed over a fixed time period.
        
        Args:
            t_now: Current timestamp
            fps: Current frames per second
            ear_score: Eye Aspect Ratio score
            
        Returns:
            Tuple of (tired, perclos_score)
        """
        delta = t_now - self.prev_time  # set delta timer
        tired = False  # set default value for the tired state of the driver

        all_frames_numbers_in_perclos_duration = int(self.PERCLOS_TIME_PERIOD * fps)

        # if the ear_score is lower or equal than the threshold, increase the eye_closure_counter
        if (ear_score is not None) and (ear_score <= self.ear_thresh):
            self.eye_closure_counter += 1

        # compute the PERCLOS over a given time period
        perclos_score = (
            self.eye_closure_counter
        ) / all_frames_numbers_in_perclos_duration

        if (
            perclos_score >= self.perclos_thresh
        ):  # if the PERCLOS score is higher than a threshold, tired = True
            tired = True

        if (
            delta >= self.PERCLOS_TIME_PERIOD
        ):  # at every end of the given time period, reset the counter and the timer
            self.eye_closure_counter = 0
            self.prev_time = t_now

        return tired, perclos_score

    def get_rolling_PERCLOS(self, t_now: float, ear_score: float) -> tuple:
        """
        Calculate rolling PERCLOS using a sliding time window.
        
        This is the preferred method as it provides continuous measurement
        without discrete resets. The window slides continuously over time.
        
        Args:
            t_now: Current timestamp
            ear_score: Eye Aspect Ratio score
            
        Returns:
            Tuple of (tired, perclos_score)
        """
        # Determine if the current frame indicates closed eyes
        eye_closed = (ear_score is not None) and (ear_score <= self.ear_thresh)

        # Append new values to the NumPy arrays. (np.concatenate creates new arrays.)
        self.timestamps = np.concatenate((self.timestamps, [t_now]))
        self.closed_flags = np.concatenate((self.closed_flags, [eye_closed]))

        # Create a boolean mask of entries within the rolling window.
        valid_mask = self.timestamps >= (t_now - self.PERCLOS_TIME_PERIOD)
        self.timestamps = self.timestamps[valid_mask]
        self.closed_flags = self.closed_flags[valid_mask]

        total_frames = self.timestamps.size
        if total_frames > 0:
            perclos_score = np.sum(self.closed_flags) / total_frames
        else:
            perclos_score = 0.0

        tired = perclos_score >= self.perclos_thresh
        return tired, perclos_score
