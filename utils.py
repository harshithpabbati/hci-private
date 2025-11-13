"""
Utility Functions Module

This module provides various utility functions for the driver drowsiness
detection system, including camera calibration, image processing, and
coordinate transformations.

Functions:
    load_camera_parameters: Load camera calibration from file
    resize: Resize image by percentage
    get_landmarks: Extract facial landmarks from MediaPipe results
    get_face_area: Calculate face bounding box area
    show_keypoints: Visualize facial keypoints
    midpoint: Calculate midpoint between two points
    get_array_keypoints: Convert landmarks to NumPy array
    rot_mat_to_euler: Convert rotation matrix to Euler angles
    draw_pose_info: Draw head pose visualization
"""

import json
import cv2
import numpy as np


def load_camera_parameters(file_path: str) -> tuple:
    """
    Load camera calibration parameters from a JSON file.
    
    Camera calibration improves accuracy of head pose estimation
    by accounting for camera distortion and intrinsic parameters.
    
    Args:
        file_path: Path to JSON file containing camera_matrix and dist_coeffs
        
    Returns:
        Tuple of (camera_matrix, distortion_coefficients) as NumPy arrays
        Returns (None, None) if loading fails
    """
    try:
        with open(file_path, "r") as file:
            if file_path.endswith(".json"):
                data = json.load(file)
            else:
                raise ValueError("Unsupported file format. Use JSON.")
            return (
                np.array(data["camera_matrix"], dtype="double"),
                np.array(data["dist_coeffs"], dtype="double"),
            )
    except Exception as e:
        print(f"Failed to load camera parameters: {e}")
        return None, None


def resize(frame: np.ndarray, scale_percent: int) -> np.ndarray:
    """
    Resize an image by a percentage scale factor.
    
    Args:
        frame: Input image
        scale_percent: Percentage to scale (e.g., 100 = original, 50 = half size)
        
    Returns:
        Resized image
    """
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_LINEAR)
    return resized


def get_landmarks(lms) -> np.ndarray:
    """
    Extract facial landmarks from MediaPipe face mesh results.
    
    If multiple faces are detected, returns the largest face.
    Clamps coordinates to [0, 1] range.
    
    Args:
        lms: MediaPipe face mesh landmarks (multi_face_landmarks)
        
    Returns:
        NumPy array of landmarks with shape (478, 3) containing (x, y, z)
    """
    surface = 0
    biggest_face = None
    
    for lms0 in lms:
        landmarks = [np.array([point.x, point.y, point.z]) for point in lms0.landmark]
        landmarks = np.array(landmarks)

        # Clamp coordinates to valid range [0, 1]
        landmarks[landmarks[:, 0] < 0.0, 0] = 0.0
        landmarks[landmarks[:, 0] > 1.0, 0] = 1.0
        landmarks[landmarks[:, 1] < 0.0, 1] = 0.0
        landmarks[landmarks[:, 1] > 1.0, 1] = 1.0

        # Calculate face area to find largest face
        dx = landmarks[:, 0].max() - landmarks[:, 0].min()
        dy = landmarks[:, 1].max() - landmarks[:, 1].min()
        new_surface = dx * dy
        
        if new_surface > surface:
            biggest_face = landmarks
            surface = new_surface

    return biggest_face


def get_face_area(face) -> float:
    """
    Calculate the area of a face bounding box.
    
    Args:
        face: Face detection object with left(), right(), top(), bottom() methods
        
    Returns:
        Area in pixels
    """
    return abs((face.left() - face.right()) * (face.bottom() - face.top()))


def show_keypoints(keypoints, frame: np.ndarray) -> np.ndarray:
    """
    Draw facial keypoints on a frame.
    
    Args:
        keypoints: Keypoints object with part() method
        frame: BGR image to draw on
        
    Returns:
        Frame with keypoints drawn
    """
    for n in range(0, 68):
        x = keypoints.part(n).x
        y = keypoints.part(n).y
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
    return frame


def midpoint(p1, p2) -> np.ndarray:
    """
    Calculate the midpoint between two points.
    
    Args:
        p1: First point with x and y attributes
        p2: Second point with x and y attributes
        
    Returns:
        NumPy array with midpoint coordinates [x, y]
    """
    return np.array([int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)])


def get_array_keypoints(landmarks, dtype: str = "int", verbose: bool = False) -> np.ndarray:
    """
    Convert facial landmarks to NumPy array format.
    
    Args:
        landmarks: Facial landmarks object
        dtype: Data type for array ("int" or "float")
        verbose: Print array if True
        
    Returns:
        NumPy array of shape (68, 2) with (x, y) coordinates
    """
    points_array = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        points_array[i] = (landmarks.part(i).x, landmarks.part(i).y)

    if verbose:
        print(points_array)

    return points_array


def rot_mat_to_euler(rmat: np.ndarray) -> np.ndarray:
    """
    Convert a rotation matrix to Euler angles (roll, pitch, yaw).
    
    This function handles gimbal lock situations and returns angles
    in degrees suitable for head pose estimation.
    
    Args:
        rmat: 3x3 rotation matrix
        
    Returns:
        NumPy array with [roll, pitch, yaw] in degrees
        Returns None if input is not a valid rotation matrix
    """
    # Verify this is a valid rotation matrix
    rtr = np.transpose(rmat)
    r_identity = np.matmul(rtr, rmat)

    I = np.identity(3, dtype=rmat.dtype)
    if np.linalg.norm(r_identity - I) < 1e-6:
        sy = (rmat[:2, 0] ** 2).sum() ** 0.5
        singular = sy < 1e-6

        if not singular:  # Normal case
            x = np.arctan2(rmat[2, 1], rmat[2, 2])
            y = np.arctan2(-rmat[2, 0], sy)
            z = np.arctan2(rmat[1, 0], rmat[0, 0])
        else:  # Gimbal lock case
            x = np.arctan2(-rmat[1, 2], rmat[1, 1])
            y = np.arctan2(-rmat[2, 0], sy)
            z = 0

        # Adjust angles to proper range
        if x > 0:
            x = np.pi - x
        else:
            x = -(np.pi + x)

        if z > 0:
            z = np.pi - z
        else:
            z = -(np.pi + z)

        return (np.array([x, y, z]) * 180.0 / np.pi).round(2)
    else:
        print("Input is not a valid rotation matrix")
        return None


def draw_pose_info(frame: np.ndarray, img_point: tuple, point_proj: np.ndarray,
                  roll: float = None, pitch: float = None, yaw: float = None) -> np.ndarray:
    """
    Draw head pose axes and angle information on frame.
    
    Args:
        frame: BGR image to draw on
        img_point: Nose tip point (x, y)
        point_proj: Projected points for 3D axes
        roll: Roll angle in degrees
        pitch: Pitch angle in degrees
        yaw: Yaw angle in degrees
        
    Returns:
        Frame with pose information drawn
    """
    # Draw 3D axes (RGB = XYZ)
    frame = cv2.line(
        frame, img_point, tuple(point_proj[0].ravel().astype(int)), (255, 0, 0), 3
    )
    frame = cv2.line(
        frame, img_point, tuple(point_proj[1].ravel().astype(int)), (0, 255, 0), 3
    )
    frame = cv2.line(
        frame, img_point, tuple(point_proj[2].ravel().astype(int)), (0, 0, 255), 3
    )

    # Draw angle text if provided
    if roll is not None and pitch is not None and yaw is not None:
        cv2.putText(
            frame,
            "Roll:" + str(round(roll, 0)),
            (500, 50),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Pitch:" + str(round(pitch, 0)),
            (500, 70),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Yaw:" + str(round(yaw, 0)),
            (500, 90),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame