# 🚘 Driver Drowsiness Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.11-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange.svg)](https://mediapipe.dev/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.44-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A real-time AI-powered driver safety monitoring system that detects drowsiness, distraction, and poor posture to prevent accidents. Built with computer vision and deep learning technologies.

## 📋 Table of Contents
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### 🎯 Core Detection Capabilities

#### **Face & Eye Monitoring (Front View)**
- **Eye Aspect Ratio (EAR) Detection**: Monitors eye closure to detect drowsiness
- **PERCLOS Score**: Percentage of Eye Closure - industry-standard drowsiness metric
- **Gaze Tracking**: Detects when driver is looking away from the road
- **Iris Landmark Detection**: Precise pupil tracking for accurate gaze estimation
- **Real-time Eye Processing**: Optional visualization of eye region analysis

#### **Head Pose Estimation**
- **3D Head Orientation**: Tracks roll, pitch, and yaw angles
- **Distraction Detection**: Identifies when the driver's head is turned away
- **Visual Axis Display**: Optional 3D axis overlay showing head orientation
- **Camera Calibration Support**: Uses custom camera parameters for improved accuracy

#### **Posture Analysis (Side View)**
- **Back Posture Classification**: Detects reclined or slouched positions
- **Arm Extension Monitoring**: Identifies overextended or improper arm positions
- **MediaPipe Pose Integration**: Full-body landmark detection
- **Real-time Posture Feedback**: Visual indicators for correct/incorrect posture

#### **Alert System**
- **Multi-level Alerts**:
  - 🟡 **DROWSY**: PERCLOS threshold exceeded
  - 🔴 **ASLEEP**: Extended eye closure detected
  - 🟠 **LOOK AWAY**: Gaze diverted from road
  - 🟣 **DISTRACTED**: Head pose threshold exceeded
- **Audio Alarms**: Pygame-based sound alerts for critical conditions
- **Screenshot Capture**: Automatic screenshot when alerts are triggered
- **Alert Gallery**: Historical view of all captured alert events

### 🖥️ User Interfaces

#### **CLI Mode** (`main.py`)
- Lightweight OpenCV-based window
- Real-time video feed with overlays
- Minimal resource usage
- Perfect for testing and development
- Keyboard controls (press 'q' to quit)

#### **Streamlit Web App** (`app.py`)
- Modern, responsive web interface
- Multi-view monitoring:
  - **Front View**: Face and eye analysis
  - **Side View**: Posture detection
  - **Alerts Gallery**: Historical alert screenshots
- Live FPS counter
- Real-time metric displays
- Start/stop camera controls
- Screenshot management

### 🔧 Advanced Features
- **Multiple Camera Support**: Configure different camera sources
- **Custom Thresholds**: Adjustable sensitivity for all detection metrics
- **Verbose Logging**: Detailed console output for debugging
- **Smooth Tracking**: Configurable smoothing factors for stable detection
- **Frame Processing Optimization**: OpenCV hardware acceleration support

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Input Sources                            │
│  (Webcam / External Camera / Video File)                     │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│              MediaPipe Face Mesh                             │
│  • 478 Facial Landmarks (468 face + 10 iris)                │
│  • Real-time landmark detection                              │
└────────────────┬────────────────────────────────────────────┘
                 │
      ┌──────────┴──────────┬─────────────────┐
      ▼                     ▼                  ▼
┌─────────────┐   ┌─────────────────┐   ┌──────────────┐
│ Eye Detector│   │ Head Pose       │   │ Posture      │
│  • EAR      │   │ Estimator       │   │ Classifier   │
│  • Gaze     │   │ • Roll/Pitch/Yaw│   │ • Pose Model │
└──────┬──────┘   └────────┬────────┘   └──────┬───────┘
       │                   │                    │
       └───────────┬───────┴──────────┬─────────┘
                   ▼                  ▼
           ┌────────────────┐  ┌─────────────┐
           │ Attention      │  │ Alert       │
           │ Scorer         │  │ System      │
           └────────┬───────┘  └──────┬──────┘
                    │                 │
                    ▼                 ▼
           ┌─────────────────────────────┐
           │    UI Layer (CLI/Web)       │
           │  • Real-time Display        │
           │  • Metrics & Visualization  │
           └─────────────────────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam or external camera
- (Optional) GPU for faster processing

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/harshithpabbati/hci-private.git
cd hci-private
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python main.py --help
```

## 🚀 Usage

### CLI Mode (Recommended for Testing)

**Basic usage with default settings:**
```bash
python main.py
```

**With custom camera:**
```bash
python main.py --camera 1
```

**With camera calibration file:**
```bash
python main.py --camera_params assets/camera_params.json
```

**With custom thresholds:**
```bash
python main.py --ear_thresh 0.2 --gaze_thresh 0.02 --pose_time_thresh 3.0
```

**Enable debug visualization:**
```bash
python main.py --show_eye_proc True --show_axis True --verbose True
```

### Streamlit Web Interface

**Launch the web app:**
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

**Features:**
- Use the sidebar to switch between Front View, Side View, and Alerts Gallery
- Click "Start Camera" to begin monitoring
- Click "Stop Camera" to end the session
- Screenshots are automatically saved to `screenshots/` directory
- View all captured alerts in the Gallery tab

## ⚙️ Configuration

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--camera` | int | 0 | Camera device number (0 for webcam) |
| `--camera_params` | str | None | Path to camera calibration JSON file |
| `--ear_thresh` | float | 0.15 | Eye Aspect Ratio threshold for drowsiness |
| `--ear_time_thresh` | float | 2.0 | Seconds of low EAR before "ASLEEP" alert |
| `--gaze_thresh` | float | 0.015 | Gaze deviation threshold |
| `--gaze_time_thresh` | float | 2.0 | Seconds of gaze deviation before alert |
| `--roll_thresh` | float | 20.0 | Head roll angle threshold (degrees) |
| `--pitch_thresh` | float | 20.0 | Head pitch angle threshold (degrees) |
| `--yaw_thresh` | float | 20.0 | Head yaw angle threshold (degrees) |
| `--pose_time_thresh` | float | 2.5 | Seconds of pose threshold before alert |
| `--show_fps` | bool | True | Display FPS counter |
| `--show_proc_time` | bool | True | Display processing time |
| `--show_eye_proc` | bool | False | Show eye processing windows |
| `--show_axis` | bool | True | Show head pose 3D axis |
| `--verbose` | bool | False | Print detailed debug information |

### Camera Calibration

For improved accuracy, you can provide camera calibration parameters:

**Example `camera_params.json`:**
```json
{
  "camera_matrix": [
    [800, 0, 320],
    [0, 800, 240],
    [0, 0, 1]
  ],
  "dist_coeffs": [0, 0, 0, 0, 0]
}
```

## 🔬 How It Works

### Eye Aspect Ratio (EAR)
The EAR is calculated using eye landmark distances:
```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```
Where p1-p6 are eye landmarks. Lower EAR values indicate closed eyes.

### PERCLOS (Percentage of Eye Closure)
Industry-standard drowsiness metric that measures the percentage of time eyes are closed over a rolling time window (default: 60 seconds).

### Head Pose Estimation
Uses solvePnP algorithm to estimate 3D head orientation from 2D facial landmarks:
- **Roll**: Head tilt left/right
- **Pitch**: Head nod up/down  
- **Yaw**: Head turn left/right

### Attention Scoring
Combines multiple metrics with time-based smoothing:
- Accumulates time when conditions are met (eyes closed, looking away, etc.)
- Applies decay factor when conditions are not met
- Triggers alerts when accumulated time exceeds thresholds

## 📁 Project Structure

```
hci-private/
├── app.py                  # Streamlit web application
├── main.py                 # CLI application
├── arg_parser.py          # Command-line argument parser
├── attention_scorer.py    # Attention/drowsiness scoring logic
├── eye_detector.py        # Eye detection and EAR calculation
├── pose_estimation.py     # Head pose estimation
├── posture.py             # Body posture classification
├── face_geometry.py       # 3D face geometry utilities
├── metric_landmarks.py    # Facial landmark metrics
├── utils.py               # Utility functions
├── requirements.txt       # Python dependencies
├── assets/
│   ├── alarm.mp3         # Alert sound file
│   └── camera_params.json # Camera calibration parameters
├── screenshots/           # Auto-generated alert screenshots
└── README.md             # This file
```

## 📋 Requirements

### Core Dependencies
- **opencv-contrib-python** (4.11.0.86): Computer vision operations
- **opencv-python** (4.11.0.86): Core OpenCV library
- **mediapipe**: Face mesh and pose detection
- **numpy** (1.26.4): Numerical computing
- **pygame**: Audio alert system
- **streamlit** (1.44.1): Web interface framework

### Additional Dependencies
- **pillow**: Image processing
- **pandas** (2.2.3): Data manipulation
- **matplotlib**: Visualization (optional)
- **scipy**: Scientific computing

See `requirements.txt` for complete list with version numbers.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe** by Google for facial landmark detection
- **OpenCV** community for computer vision tools
- **Streamlit** team for the web framework
- Research papers on drowsiness detection and driver safety

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**⚠️ Disclaimer**: This system is designed as an educational and research tool. It should not be used as the sole safety system in vehicles. Always follow proper driving safety guidelines and regulations.
