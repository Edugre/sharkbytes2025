<<<<<<< HEAD
# Person-Tracking Sentry - Complete Guide

## 📋 Table of Contents
1. [Quick Start](#quick-start)
2. [Project Structure](#project-structure)
3. [Module Architecture](#module-architecture)
4. [Configuration](#configuration)
5. [Keyboard Controls](#keyboard-controls)
6. [Tuning & Optimization](#tuning--optimization)
7. [Development Guide](#development-guide)
8. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Running the System
```bash
cd "c:\Users\Jorge Taban\Documents\sharkbytes2025"
python person_tracking_sentry.py
```

### Basic Controls
- **L** - Toggle auto-lock ON/OFF
- **C** - Center servos (when unlocked)
- **Q** - Quit

### Quick Configuration
Edit `config/settings.py` for all adjustable parameters.

---

## 📁 Project Structure

```
sharkbytes2025/
├── person_tracking_sentry.py    # Main entry point (launcher)
├── requirements.txt               # Python dependencies
├── README.md                      # This file - complete guide
├── CONTROLS.md                    # Detailed keyboard controls
├── TUNING_GUIDE.md                # Parameter optimization tips
│
├── config/                        # Configuration package
│   ├── __init__.py
│   └── settings.py                # All tunable parameters
│
├── src/                           # Source code package
│   ├── __init__.py
│   ├── detector.py                # YOLO person detection
│   ├── tracker.py                 # DeepSORT object tracking
│   ├── servo_controller.py        # PCA9685 servo control
│   ├── target_tracker.py          # Target locking logic
│   └── sentry.py                  # Main sentry system
│
└── utils/                         # Utility functions package
    ├── __init__.py
    └── ui_utils.py                # UI drawing functions
```

---

## �️ Module Architecture

### System Flow Diagram

```
person_tracking_sentry.py (Entry Point)
           │
           └─> PersonTrackingSentry
                      │
                      ├─> PersonDetector (YOLO)
                      ├─> ObjectTracker (DeepSORT)
                      ├─> ServoController (PCA9685)
                      ├─> TargetTracker (State Logic)
                      └─> UI Utils (Drawing)
```

### Data Flow

```
Camera Frame
     ↓
PersonDetector → Raw Detections [x1, y1, x2, y2, conf]
     ↓
ObjectTracker → Tracked Objects [{id, bbox}, ...]
     ↓
TargetTracker → Select/Lock Target ID
     ↓
PD Control → Calculate Servo Movements
     ↓
ServoController → Smooth Movement
     ↓
Hardware → Servos Move
```

### Module Responsibilities

| Module | File | Purpose |
|--------|------|---------|
| **PersonDetector** | `src/detector.py` | YOLO-based person detection |
| **ObjectTracker** | `src/tracker.py` | DeepSORT multi-object tracking |
| **ServoController** | `src/servo_controller.py` | PCA9685 hardware interface |
| **TargetTracker** | `src/target_tracker.py` | Target locking state machine |
| **PersonTrackingSentry** | `src/sentry.py` | Main system orchestration |
| **UI Utils** | `utils/ui_utils.py` | Drawing and visualization |
| **Settings** | `config/settings.py` | All configuration constants |

---

## ⚙️ Configuration

All settings are in `config/settings.py`. Key sections:

### Camera Settings
```python
CAMERA_WIDTH = 320        # Resolution for speed
CAMERA_HEIGHT = 320
TARGET_FPS = 30
```

### Servo Hardware
```python
PAN_MIN = 10             # Angle limits
PAN_MAX = 170
PAN_DEFAULT = 90

TILT_MIN = 20
TILT_MAX = 150
TILT_DEFAULT = 90
```

### Control Parameters (Smooth Motion)
```python
KP = 0.020               # Proportional gain (lower = smoother)
KD = 0.25                # Derivative gain (higher = less overshoot)
MAX_SERVO_STEP = 1.5     # Max degrees per frame
SMOOTHING_FACTOR = 0.3   # Exponential smoothing (0-1)
DEADBAND_X = 25          # Deadband zone in pixels
DEADBAND_Y = 25
```

### Detection & Tracking
```python
YOLO_MODEL = 'yolo11n.pt'
YOLO_CONFIDENCE = 0.35
TARGET_LOST_TIMEOUT = 2.0  # Seconds before unlocking
```

---

## 🎮 Keyboard Controls

### L - Lock Toggle

**When Auto-Lock is ON (default):**
- System automatically locks to first person detected
- Follows that person until they disappear for >2 seconds
- Press L to unlock and disable auto-locking

**When Auto-Lock is OFF:**
- System will NOT lock onto people
- Status shows: `MANUAL MODE (Lock OFF)`
- Press L to re-enable auto-locking

### C - Center Servos

**When Unlocked:**
- Centers servos to default position (90°, 90°)
- Resets sweep angle

**When Locked:**
- Blocked - must press L to unlock first

### Q - Quit

- Gracefully shuts down system
- Resets servos to center
- Releases camera
- Closes all windows

---

## 🔧 Tuning & Optimization

### For Smoother Motion
```python
# In config/settings.py
KP = 0.015               # Lower = smoother (from 0.020)
MAX_SERVO_STEP = 1.0     # Lower = smoother (from 1.5)
SMOOTHING_FACTOR = 0.2   # Lower = smoother (from 0.3)
KD = 0.30                # Higher = more damping (from 0.25)
```

### For Faster Response
```python
KP = 0.030               # Higher = faster (from 0.020)
MAX_SERVO_STEP = 2.5     # Higher = faster (from 1.5)
SMOOTHING_FACTOR = 0.4   # Higher = faster (from 0.3)
```

### For Better Performance (FPS)
```python
CAMERA_WIDTH = 320       # Lower resolution
CAMERA_HEIGHT = 240
YOLO_CONFIDENCE = 0.40   # Higher threshold = fewer detections
```

### Current Optimizations ✅
- Exponential smoothing for gimbal-like motion
- PD control prevents overshoot
- Step limiting prevents jerky movement
- Deadband zones reduce jitter
- YOLOv11n for speed and accuracy

---

## 👨‍💻 Development Guide

### Testing Individual Modules

**Test Detector:**
```python
from src.detector import PersonDetector
import cv2

detector = PersonDetector()
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
detections = detector.detect(frame)
print(f"Found {len(detections)} people")
```

**Test Servo Controller:**
```python
from src.servo_controller import ServoController

servo = ServoController()
servo.set_pan(90)
servo.set_tilt(90)
pan, tilt = servo.get_position()
print(f"Pan: {pan}°, Tilt: {tilt}°")
```

**Test Tracker:**
```python
from src.tracker import ObjectTracker
from src.detector import PersonDetector
import cv2

detector = PersonDetector()
tracker = ObjectTracker()
cap = cv2.VideoCapture(0)

ret, frame = cap.read()
detections = detector.detect(frame)
tracks = tracker.update(detections, frame)
print(f"Tracking {len(tracks)} objects")
```

### Adding New Features

**Example: Add a new movement pattern**
1. Edit `src/servo_controller.py`
2. Add new method:
```python
def spiral_pattern(self, radius=20):
    """Move in a spiral pattern."""
    # Implementation here
```
3. Call from `src/sentry.py` in the appropriate place

**Example: Change detection algorithm**
1. Edit `src/detector.py`
2. Keep the same interface (`detect()` method)
3. No changes needed elsewhere!

### Code Style Guidelines
- Use descriptive variable names
- Add docstrings to all functions/classes
- Keep functions focused (single responsibility)
- Use type hints where beneficial
- Comment complex logic

---

## 🛠️ Troubleshooting

### Import Errors
```
ImportError: No module named 'src.sentry'
```
**Solution:** Run from project root directory
```bash
cd "c:\Users\Jorge Taban\Documents\sharkbytes2025"
python person_tracking_sentry.py
```

### Dependencies Not Found
```
Import "ultralytics" could not be resolved
```
**Solution:** Install requirements
```bash
pip install -r requirements.txt
```

### Camera Not Opening
```
RuntimeError: Failed to open camera at /dev/video0
```
**Solutions:**
- Check camera is connected
- Try different `CAMERA_INDEX` in `config/settings.py`
- Verify camera permissions

### Servos Not Moving
**Check:**
1. PCA9685 connected to I2C bus
2. Correct I2C address in `config/settings.py`
3. Servo power supply connected
4. Channels configured correctly

### Low FPS
**Quick fixes:**
1. Reduce camera resolution in `config/settings.py`
2. Increase `YOLO_CONFIDENCE` threshold
3. Reduce `YOLO_MAX_DETECTIONS`
4. Enable GPU acceleration (if available)

### Jerky Movement
**Solutions:**
1. Lower `KP` gain
2. Reduce `MAX_SERVO_STEP`
3. Lower `SMOOTHING_FACTOR`
4. Increase `KD` gain
5. Widen deadband zones

---

## 📦 Dependencies

### Required Packages
```
opencv-python>=4.12.0
ultralytics>=8.3.0       # YOLOv11
deep-sort-realtime>=1.3.2
adafruit-circuitpython-servokit
numpy
```

### Installation
```bash
pip install -r requirements.txt
```

### Hardware Requirements
- Jetson device (or any system with camera)
- USB Camera (/dev/video0)
- PCA9685 Servo Driver (I2C)
- 2 servos (pan and tilt)

---

## 🌟 Features

✅ **Smooth Motion Control** - Exponential smoothing + PD control  
✅ **Automatic Tracking** - Locks onto first person detected  
✅ **Manual Override** - Full keyboard control  
✅ **Modular Design** - Easy to customize and extend  
✅ **Real-time Performance** - Optimized for 15-30 FPS  
✅ **YOLOv11** - Latest detection model for accuracy  
✅ **DeepSORT** - Robust multi-object tracking  
✅ **Visual Feedback** - On-screen status and bounding boxes  

---

## 📚 Additional Files

- **CONTROLS.md** - Detailed keyboard controls reference
- **TUNING_GUIDE.md** - Performance optimization guide  
- **VERSION_INFO.md** - Package versions and changelog
- **ARCHITECTURE.md** - Technical architecture diagrams

---

## 🏆 Benefits of Modular Structure

✅ **Separation of Concerns** - Each module has one clear responsibility  
✅ **Easy Testing** - Test modules independently  
✅ **Reusability** - Import modules in other projects  
✅ **Maintainability** - Changes are localized  
✅ **Readability** - Smaller, focused files  
✅ **Configuration** - All settings in one place  
✅ **Scalability** - Easy to add new features  

---

## 📄 License

See project documentation for license information.

## 👤 Author

AI Assistant - November 8, 2025

---

**For detailed technical architecture, see ARCHITECTURE.md**

=======
# sharkbytes2025
a
>>>>>>> 6f1256ca208da0238606f21a43c65b69f5b18379
