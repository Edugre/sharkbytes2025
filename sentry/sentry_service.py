#!/usr/bin/env python3
"""
Sentry Service - Refactored PersonTrackingSentry for use as a background service.
This version doesn't use cv2.imshow() and is designed to be controlled via API.
"""

import cv2
import numpy as np
import time
import threading
from queue import Queue
from typing import Optional, Dict, Any
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Import configurations from original sentry
import sys
import os
sys.path.append(os.path.dirname(__file__))

# Try to import servo controller (will fail on non-Jetson systems)
try:
    from adafruit_servokit import ServoKit
    SERVOS_AVAILABLE = True
except ImportError:
    SERVOS_AVAILABLE = False
    print("[WARN] ServoKit not available - running in simulation mode")


# ========================================
# Configuration Constants
# ========================================

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 320
TARGET_FPS = 30

# Servo settings
PAN_MIN = 10
PAN_MAX = 170
PAN_DEFAULT = 90
TILT_MIN = 20
TILT_MAX = 150
TILT_DEFAULT = 90

# Control parameters
KP = 0.035
MAX_SERVO_STEP = 3.0
DEADBAND_X = 25
DEADBAND_Y = 25
PAN_INVERT = -1
TILT_INVERT = -1

# Tracking parameters
TARGET_LOST_TIMEOUT = 2.0
PERSON_CLASS_ID = 0

# Performance optimization
DETECTION_SKIP_FRAMES = 3  # Run YOLO every N frames (1=every frame, 2=every other, 3=every third)
YOLO_IMGSZ = 160  # Reduced from 320 for faster inference on Jetson
TRACKING_UPDATE_SKIP = 3  # Run DeepSORT embedding every N frames (major bottleneck!)

# Face detection parameters
FACE_PRIORITY = True  # Prioritize face tracking over body tracking
FACE_SCALE_FACTOR = 1.2  # Increased for faster detection (was 1.1)
FACE_MIN_NEIGHBORS = 4  # Reduced for faster detection (was 5)
FACE_MIN_SIZE = (40, 40)  # Increased minimum size for faster detection
FACE_DETECTION_SKIP_FRAMES = 5  # Run face detection every N frames when target locked


# ========================================
# Servo Controller (with simulation mode)
# ========================================

class ServoController:
    """Manages servo control with simulation fallback."""

    def __init__(self):
        self.pan_angle = PAN_DEFAULT
        self.tilt_angle = TILT_DEFAULT

        if SERVOS_AVAILABLE:
            print("[SERVO] Initializing PCA9685...")
            self.kit = ServoKit(channels=16, address=0x40)
            self.set_pan(PAN_DEFAULT)
            self.set_tilt(TILT_DEFAULT)
        else:
            print("[SERVO] Running in simulation mode")
            self.kit = None

    def set_pan(self, angle):
        angle = np.clip(angle, PAN_MIN, PAN_MAX)
        self.pan_angle = angle
        if self.kit:
            self.kit.servo[2].angle = angle

    def set_tilt(self, angle):
        angle = np.clip(angle, TILT_MIN, TILT_MAX)
        self.tilt_angle = angle
        if self.kit:
            self.kit.servo[3].angle = angle

    def move_smooth(self, target_pan, target_tilt):
        delta_pan = np.clip(target_pan - self.pan_angle, -MAX_SERVO_STEP, MAX_SERVO_STEP)
        delta_tilt = np.clip(target_tilt - self.tilt_angle, -MAX_SERVO_STEP, MAX_SERVO_STEP)

        self.set_pan(self.pan_angle + delta_pan)
        self.set_tilt(self.tilt_angle + delta_tilt)

    def reset(self):
        self.set_pan(PAN_DEFAULT)
        self.set_tilt(TILT_DEFAULT)


# ========================================
# Target Tracker
# ========================================

class TargetTracker:
    """Manages target locking and tracking state."""

    def __init__(self):
        self.locked_id = None
        self.last_seen_time = None
        self.is_locked = False
        self.manual_lock_disabled = False

    def lock_target(self, track_id):
        if not self.is_locked and not self.manual_lock_disabled:
            self.locked_id = track_id
            self.is_locked = True
            self.last_seen_time = time.time()

    def manual_unlock(self):
        if self.is_locked:
            self.unlock()
        self.manual_lock_disabled = True

    def manual_lock_enable(self):
        self.manual_lock_disabled = False

    def update_target(self, track_id):
        if self.is_locked and track_id == self.locked_id:
            self.last_seen_time = time.time()

    def check_timeout(self):
        if self.is_locked and self.last_seen_time is not None:
            elapsed = time.time() - self.last_seen_time
            if elapsed > TARGET_LOST_TIMEOUT:
                self.unlock()
                return True
        return False

    def unlock(self):
        self.locked_id = None
        self.is_locked = False
        self.last_seen_time = None

    def get_status(self):
        if self.manual_lock_disabled:
            return "MANUAL MODE"
        if self.is_locked:
            return f"LOCKED ID:{self.locked_id}"
        return "SEARCHING"


# ========================================
# Sentry Service (Background Thread)
# ========================================

class SentryService:
    """
    Background service that runs person tracking and generates annotated frames.
    Designed to be integrated into FastAPI.
    """

    def __init__(self):
        print("\n[SENTRY] Initializing service...")

        # Camera
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Use MJPEG for faster decoding

        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")

        # YOLO
        print("[SENTRY] Loading YOLO model...")
        self.yolo_model = YOLO('yolo11n.pt')

        # Face detection
        print("[FACE] Loading face detector...")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.face_cascade.empty():
            print("[FACE] Warning: Face detector failed to load - will use body tracking only")
            self.face_detection_enabled = False
        else:
            print("[FACE] Face detector loaded successfully")
            self.face_detection_enabled = True and FACE_PRIORITY

        # DeepSORT
        # Use faster embedding model and reduced parameters for Jetson
        self.tracker = DeepSort(
            max_age=30, 
            n_init=3,
            embedder="mobilenet",  # Faster than default
            half=True,  # Use FP16 for speed
            bgr=True,  # Our frames are BGR
            embedder_gpu=True  # Use GPU for embeddings
        )

        # Servo
        self.servo = ServoController()

        # Target tracker
        self.target = TargetTracker()

        # Frame management
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # Command queue for external control
        self.command_queue = Queue()

        # State
        self.running = False
        self.thread = None
        self.frame_center_x = CAMERA_WIDTH // 2
        self.frame_center_y = CAMERA_HEIGHT // 2

        # FPS
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0

        # Performance optimization
        self.frame_counter = 0
        self.last_detections = []  # Cache last YOLO detections
        self.last_tracks = []  # Cache last tracking results
        self.last_face_center = None  # Cache last face detection
        self.face_frame_counter = 0
        
        # Performance profiling
        self.profiling_enabled = True
        self.profile_times = {
            'yolo': [],
            'tracking': [],
            'face': [],
            'drawing': [],
            'total': []
        }

        # Stats for API
        self.stats = {
            'fps': 0,
            'tracking_status': 'SEARCHING',
            'pan_angle': PAN_DEFAULT,
            'tilt_angle': TILT_DEFAULT,
            'people_count': 0
        }

        print("[SENTRY] Service initialized")

    def start(self):
        """Start the background thread."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()
            print("[SENTRY] Background thread started")

    def stop(self):
        """Stop the background thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.cleanup()

    def send_command(self, command: str):
        """Send a command to the sentry (from API)."""
        self.command_queue.put(command)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest annotated frame (thread-safe)."""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def detect_faces(self, frame, person_bbox):
        """
        Detect faces within a person's bounding box.
        Returns (x, y) center of the largest face, or None if no faces found.
        
        Args:
            frame: BGR frame from camera
            person_bbox: [x1, y1, x2, y2] bounding box of detected person
        
        Returns:
            tuple: (center_x, center_y) of largest face in absolute frame coordinates,
                   or None if no face detected
        """
        if not self.face_detection_enabled:
            return None
        
        x1, y1, x2, y2 = map(int, person_bbox)
        
        # Ensure bbox is within frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        # Extract person region of interest
        person_roi = frame[y1:y2, x1:x2]
        
        if person_roi.size == 0:
            return None
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=FACE_SCALE_FACTOR,
            minNeighbors=FACE_MIN_NEIGHBORS,
            minSize=FACE_MIN_SIZE
        )
        
        if len(faces) == 0:
            return None
        
        # Find largest face (by area)
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        fx, fy, fw, fh = largest_face
        
        # Convert face center from ROI coordinates to absolute frame coordinates
        face_center_x = x1 + fx + fw // 2
        face_center_y = y1 + fy + fh // 2
        
        return (face_center_x, face_center_y)

    def get_stats(self) -> Dict[str, Any]:
        """Get current stats for API."""
        return self.stats.copy()

    def _process_commands(self):
        """Process commands from the queue."""
        while not self.command_queue.empty():
            cmd = self.command_queue.get()

            if cmd == 'toggle_lock':
                if self.target.manual_lock_disabled:
                    self.target.manual_lock_enable()
                else:
                    self.target.manual_unlock()

            elif cmd == 'center':
                if not self.target.is_locked:
                    self.servo.reset()

            elif cmd == 'pan_left':
                new_pan = max(PAN_MIN, self.servo.pan_angle - 5)
                self.servo.set_pan(new_pan)

            elif cmd == 'pan_right':
                new_pan = min(PAN_MAX, self.servo.pan_angle + 5)
                self.servo.set_pan(new_pan)

            elif cmd == 'tilt_up':
                new_tilt = max(TILT_MIN, self.servo.tilt_angle - 5)
                self.servo.set_tilt(new_tilt)

            elif cmd == 'tilt_down':
                new_tilt = min(TILT_MAX, self.servo.tilt_angle + 5)
                self.servo.set_tilt(new_tilt)

    def _run_loop(self):
        """Main processing loop (runs in background thread)."""
        print("[SENTRY] Processing loop started")

        while self.running:
            loop_start = time.time()
            
            # Process external commands
            self._process_commands()

            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            self.frame_counter += 1

            # Run YOLO detection only every N frames
            yolo_start = time.time()
            if self.frame_counter % DETECTION_SKIP_FRAMES == 0:
                self.last_detections = self._detect_people(frame)
            yolo_time = time.time() - yolo_start

            # Update tracker with cached detections - MAJOR BOTTLENECK
            # DeepSORT embedding extraction is expensive, skip frames
            track_start = time.time()
            if self.frame_counter % TRACKING_UPDATE_SKIP == 0:
                self.last_tracks = self._update_tracks(frame, self.last_detections)
            tracks = self.last_tracks
            track_time = time.time() - track_start

            # Check timeout
            self.target.check_timeout()

            # Process tracking
            face_start = time.time()
            target_found = False
            for track in tracks:
                track_id = track['id']
                bbox = track['bbox']

                if not self.target.is_locked:
                    self.target.lock_target(track_id)

                if self.target.is_locked and track_id == self.target.locked_id:
                    self.target.update_target(track_id)
                    target_found = True

                    # Get target center - prioritize face if detected
                    if FACE_PRIORITY and self.face_detection_enabled:
                        self.face_frame_counter += 1
                        # Run face detection every N frames to reduce load
                        if self.face_frame_counter % FACE_DETECTION_SKIP_FRAMES == 0:
                            self.last_face_center = self.detect_faces(frame, bbox)
                        
                        if self.last_face_center:
                            cx, cy = self.last_face_center
                        else:
                            # No face detected, fall back to body center
                            cx, cy = self._get_bbox_center(bbox)
                    else:
                        cx, cy = self._get_bbox_center(bbox)
                    
                    self._control_servos(cx, cy)
                    break
            
            face_time = time.time() - face_start
            
            # Clear cached face if target lost
            if not target_found:
                self.last_face_center = None
                self.face_frame_counter = 0

            # Update FPS
            self._update_fps()

            # Draw UI
            draw_start = time.time()
            annotated_frame = self._draw_ui(frame, tracks)
            draw_time = time.time() - draw_start

            # Store latest frame (thread-safe)
            with self.frame_lock:
                self.latest_frame = annotated_frame

            # Update stats
            loop_time = time.time() - loop_start
            
            # Profile logging every 30 frames
            if self.profiling_enabled and self.frame_counter % 30 == 0:
                print(f"[PROFILE] YOLO: {yolo_time*1000:.1f}ms | Track: {track_time*1000:.1f}ms | "
                      f"Face: {face_time*1000:.1f}ms | Draw: {draw_time*1000:.1f}ms | "
                      f"Total: {loop_time*1000:.1f}ms | FPS: {self.current_fps:.1f}")
            
            self.stats = {
                'fps': self.current_fps,
                'tracking_status': self.target.get_status(),
                'pan_angle': self.servo.pan_angle,
                'tilt_angle': self.servo.tilt_angle,
                'people_count': len(tracks)
            }

        print("[SENTRY] Processing loop stopped")

    def _detect_people(self, frame):
        """Detect people using YOLO."""
        results = self.yolo_model(frame, verbose=False, conf=0.35, classes=[0], imgsz=YOLO_IMGSZ)
        detections = []

        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == PERSON_CLASS_ID:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    detections.append([x1, y1, x2, y2, conf])

        return detections

    def _update_tracks(self, frame, detections):
        """Update DeepSORT tracker."""
        deepsort_detections = []
        for det in detections:
            x1, y1, x2, y2, conf = det
            w, h = x2 - x1, y2 - y1
            deepsort_detections.append(([x1, y1, w, h], conf, 'person'))

        tracks = self.tracker.update_tracks(deepsort_detections, frame=frame)

        confirmed_tracks = []
        for track in tracks:
            if track.is_confirmed():
                bbox = track.to_ltrb()
                confirmed_tracks.append({'id': track.track_id, 'bbox': bbox})

        return confirmed_tracks

    def _get_bbox_center(self, bbox):
        """Get center of bounding box."""
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def _control_servos(self, target_x, target_y):
        """Control servos to center target."""
        error_x = target_x - self.frame_center_x
        error_y = target_y - self.frame_center_y

        if abs(error_x) < DEADBAND_X:
            error_x = 0
        if abs(error_y) < DEADBAND_Y:
            error_y = 0

        delta_pan = error_x * KP * PAN_INVERT
        delta_tilt = -error_y * KP * TILT_INVERT

        target_pan = self.servo.pan_angle + delta_pan
        target_tilt = self.servo.tilt_angle + delta_tilt

        self.servo.move_smooth(target_pan, target_tilt)

    def _update_fps(self):
        """Update FPS counter."""
        self.fps_frame_count += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed > 1.0:
            self.current_fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.fps_start_time = time.time()

    def _draw_ui(self, frame, tracks):
        """Draw UI overlays on frame."""
        # Draw tracked people
        for track in tracks:
            bbox = track['bbox']
            track_id = track['id']
            x1, y1, x2, y2 = map(int, bbox)

            if self.target.is_locked and track_id == self.target.locked_id:
                color = (0, 255, 0)  # Green
                thickness = 3
                label = f"TARGET ID:{track_id}"
                
                # Draw cached face if available
                if FACE_PRIORITY and self.face_detection_enabled and self.last_face_center:
                    fx, fy = self.last_face_center
                    # Draw circle around face center
                    cv2.circle(frame, (fx, fy), 8, (0, 255, 255), -1)  # Yellow dot
                    cv2.circle(frame, (fx, fy), 20, (0, 255, 255), 2)  # Yellow circle
                    # Draw line from face to frame center
                    cv2.line(frame, (fx, fy), (self.frame_center_x, self.frame_center_y), (0, 255, 255), 1)
            else:
                color = (255, 100, 0)  # Blue
                thickness = 2
                label = f"ID:{track_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw crosshair
        cv2.line(frame, (self.frame_center_x - 20, self.frame_center_y),
                (self.frame_center_x + 20, self.frame_center_y), (0, 0, 255), 2)
        cv2.line(frame, (self.frame_center_x, self.frame_center_y - 20),
                (self.frame_center_x, self.frame_center_y + 20), (0, 0, 255), 2)

        # Draw status
        status_text = self.target.get_status()
        cv2.putText(frame, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw servo angles
        servo_text = f"Pan: {self.servo.pan_angle:.1f}  Tilt: {self.servo.tilt_angle:.1f}"
        cv2.putText(frame, servo_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw FPS
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw people count
        cv2.putText(frame, f"Tracks: {len(tracks)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def cleanup(self):
        """Clean up resources."""
        print("[SENTRY] Cleaning up...")
        if self.cap:
            self.cap.release()
        self.servo.reset()
        print("[SENTRY] Cleanup complete")
