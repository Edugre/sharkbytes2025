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
from datetime import datetime
from pathlib import Path
import json

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
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
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

# Auto-scan parameters (when no target)
AUTO_SCAN_ENABLED = True  # Enable automatic scanning when no target
SCAN_SPEED = 0.8  # Degrees per frame (lower = smoother)
SCAN_RANGE = 60  # Degrees to scan left/right from center
SCAN_CENTER_PAUSE = 1.0  # Seconds to pause at center before scanning

# Performance optimization
DETECTION_SKIP_FRAMES = 3  # Run YOLO every N frames (1=every frame, 2=every other, 3=every third)
YOLO_IMGSZ = 160  # Reduced from 320 for faster inference on Jetson
TRACKING_UPDATE_SKIP = 2  # Run DeepSORT embedding every N frames (major bottleneck!)

# Face detection parameters
FACE_PRIORITY = True  # Prioritize face tracking over body tracking
FACE_SCALE_FACTOR = 1.2  # Increased for faster detection (was 1.1)
FACE_MIN_NEIGHBORS = 4  # Reduced for faster detection (was 5)
FACE_MIN_SIZE = (40, 40)  # Increased minimum size for faster detection
FACE_DETECTION_SKIP_FRAMES = 5  # Run face detection every N frames when target locked

# Snapshot parameters
SNAPSHOT_INTERVAL = 15  # Seconds between snapshots for same person
ENABLE_GEMINI_ANALYSIS = True  # Enable Gemini AI analysis of snapshots (uploads to Supabase)


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
        # Initialize YOLO model
        self.model = YOLO('models/yolo11n_160_fp16.engine')  # Load TensorRT engine for faster inference

        # Face detection
        print("[FACE] Loading face detector...")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.face_cascade.empty():
            print("[FACE] Warning: Face detector failed to load - will use body tracking only")
            self.face_detection_enabled = False
        else:
            print("[FACE] Face detector loaded successfully")
            self.face_detection_enabled = True and FACE_PRIORITY

        # ByteTrack is built into YOLO - no separate tracker needed!
        print("[TRACK] Using ByteTrack (built-in)")
        self.use_bytetrack = True

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
        self.last_tracks = []  # Cache last tracking results (ByteTrack is built-in)
        self.last_face_center = None  # Cache last face detection
        self.face_frame_counter = 0
        
        # Snapshot tracking (in-memory only, no local storage)
        self.track_snapshots = {}  # {track_id: last_snapshot_time}
        self.seen_track_ids = set()  # Set of all track IDs seen
        self.snapshot_queue = Queue()  # Queue for Gemini analysis
        
        # Performance profiling
        self.profiling_enabled = True
        self.profile_times = {
            'yolo': [],
            'tracking': [],
            'face': [],
            'drawing': [],
            'total': []
        }

        # Auto-scan state (when no target locked)
        self.scan_direction = 1  # 1 = scanning right, -1 = scanning left
        self.scan_center_time = None  # Time when we last centered
        self.is_scanning = False  # Currently in scan mode
        
        # Manual control override
        self.manual_control_active = False  # Manual control has priority
        self.manual_control_timeout = 2.0  # Seconds before reverting to auto
        self.last_manual_command_time = 0  # Timestamp of last manual command

        # Auto-tracking state
        self.auto_tracking_enabled = True  # Auto-tracking is on by default

        # Stats for API
        self.stats = {
            'fps': 0,
            'tracking_status': 'SEARCHING',
            'pan_angle': PAN_DEFAULT,
            'tilt_angle': TILT_DEFAULT,
            'people_count': 0
        }

        print("[SENTRY] Service initialized")
        
        # Start Gemini analysis worker thread if enabled
        if ENABLE_GEMINI_ANALYSIS:
            self.gemini_worker = threading.Thread(target=self._gemini_analysis_worker, daemon=True)
            self.gemini_worker.start()
            print("[GEMINI] Analysis worker started")

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
    
    def get_snapshot_stats(self) -> Dict[str, Any]:
        """Get snapshot and analysis statistics."""
        return {
            'total_people_seen': len(self.seen_track_ids),
            'active_snapshots': len(self.track_snapshots),
            'pending_analyses': self.snapshot_queue.qsize(),
            'storage_mode': 'supabase_only',
            'gemini_enabled': ENABLE_GEMINI_ANALYSIS
        }

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
        # Ensure all values are JSON serializable (convert numpy types to Python types)
        stats = self.stats.copy()
        return {
            'fps': float(stats.get('fps', 0)),
            'tracking_status': str(stats.get('tracking_status', 'UNKNOWN')),
            'pan_angle': float(stats.get('pan_angle', 90)),
            'tilt_angle': float(stats.get('tilt_angle', 90)),
            'people_count': int(stats.get('people_count', 0))
        }

    def _process_commands(self):
        """Process commands from the queue."""
        while not self.command_queue.empty():
            cmd = self.command_queue.get()

            if cmd == 'toggle_lock':
                # Toggle between Auto-Tracking and Manual mode
                self.auto_tracking_enabled = not self.auto_tracking_enabled
                if not self.auto_tracking_enabled:
                    # Switching to manual mode - unlock any target
                    self.target.unlock()
                    self.is_scanning = False
                    self.scan_center_time = None
                print(f"[CONTROL] Auto-tracking: {'ENABLED' if self.auto_tracking_enabled else 'DISABLED'}")

            elif cmd == 'center':
                self.servo.reset()
                # Manual control overrides auto behavior
                self.manual_control_active = True
                self.last_manual_command_time = time.time()

            elif cmd == 'pan_left':
                new_pan = max(PAN_MIN, self.servo.pan_angle - 5)
                self.servo.set_pan(new_pan)
                # Manual control overrides auto behavior
                self.manual_control_active = True
                self.last_manual_command_time = time.time()

            elif cmd == 'pan_right':
                new_pan = min(PAN_MAX, self.servo.pan_angle + 5)
                self.servo.set_pan(new_pan)
                # Manual control overrides auto behavior
                self.manual_control_active = True
                self.last_manual_command_time = time.time()

            elif cmd == 'tilt_up':
                new_tilt = min(TILT_MAX, self.servo.tilt_angle + 5)  # Fixed: up = increase angle
                self.servo.set_tilt(new_tilt)
                # Manual control overrides auto behavior
                self.manual_control_active = True
                self.last_manual_command_time = time.time()

            elif cmd == 'tilt_down':
                new_tilt = max(TILT_MIN, self.servo.tilt_angle - 5)  # Fixed: down = decrease angle
                self.servo.set_tilt(new_tilt)
                # Manual control overrides auto behavior
                self.manual_control_active = True
                self.last_manual_command_time = time.time()

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

            # Run YOLO detection with ByteTrack (tracking built-in)
            yolo_start = time.time()
            if self.frame_counter % DETECTION_SKIP_FRAMES == 0:
                self.last_tracks = self._detect_and_track(frame)
            tracks = self.last_tracks
            yolo_time = time.time() - yolo_start
            
            # No separate tracking step - ByteTrack runs with YOLO
            track_time = 0

            # Check timeout
            self.target.check_timeout()
            
            # Check if manual control has timed out
            current_time = time.time()
            if self.manual_control_active:
                if current_time - self.last_manual_command_time > self.manual_control_timeout:
                    self.manual_control_active = False
                    print("[MANUAL] Control timeout - returning to auto mode")

            # Process tracking (ONLY if manual control is not active AND auto-tracking is enabled)
            face_start = time.time()
            target_found = False
            
            if not self.manual_control_active and self.auto_tracking_enabled:
                for track in tracks:
                    track_id = track['id']
                    bbox = track['bbox']
                    
                    # Check if this is a new ID - take immediate snapshot
                    is_new_id = track_id not in self.seen_track_ids
                    if is_new_id:
                        self.seen_track_ids.add(track_id)
                        self._take_snapshot(frame, track_id, bbox, is_new=True)
                        print(f"[SNAPSHOT] New person detected (ID: {track_id})")
                    else:
                        # Check if it's time for periodic snapshot
                        self._check_periodic_snapshot(frame, track_id, bbox)

                    if not self.target.is_locked:
                        self.target.lock_target(track_id)
                        # Reset scanning state when locking onto target
                        self.is_scanning = False
                        self.scan_center_time = None

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
                
                # Auto-scan when no target is locked (ONLY if auto-tracking enabled and manual control is not active)
                if AUTO_SCAN_ENABLED and self.auto_tracking_enabled and not self.target.is_locked and not self.manual_control_active:
                    self._auto_scan()

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

    def _detect_and_track(self, frame):
        """Detect and track people using YOLO with ByteTrack."""
        # Use YOLO's track() method which includes ByteTrack
        results = self.model.track(
            frame, 
            persist=True,  # Persist tracks across frames
            verbose=False, 
            conf=0.35, 
            classes=[0],  # Person class
            imgsz=YOLO_IMGSZ,
            tracker="bytetrack.yaml"  # Use ByteTrack
        )
        
        tracks = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                
                for box, track_id in zip(boxes, track_ids):
                    tracks.append({
                        'id': int(track_id),
                        'bbox': box  # [x1, y1, x2, y2]
                    })
        
        return tracks

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

    def _auto_scan(self):
        """
        Automatically scan left and right when no target is locked.
        Centers first, pauses briefly, then scans side to side.
        """
        current_time = time.time()
        
        # If not currently scanning, return to center first
        if not self.is_scanning:
            # Check if we're already centered
            if abs(self.servo.pan_angle - PAN_DEFAULT) > 2:
                # Move towards center
                if self.servo.pan_angle > PAN_DEFAULT:
                    self.servo.set_pan(max(PAN_DEFAULT, self.servo.pan_angle - SCAN_SPEED * 2))
                else:
                    self.servo.set_pan(min(PAN_DEFAULT, self.servo.pan_angle + SCAN_SPEED * 2))
                return
            else:
                # We're centered, start pause timer if not already set
                if self.scan_center_time is None:
                    self.scan_center_time = current_time
                    self.servo.set_pan(PAN_DEFAULT)
                    self.servo.set_tilt(TILT_DEFAULT)
                    return
                
                # Check if pause is complete
                if current_time - self.scan_center_time < SCAN_CENTER_PAUSE:
                    return
                
                # Pause complete, start scanning
                self.is_scanning = True
                self.scan_center_time = None
        
        # Calculate scan boundaries
        scan_left = PAN_DEFAULT - SCAN_RANGE
        scan_right = PAN_DEFAULT + SCAN_RANGE
        
        # Clamp to servo limits
        scan_left = max(PAN_MIN, scan_left)
        scan_right = min(PAN_MAX, scan_right)
        
        # Update pan angle based on scan direction
        new_pan = self.servo.pan_angle + (SCAN_SPEED * self.scan_direction)
        
        # Check if we've reached the boundary and need to reverse
        if self.scan_direction > 0 and new_pan >= scan_right:
            new_pan = scan_right
            self.scan_direction = -1  # Reverse to scan left
        elif self.scan_direction < 0 and new_pan <= scan_left:
            new_pan = scan_left
            self.scan_direction = 1  # Reverse to scan right
        
        # Apply the new pan angle
        self.servo.set_pan(new_pan)

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
                color = (249, 221, 53)  # Cyan #35DDF9 (BGR format)
                thickness = 3
                label = f"TARGET ID:{track_id}"
                
                # Draw cached face if available
                if FACE_PRIORITY and self.face_detection_enabled and self.last_face_center:
                    fx, fy = self.last_face_center
                    # Draw circle around face center
                    cv2.circle(frame, (fx, fy), 8, (249, 221, 53), -1)  # Cyan dot
                    cv2.circle(frame, (fx, fy), 20, (249, 221, 53), 2)  # Cyan circle
                    # Draw line from face to frame center
                    cv2.line(frame, (fx, fy), (self.frame_center_x, self.frame_center_y), (249, 221, 53), 1)
            else:
                color = (255, 180, 100)  # Light blue (BGR format)
                thickness = 2
                label = f"ID:{track_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw crosshair - cyan theme
        cv2.line(frame, (self.frame_center_x - 20, self.frame_center_y),
                (self.frame_center_x + 20, self.frame_center_y), (255, 139, 61), 2)  # Blue #3D8BFF
        cv2.line(frame, (self.frame_center_x, self.frame_center_y - 20),
                (self.frame_center_x, self.frame_center_y + 20), (255, 139, 61), 2)  # Blue #3D8BFF

        # Note: Status, FPS, and Tracks text moved to frontend overlay for better readability
        return frame

    def _take_snapshot(self, frame, track_id, bbox, is_new=False):
        """
        Take a snapshot of the tracked person and queue for Gemini analysis.
        Snapshot is kept in memory only, not saved to disk.
        
        Args:
            frame: Current video frame
            track_id: ID of the tracked person
            bbox: Bounding box [x1, y1, x2, y2]
            is_new: True if this is a new person (overrides timer)
        """
        current_time = time.time()
        
        # Check if we should take a snapshot
        should_snapshot = False
        
        if is_new:
            # New person - always take snapshot
            should_snapshot = True
            reason = "new_person"
        elif track_id in self.track_snapshots:
            # Existing person - check timer
            time_since_last = current_time - self.track_snapshots[track_id]
            if time_since_last >= SNAPSHOT_INTERVAL:
                should_snapshot = True
                reason = "periodic"
        else:
            # First time seeing this ID (shouldn't happen if is_new works)
            should_snapshot = True
            reason = "first_sighting"
        
        if should_snapshot:
            # Update last snapshot time
            self.track_snapshots[track_id] = current_time
            
            # Encode frame to JPEG in memory (no disk save)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            _, img_encoded = cv2.imencode('.jpg', frame)
            
            # Queue for Gemini analysis
            if ENABLE_GEMINI_ANALYSIS:
                self.snapshot_queue.put({
                    'image_data': img_encoded.tobytes(),  # Raw JPEG bytes
                    'track_id': track_id,
                    'bbox': bbox,
                    'timestamp': timestamp,
                    'reason': reason
                })
            
            print(f"[SNAPSHOT] Queued person_{track_id}_{timestamp}_{reason} for analysis (memory only)")
    
    def _check_periodic_snapshot(self, frame, track_id, bbox):
        """
        Check if it's time for a periodic snapshot of an existing person.
        
        Args:
            frame: Current video frame
            track_id: ID of the tracked person
            bbox: Bounding box [x1, y1, x2, y2]
        """
        if track_id in self.track_snapshots:
            current_time = time.time()
            time_since_last = current_time - self.track_snapshots[track_id]
            
            if time_since_last >= SNAPSHOT_INTERVAL:
                self._take_snapshot(frame, track_id, bbox, is_new=False)
    
    def _gemini_analysis_worker(self):
        """
        Background worker that processes snapshots through Gemini API,
        uploads to Supabase, and sends Discord alerts.
        Does NOT save files locally - everything goes directly to Supabase.
        Runs in a separate thread to avoid blocking the main loop.
        """
        print("[GEMINI] Worker thread started")
        
        # Import required modules
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / 'gemini'))
            from gemini_description import analyze_security_image
            print("[GEMINI] Successfully imported analyzer")
        except Exception as e:
            print(f"[GEMINI] Failed to import analyzer: {e}")
            print("[GEMINI] Worker disabled - analyzer not available")
            return
        
        # Import Supabase and Discord alert function
        try:
            from dotenv import load_dotenv
            from supabase import create_client, Client
            sys.path.insert(0, str(Path(__file__).parent.parent / 'web'))
            from alerts import send_discord_alert
            
            load_dotenv()
            SUPABASE_URL = os.getenv("SUPABASE_URL")
            SUPABASE_KEY = os.getenv("SUPABASE_KEY")
            
            if SUPABASE_URL and SUPABASE_KEY:
                supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
                supabase_enabled = True
                print("[GEMINI] Supabase integration enabled")
            else:
                supabase_enabled = False
                print("[GEMINI] ERROR: Supabase not configured - worker disabled")
                return
        except Exception as e:
            print(f"[GEMINI] Failed to initialize Supabase: {e}")
            return
        
        while self.running:
            try:
                # Wait for a snapshot to analyze (with timeout to check self.running)
                try:
                    snapshot_data = self.snapshot_queue.get(timeout=1.0)
                except:
                    continue
                
                image_data = snapshot_data['image_data']
                track_id = snapshot_data['track_id']
                reason = snapshot_data['reason']
                timestamp_str = snapshot_data['timestamp']
                
                print(f"[GEMINI] Analyzing snapshot for person {track_id}...")
                
                # Create temporary file for Gemini analysis (will be deleted after)
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    tmp_file.write(image_data)
                    tmp_filepath = tmp_file.name
                
                try:
                    # Analyze with Gemini
                    result = analyze_security_image(tmp_filepath)
                    
                    # Print summary
                    if result['status'] == 'success':
                        severity_emoji = {
                            'info': 'ℹ️',
                            'warning': '⚠️',
                            'critical': '🚨'
                        }
                        emoji = severity_emoji.get(result.get('severity', 'info'), 'ℹ️')
                        print(f"[GEMINI] {emoji} {result['analysis']}")
                        
                        # Upload to Supabase
                        try:
                            # Upload to Supabase Storage
                            storage_filename = f"person_{track_id}_{timestamp_str}_{reason}.jpg"
                            supabase.storage.from_("security-frames").upload(
                                path=storage_filename,
                                file=image_data,
                                file_options={"content-type": "image/jpeg"}
                            )
                            
                            # Get public URL
                            image_url = supabase.storage.from_("security-frames").get_public_url(storage_filename)
                            
                            # Create event in database
                            event_data = {
                                "event_type": "person_detected",
                                "description": result['analysis'],
                                "severity": result.get('severity', 'info'),
                                "timestamp": result['timestamp'],
                                "image_url": image_url
                            }
                            
                            db_response = supabase.table("events").insert(event_data).execute()
                            
                            if db_response.data:
                                event_id = db_response.data[0].get('id')
                                print(f"[SUPABASE] Event created (ID: {event_id})")
                            
                            # Send Discord alert for warning/critical
                            if result.get('severity') in ['warning', 'critical']:
                                send_discord_alert(
                                    event_type="person_detected",
                                    description=result['analysis'],
                                    severity=result['severity'],
                                    image_url=image_url
                                )
                                print(f"[DISCORD] Alert sent")
                                
                        except Exception as e:
                            print(f"[SUPABASE] Error uploading snapshot: {e}")
                    else:
                        print(f"[GEMINI] ❌ Analysis failed: {result.get('error', 'Unknown error')}")
                
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_filepath)
                    except:
                        pass
                
                # Mark task as done
                self.snapshot_queue.task_done()
                
            except Exception as e:
                print(f"[GEMINI] Error in analysis worker: {e}")
                import traceback
                traceback.print_exc()
        
        print("[GEMINI] Worker thread stopped")

    def cleanup(self):
        """Clean up resources."""
        print("[SENTRY] Cleaning up...")
        
        # Wait for any remaining Gemini analyses to complete
        if ENABLE_GEMINI_ANALYSIS and not self.snapshot_queue.empty():
            print("[GEMINI] Waiting for pending analyses...")
            self.snapshot_queue.join()
        
        if self.cap:
            self.cap.release()
        self.servo.reset()
        print("[SENTRY] Cleanup complete")
