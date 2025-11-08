#!/usr/bin/env python3
"""
Jetson Person-Tracking Sentry Turret
=====================================
Uses YOLO for person detection, DeepSORT for tracking, and PCA9685 servo driver
to keep a tracked person centered in the camera frame.

Hardware:
- Jetson device
- USB Camera (/dev/video0)
- PCA9685 Servo Driver (I2C address 0x40)
- 2 Servos: Pan (channel 0) and Tilt (channel 1)

Author: AI Assistant
Date: November 7, 2025
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from adafruit_servokit import ServoKit

# ========================================
# Configuration Constants
# ========================================

# Camera settings
CAMERA_INDEX = 0  # /dev/video0
CAMERA_WIDTH = 320  # Small resolution for better FPS
CAMERA_HEIGHT = 320  # Small resolution for better FPS
TARGET_FPS = 30
SKIP_FRAMES = 1  # Process every frame (1=no skipping) - was 2

# Servo settings
PCA9685_ADDRESS = 0x40
PCA9685_CHANNELS = 16
PAN_CHANNEL = 0
TILT_CHANNEL = 1

# Servo angle limits (degrees)
PAN_MIN = 10
PAN_MAX = 170
PAN_DEFAULT = 90

TILT_MIN = 20
TILT_MAX = 150
TILT_DEFAULT = 90

# Control parameters
KP = 0.035  # Proportional gain for servo control - REDUCED to prevent overshooting
MAX_SERVO_STEP = 3.0  # Maximum servo movement per iteration (degrees) - REDUCED for stability
DEADBAND_X = 25  # Horizontal deadband in pixels (±) - increased to reduce jitter
DEADBAND_Y = 25  # Vertical deadband in pixels (±) - increased to reduce jitter

# Servo direction multipliers (change to -1 to invert direction)
PAN_INVERT = -1   # Set to -1 if servo moves opposite direction
TILT_INVERT = -1  # Set to -1 if servo moves opposite direction

# Tracking parameters
TARGET_LOST_TIMEOUT = 2.0  # Seconds before considering target lost
PERSON_CLASS_ID = 0  # COCO class ID for person

# Face detection parameters
FACE_PRIORITY = True  # Prioritize face tracking over body tracking
FACE_SCALE_FACTOR = 1.1  # Face detection scale factor
FACE_MIN_NEIGHBORS = 5  # Minimum neighbors for face detection
FACE_MIN_SIZE = (30, 30)  # Minimum face size in pixels

# Idle sweep parameters (when no target)
SWEEP_SPEED = 0.5  # Degrees per frame
SWEEP_MIN = 30
SWEEP_MAX = 150
SWEEP_DIRECTION = 1  # 1 = right, -1 = left


# ========================================
# Servo Controller Class
# ========================================

class ServoController:
    """Manages PCA9685 servo control with smoothing and safety limits."""
    
    def __init__(self):
        """Initialize the ServoKit and set default positions."""
        print("[SERVO] Initializing PCA9685 servo controller...")
        self.kit = ServoKit(channels=PCA9685_CHANNELS, address=PCA9685_ADDRESS)
        
        # Current servo positions
        self.pan_angle = PAN_DEFAULT
        self.tilt_angle = TILT_DEFAULT
        
        # Set initial positions
        self.set_pan(PAN_DEFAULT)
        self.set_tilt(TILT_DEFAULT)
        print(f"[SERVO] Initialized at pan={PAN_DEFAULT}°, tilt={TILT_DEFAULT}°")
    
    def set_pan(self, angle):
        """Set pan servo angle with clamping."""
        angle = np.clip(angle, PAN_MIN, PAN_MAX)
        self.pan_angle = angle
        self.kit.servo[PAN_CHANNEL].angle = angle
    
    def set_tilt(self, angle):
        """Set tilt servo angle with clamping."""
        angle = np.clip(angle, TILT_MIN, TILT_MAX)
        self.tilt_angle = angle
        self.kit.servo[TILT_CHANNEL].angle = angle
    
    def move_smooth(self, target_pan, target_tilt):
        """
        Move servos toward target angles with maximum step size limiting.
        This provides smooth motion and prevents jitter.
        """
        # Calculate deltas
        delta_pan = target_pan - self.pan_angle
        delta_tilt = target_tilt - self.tilt_angle
        
        # Limit maximum step size
        delta_pan = np.clip(delta_pan, -MAX_SERVO_STEP, MAX_SERVO_STEP)
        delta_tilt = np.clip(delta_tilt, -MAX_SERVO_STEP, MAX_SERVO_STEP)
        
        # Apply movements
        new_pan = self.pan_angle + delta_pan
        new_tilt = self.tilt_angle + delta_tilt
        
        self.set_pan(new_pan)
        self.set_tilt(new_tilt)
    
    def reset(self):
        """Reset servos to default position."""
        self.set_pan(PAN_DEFAULT)
        self.set_tilt(TILT_DEFAULT)


# ========================================
# Target Tracker Class
# ========================================

class TargetTracker:
    """Manages target locking and tracking state."""
    
    def __init__(self):
        """Initialize tracker state."""
        self.locked_id = None
        self.last_seen_time = None
        self.is_locked = False
        self.manual_lock_disabled = False  # Manual lock override
    
    def lock_target(self, track_id):
        """Lock onto a specific track ID."""
        if not self.is_locked and not self.manual_lock_disabled:
            self.locked_id = track_id
            self.is_locked = True
            self.last_seen_time = time.time()
            print(f"[TRACK] Locked onto target ID: {track_id}")
    
    def manual_unlock(self):
        """Manually unlock and disable auto-locking."""
        if self.is_locked:
            print(f"[TRACK] Manually unlocked from target ID: {self.locked_id}")
            self.unlock()
        self.manual_lock_disabled = True
        print("[TRACK] Auto-lock DISABLED - Press 'L' to enable")
    
    def manual_lock_enable(self):
        """Re-enable auto-locking."""
        self.manual_lock_disabled = False
        print("[TRACK] Auto-lock ENABLED - Will lock to next person detected")
    
    def update_target(self, track_id):
        """Update last seen time for the locked target."""
        if self.is_locked and track_id == self.locked_id:
            self.last_seen_time = time.time()
    
    def check_timeout(self):
        """Check if target has been lost for too long."""
        if self.is_locked and self.last_seen_time is not None:
            elapsed = time.time() - self.last_seen_time
            if elapsed > TARGET_LOST_TIMEOUT:
                print(f"[TRACK] Target {self.locked_id} lost for {elapsed:.1f}s - unlocking")
                self.unlock()
                return True
        return False
    
    def unlock(self):
        """Unlock from current target."""
        self.locked_id = None
        self.is_locked = False
        self.last_seen_time = None
    
    def get_status(self):
        """Get current tracking status string."""
        if self.manual_lock_disabled:
            return "MANUAL MODE (Lock OFF)"
        if self.is_locked:
            elapsed = time.time() - self.last_seen_time if self.last_seen_time else 0
            return f"LOCKED ID:{self.locked_id} ({elapsed:.1f}s)"
        return "SEARCHING"


# ========================================
# Main Sentry System
# ========================================

class PersonTrackingSentry:
    """Main sentry system integrating detection, tracking, and servo control."""
    
    def __init__(self):
        """Initialize all subsystems."""
        print("\n" + "="*50)
        print("Person-Tracking Sentry Initializing...")
        print("="*50 + "\n")
        
        # Initialize camera
        print("[CAMERA] Opening camera...")
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera at /dev/video0")
        
        print(f"[CAMERA] Opened at {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {TARGET_FPS}fps")
        
        # Check CUDA availability
        import torch
        if torch.cuda.is_available():
            print(f"[GPU] CUDA Available: {torch.cuda.get_device_name(0)}")
            print(f"[GPU] CUDA Version: {torch.version.cuda}")
            print(f"[GPU] PyTorch Version: {torch.__version__}")
        else:
            print("[GPU] CUDA not available - using CPU (slower)")
        
        # Initialize YOLO model
        print("[YOLO] Loading YOLOv11 model...")
        self.yolo_model = YOLO('yolo11n.pt')  # YOLOv11 Nano - faster and more accurate than v8
        print("[YOLO] Model loaded successfully")
        
        # Initialize face detector (Haar Cascade)
        print("[FACE] Loading face detector...")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.face_cascade.empty():
            print("[FACE] Warning: Face detector failed to load - will use body tracking only")
            self.face_detection_enabled = False
        else:
            print("[FACE] Face detector loaded successfully")
            self.face_detection_enabled = True and FACE_PRIORITY
        
        # Initialize DeepSORT tracker
        print("[DEEPSORT] Initializing tracker...")
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            max_iou_distance=0.7,
            max_cosine_distance=0.3,
            nn_budget=100
        )
        print("[DEEPSORT] Tracker initialized")
        
        # Initialize servo controller
        self.servo = ServoController()
        
        # Initialize target tracker
        self.target = TargetTracker()
        
        # Idle sweep state
        self.sweep_angle = PAN_DEFAULT
        self.sweep_direction = SWEEP_DIRECTION
        
        # Frame center
        self.frame_center_x = CAMERA_WIDTH // 2
        self.frame_center_y = CAMERA_HEIGHT // 2
        
        # FPS calculation
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0
        
        # Frame skipping for performance
        self.frame_counter = 0
        self.last_detections = []
        self.last_tracks = []
        
        # Last known target position for smooth tracking between detections
        self.last_target_pos = None
        
        # Previous error for derivative control (prevents overshooting)
        self.prev_error_x = 0
        self.prev_error_y = 0
        
        print("\n" + "="*50)
        print("Sentry System Ready!")
        print("="*50 + "\n")
    
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
    
    def detect_people(self, frame):
        """
        Detect people in frame using YOLO.
        Returns list of [x1, y1, x2, y2, confidence] for each person.
        """
        # Run YOLO with optimizations for speed
        # Device is auto-detected (will use CUDA if available, CPU otherwise)
        results = self.yolo_model(
            frame, 
            verbose=False,
            conf=0.35,  # Confidence threshold
            iou=0.5,   # IoU threshold for NMS
            imgsz=320,  # Match camera resolution for speed
            classes=[0],  # Only detect person class
            half=True,   # Enable FP16 for CUDA acceleration (2x faster on GPU)
            max_det=5,  # Limit to 5 detections max (we only need 1 person anyway)
            agnostic_nms=True  # Faster NMS
        )
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Filter for person class only
                if int(box.cls[0]) == PERSON_CLASS_ID:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    detections.append([x1, y1, x2, y2, conf])
        
        return detections
    
    def update_tracks(self, frame, detections):
        """
        Update DeepSORT tracker with new detections.
        Returns list of tracks with track_id and bounding box.
        """
        # Convert detections to DeepSORT format: ([x1, y1, w, h], confidence, class)
        deepsort_detections = []
        for det in detections:
            x1, y1, x2, y2, conf = det
            w = x2 - x1
            h = y2 - y1
            deepsort_detections.append(([x1, y1, w, h], conf, 'person'))
        
        # Update tracker
        tracks = self.tracker.update_tracks(deepsort_detections, frame=frame)
        
        # Return confirmed tracks only
        confirmed_tracks = []
        for track in tracks:
            if track.is_confirmed():
                bbox = track.to_ltrb()  # [left, top, right, bottom]
                confirmed_tracks.append({
                    'id': track.track_id,
                    'bbox': bbox
                })
        
        return confirmed_tracks
    
    def get_bbox_center(self, bbox):
        """Calculate center point of bounding box."""
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return cx, cy
    
    def control_servos_proportional(self, target_x, target_y):
        """
        Use PD (Proportional-Derivative) control to drive servos toward target position.
        Includes deadband to prevent jitter and derivative term to prevent overshooting.
        """
        # Calculate error (distance from center)
        error_x = target_x - self.frame_center_x
        error_y = target_y - self.frame_center_y
        
        # Apply deadband
        if abs(error_x) < DEADBAND_X:
            error_x = 0
        if abs(error_y) < DEADBAND_Y:
            error_y = 0
        
        # Calculate derivative (change in error) to prevent overshooting
        # KD is the derivative gain - lower values make it less aggressive
        KD = 0.15  # Derivative gain for damping
        deriv_x = error_x - self.prev_error_x
        deriv_y = error_y - self.prev_error_y
        
        # Store current error for next iteration
        self.prev_error_x = error_x
        self.prev_error_y = error_y
        
        # Calculate servo adjustments using PD control
        # P term: proportional to error (main correction)
        # D term: proportional to rate of change (damping/smoothing)
        delta_pan = (error_x * KP - deriv_x * KD) * PAN_INVERT
        delta_tilt = -(error_y * KP - deriv_y * KD) * TILT_INVERT
        
        # Debug output (disabled for performance - uncomment if needed for debugging)
        # if error_x != 0 or error_y != 0:
        #     print(f"[TRACK] Error X: {error_x:+4.0f}px  Error Y: {error_y:+4.0f}px | "
        #           f"Delta Pan: {delta_pan:+.2f}°  Delta Tilt: {delta_tilt:+.2f}°")
        
        # Calculate target angles
        target_pan = self.servo.pan_angle + delta_pan
        target_tilt = self.servo.tilt_angle + delta_tilt
        
        # Move servos smoothly
        self.servo.move_smooth(target_pan, target_tilt)
    
    def idle_sweep(self):
        """
        Perform slow pan sweep when no target is locked.
        Looks for new targets by sweeping left and right.
        """
        self.sweep_angle += SWEEP_SPEED * self.sweep_direction
        
        # Reverse direction at limits
        if self.sweep_angle >= SWEEP_MAX:
            self.sweep_angle = SWEEP_MAX
            self.sweep_direction = -1
        elif self.sweep_angle <= SWEEP_MIN:
            self.sweep_angle = SWEEP_MIN
            self.sweep_direction = 1
        
        # Move to sweep position
        self.servo.move_smooth(self.sweep_angle, TILT_DEFAULT)
    
    def center_servos(self):
        """Center servos to default position."""
        print("[SERVO] Centering to default position...")
        self.servo.set_pan(PAN_DEFAULT)
        self.servo.set_tilt(TILT_DEFAULT)
        # Reset sweep angle too
        self.sweep_angle = PAN_DEFAULT
    
    def update_fps(self):
        """Calculate and update FPS counter."""
        self.fps_frame_count += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed > 1.0:
            self.current_fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.fps_start_time = time.time()
    
    def draw_ui(self, frame, tracks):
        """Draw UI overlays on frame."""
        # Draw all tracked people
        for track in tracks:
            bbox = track['bbox']
            track_id = track['id']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Color: green for locked target, blue for others
            if self.target.is_locked and track_id == self.target.locked_id:
                color = (0, 255, 0)  # Green
                thickness = 3
                label = f"TARGET ID:{track_id}"
                
                # Detect and draw face if this is the locked target
                if FACE_PRIORITY and self.face_detection_enabled:
                    face_center = self.detect_faces(frame, bbox)
                    if face_center:
                        # Draw face center point in yellow
                        cv2.circle(frame, face_center, 5, (0, 255, 255), -1)
                        cv2.putText(frame, "FACE", (face_center[0] - 20, face_center[1] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        # Draw line from body center to face center
                        body_center = self.get_bbox_center(bbox)
                        cv2.line(frame, body_center, face_center, (0, 255, 255), 1)
                    else:
                        # No face detected, show body center
                        cx, cy = self.get_bbox_center(bbox)
                        cv2.circle(frame, (cx, cy), 5, color, -1)
                else:
                    # Face tracking disabled, show body center
                    cx, cy = self.get_bbox_center(bbox)
                    cv2.circle(frame, (cx, cy), 5, color, -1)
            else:
                color = (255, 100, 0)  # Blue
                thickness = 2
                label = f"ID:{track_id}"
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw frame center crosshair
        cv2.line(frame, (self.frame_center_x - 20, self.frame_center_y),
                (self.frame_center_x + 20, self.frame_center_y), (0, 0, 255), 2)
        cv2.line(frame, (self.frame_center_x, self.frame_center_y - 20),
                (self.frame_center_x, self.frame_center_y + 20), (0, 0, 255), 2)
        
        # Draw deadband zone
        cv2.rectangle(frame,
                     (self.frame_center_x - DEADBAND_X, self.frame_center_y - DEADBAND_Y),
                     (self.frame_center_x + DEADBAND_X, self.frame_center_y + DEADBAND_Y),
                     (0, 255, 255), 1)
        
        # Draw status text
        status_text = self.target.get_status()
        cv2.putText(frame, f"Status: {status_text}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw servo angles
        servo_text = f"Pan: {self.servo.pan_angle:.1f}  Tilt: {self.servo.tilt_angle:.1f}"
        cv2.putText(frame, servo_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw people count
        cv2.putText(frame, f"Tracks: {len(tracks)}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw keyboard controls (bottom of screen)
        controls_y_start = CAMERA_HEIGHT - 80
        cv2.putText(frame, "Controls:", (10, controls_y_start),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "L: Lock ON/OFF", (10, controls_y_start + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, "C: Center (when unlocked)", (10, controls_y_start + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, "Q: Quit", (10, controls_y_start + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def run(self):
        """Main loop."""
        print("[SENTRY] Starting main loop...")
        print("="*50)
        print("Keyboard Controls:")
        print("  L - Toggle Lock ON/OFF")
        print("  C - Center servos (only when unlocked)")
        print("  Q - Quit")
        print("="*50 + "\n")
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("[ERROR] Failed to read frame from camera")
                    break
                
                # Increment frame counter
                self.frame_counter += 1
                
                # Run detection every frame (removed frame skipping as it causes stale bounding boxes)
                # Note: If FPS is too low, increase SKIP_FRAMES back to 2 or 3
                detections = self.detect_people(frame)
                
                # Update tracker with current frame
                tracks = self.update_tracks(frame, detections)
                
                # Check if locked target timed out
                self.target.check_timeout()
                
                # Process tracks
                target_found = False
                
                for track in tracks:
                    track_id = track['id']
                    bbox = track['bbox']
                    
                    # If not locked, lock onto first person detected
                    if not self.target.is_locked:
                        self.target.lock_target(track_id)
                    
                    # If this is our locked target, track it
                    if self.target.is_locked and track_id == self.target.locked_id:
                        self.target.update_target(track_id)
                        target_found = True
                        
                        # Get target center - prioritize face if detected
                        if FACE_PRIORITY and self.face_detection_enabled:
                            face_center = self.detect_faces(frame, bbox)
                            if face_center:
                                target_x, target_y = face_center
                            else:
                                # No face detected, fall back to body center
                                target_x, target_y = self.get_bbox_center(bbox)
                        else:
                            target_x, target_y = self.get_bbox_center(bbox)
                        
                        # Drive servos to center target
                        self.control_servos_proportional(target_x, target_y)
                        break
                
                # If no target found and not locked, perform idle sweep
                if not target_found and not self.target.is_locked:
                    self.idle_sweep()
                
                # Update FPS
                self.update_fps()
                
                # Draw UI
                frame = self.draw_ui(frame, tracks)
                
                # Display frame
                cv2.imshow('Person-Tracking Sentry', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("\n[SENTRY] Quit command received")
                    break
                
                elif key == ord('l') or key == ord('L'):
                    # Toggle lock on/off
                    if self.target.manual_lock_disabled:
                        # Currently disabled, enable it
                        self.target.manual_lock_enable()
                    else:
                        # Currently enabled, disable it
                        self.target.manual_unlock()
                
                elif key == ord('c') or key == ord('C'):
                    # Center servos (only if not locked)
                    if not self.target.is_locked:
                        self.center_servos()
                    else:
                        print("[SERVO] Cannot center - locked to target. Press 'L' to unlock first.")
        
        except KeyboardInterrupt:
            print("\n[SENTRY] Interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("\n[SENTRY] Cleaning up...")
        
        # Reset servos to default position
        print("[SERVO] Resetting to default position...")
        self.servo.reset()
        
        # Release camera
        print("[CAMERA] Releasing camera...")
        self.cap.release()
        
        # Close windows
        cv2.destroyAllWindows()
        
        print("[SENTRY] Shutdown complete")


# ========================================
# Main Entry Point
# ========================================

def main():
    """Main entry point."""
    try:
        sentry = PersonTrackingSentry()
        sentry.run()
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
