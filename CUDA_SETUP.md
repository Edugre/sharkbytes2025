# PyTorch + CUDA Setup Guide

## ✅ Complete Installation

The `requirements.txt` includes everything needed for GPU-accelerated tracking:

### What's Included:
- ✅ **PyTorch 2.5.0 with CUDA 12.6** - GPU acceleration
- ✅ **TorchVision 0.20.0** - Vision utilities
- ✅ **YOLOv11** - Latest detection model
- ✅ **DeepSORT** - Multi-object tracking
- ✅ **ServoKit** - Hardware control
- ✅ **OpenCV** - Camera interface

---

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
cd "c:\Users\Jorge Taban\Documents\sharkbytes2025"
pip install -r requirements.txt
```

This installs:
- PyTorch 2.5.0 with CUDA 12.6 support (JetPack 6.x wheels)
- All required packages for person tracking

### 2. Verify CUDA Setup

```bash
python verify_cuda.py
```

**Expected Output:**
```
✅ PyTorch installed: 2.5.0
✅ CUDA available: YES
   CUDA Version: 12.6
   GPU Device: Orin
   GPU Memory: X.XX GB
✅ YOLOv11n model loaded successfully
   Model will use: CUDA
```

### 3. Run the Tracker

```bash
python person_tracking_sentry.py
```

---

## 🔍 CUDA Verification

The system automatically detects and uses CUDA when available:

### At Startup:
```
[GPU] CUDA Available: Orin
[GPU] CUDA Version: 12.6
[GPU] PyTorch Version: 2.5.0
[YOLO] Loading YOLOv11 model...
[YOLO] Model loaded successfully
```

### FP16 Acceleration:
- **Enabled automatically** when CUDA is available
- **2x faster** inference than FP32
- Set in `person_tracking_sentry.py`:
  ```python
  half=True,   # Enable FP16 for CUDA acceleration
  ```

---

## ⚡ Performance Expectations

### With CUDA (GPU):
- **Inference**: 10-20ms per frame
- **FPS**: 25-30 fps
- **FP16 enabled**: ~2x faster than FP32

### Without CUDA (CPU):
- **Inference**: 100-200ms per frame  
- **FPS**: 5-10 fps
- **FP16 disabled**: Uses FP32

---

## 🔧 Troubleshooting

### CUDA Not Available

**Check 1: Verify PyTorch Installation**
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

**Check 2: Verify CUDA Drivers**
```bash
nvidia-smi
```

**Check 3: Reinstall PyTorch**
```bash
pip uninstall torch torchvision
pip install -r requirements.txt
```

### Wrong PyTorch Version

The `requirements.txt` specifies:
- **PyTorch 2.5.0** with CUDA 12.6 (JetPack 6.x wheels)
- Pre-built for Jetson ARM64 architecture

**If you get CPU-only PyTorch:**
- Make sure you're using the requirements.txt URLs
- Don't install from PyPI directly (`pip install torch` will be CPU-only)

### Slow Inference Despite CUDA

**Check FP16 is enabled:**
```python
# In person_tracking_sentry.py, line ~287
half=True,   # Should be True for CUDA
```

**Check model is on GPU:**
```bash
python verify_cuda.py
```

---

## 📊 Benchmark Results

Run the verification script to benchmark your system:

```bash
python verify_cuda.py
```

**Expected Results (with CUDA):**
```
📊 Results:
   Average time: 15.2ms
   Average FPS: 65.8
   ⚡ GPU-accelerated with FP16
```

**Expected Results (CPU only):**
```
📊 Results:
   Average time: 156.3ms
   Average FPS: 6.4
   🐌 CPU-only (expect 5-10x slower)
```

---

## 🎯 CUDA Features Enabled

### 1. Automatic Device Selection
```python
# In person_tracking_sentry.py
# Device is auto-detected (will use CUDA if available, CPU otherwise)
results = self.yolo_model(frame, ...)
```

### 2. FP16 Acceleration
```python
half=True,   # Enable FP16 for CUDA acceleration (2x faster on GPU)
```

### 3. GPU Memory Optimization
```python
imgsz=320,        # Small input size = less GPU memory
max_det=5,        # Limit detections = faster processing
classes=[0],      # Person only = fewer computations
```

---

## 📦 Requirements.txt Breakdown

### PyTorch with CUDA (JetPack 6.x)
```
torch @ https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
```
- Pre-built for Jetson Orin
- CUDA 12.6 support
- Python 3.10 compatible

### TorchVision (ARM64)
```
torchvision @ https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl
```
- Compatible with PyTorch 2.5.0
- ARM64 architecture

### YOLOv11
```
ultralytics
```
- Latest YOLO version
- Includes YOLOv11n model
- GPU-optimized

---

## 🎓 Learn More

- **PyTorch CUDA Guide**: https://pytorch.org/get-started/locally/
- **YOLOv11 Docs**: https://docs.ultralytics.com/
- **Jetson PyTorch**: https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/

---

## ✅ Quick Checklist

Before running the tracker:

- [ ] PyTorch 2.5.0 installed
- [ ] CUDA 12.6 available
- [ ] `verify_cuda.py` passes all checks
- [ ] YOLOv11n model downloads successfully
- [ ] FP16 enabled in code (`half=True`)
- [ ] Expected FPS: 25-30 (with CUDA) or 5-10 (CPU only)

---

**Ready to track!** 🎯

Run: `python person_tracking_sentry.py`
