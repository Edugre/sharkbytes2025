#!/usr/bin/env python3
"""
CUDA & PyTorch Verification Script
===================================
Verifies that PyTorch with CUDA is properly installed and configured.

Run this script to check:
- PyTorch installation
- CUDA availability
- GPU device information
- YOLOv11 model loading
- Inference speed benchmark
"""

import sys

def check_pytorch():
    """Check PyTorch installation and CUDA availability."""
    print("="*60)
    print("PyTorch & CUDA Verification")
    print("="*60 + "\n")
    
    # Check PyTorch
    try:
        import torch
        print(f"✅ PyTorch installed: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not installed!")
        print("\nInstall with:")
        print("  pip install -r requirements.txt")
        return False
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA available: YES")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"   Compute Capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("⚠️  CUDA not available - will use CPU (slower)")
        print("   This is normal on non-GPU systems")
    
    # Check TorchVision
    try:
        import torchvision
        print(f"✅ TorchVision installed: {torchvision.__version__}")
    except ImportError:
        print("⚠️  TorchVision not installed")
    
    print()
    return True


def check_yolo():
    """Check YOLO installation and model loading."""
    print("="*60)
    print("YOLOv11 Model Check")
    print("="*60 + "\n")
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics installed")
        
        # Try loading model
        print("\nLoading YOLOv11n model...")
        model = YOLO('yolo11n.pt')
        print("✅ YOLOv11n model loaded successfully")
        
        # Check device
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   Model will use: {device.upper()}")
        
        return True
        
    except ImportError:
        print("❌ Ultralytics not installed!")
        print("\nInstall with:")
        print("  pip install ultralytics")
        return False
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        return False


def benchmark_inference():
    """Benchmark YOLO inference speed."""
    print("\n" + "="*60)
    print("Inference Speed Benchmark")
    print("="*60 + "\n")
    
    try:
        import torch
        import cv2
        import numpy as np
        from ultralytics import YOLO
        import time
        
        # Load model
        print("Loading model...")
        model = YOLO('yolo11n.pt')
        
        # Create dummy frame
        frame = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        
        # Warm up (first inference is slower)
        print("Warming up...")
        _ = model(frame, verbose=False)
        
        # Benchmark
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\nRunning 10 inferences on {device.upper()}...")
        
        times = []
        for i in range(10):
            start = time.time()
            results = model(
                frame,
                verbose=False,
                conf=0.35,
                imgsz=320,
                classes=[0],
                half=torch.cuda.is_available(),  # FP16 if CUDA
            )
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Inference {i+1}: {elapsed*1000:.1f}ms")
        
        avg_time = np.mean(times)
        avg_fps = 1.0 / avg_time
        
        print(f"\n📊 Results:")
        print(f"   Average time: {avg_time*1000:.1f}ms")
        print(f"   Average FPS: {avg_fps:.1f}")
        
        if torch.cuda.is_available():
            print(f"   ⚡ GPU-accelerated with FP16")
        else:
            print(f"   🐌 CPU-only (expect 5-10x slower)")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False


def check_dependencies():
    """Check other required dependencies."""
    print("\n" + "="*60)
    print("Other Dependencies")
    print("="*60 + "\n")
    
    deps = [
        ('cv2', 'opencv-python'),
        ('deep_sort_realtime', 'deep-sort-realtime'),
        ('adafruit_servokit', 'adafruit-circuitpython-servokit'),
        ('numpy', 'numpy'),
    ]
    
    all_ok = True
    for module, package in deps:
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} not installed")
            all_ok = False
    
    return all_ok


def main():
    """Run all verification checks."""
    print("\n")
    
    results = []
    
    # Run checks
    results.append(("PyTorch & CUDA", check_pytorch()))
    results.append(("YOLOv11", check_yolo()))
    results.append(("Dependencies", check_dependencies()))
    
    # Only benchmark if PyTorch is available
    try:
        import torch
        if torch.cuda.is_available():
            results.append(("Inference Benchmark", benchmark_inference()))
    except:
        pass
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(success for _, success in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 All checks passed! System ready for tracking.")
    else:
        print("⚠️  Some checks failed. Install missing dependencies:")
        print("   pip install -r requirements.txt")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
