# GPU Setup Guide for RTX 3060

## Current Status - UPDATED
- **TensorFlow Version:** 2.10.0 GPU (installed ✓)
- **NVIDIA Driver:** 580.97 ✅ (Excellent!)
- **GPU Hardware:** RTX 3060 12GB ✅
- **CUDA Installed:** 13.0 ❌ (Wrong version - need 11.2)
- **Problem:** TensorFlow 2.10 needs CUDA 11.2, not 13.0
- **Performance:** Currently CPU only (~5-10 eps/sec)
- **Expected with correct CUDA:** ~75-150 eps/sec

## The Issue
You have CUDA 13.0 installed, but TensorFlow 2.10 needs CUDA 11.2. The error messages show:
```
Could not load dynamic library 'cudart64_110.dll'
Could not load dynamic library 'cublas64_11.dll'
```

## SOLUTION: Install CUDA 11.2 (Recommended)

### Step 1: Download CUDA 11.2
1. Go to: https://developer.nvidia.com/cuda-11.2.0-download-archive
2. Select: **Windows → x86_64 → 10 → exe (local)**
3. Download the installer (~2.5 GB)

### Step 2: Install CUDA 11.2
1. **Run the installer**
2. **Choose "Custom (Advanced)" installation**
3. **Uncheck these components:**
   - ❌ Display Driver (you already have better drivers!)
   - ❌ Visual Studio Integration (not needed)
   - ❌ Nsight tools (not needed)
4. **Keep checked:**
   - ✓ CUDA Toolkit 11.2
   - ✓ CUDA Runtime
   - ✓ CUDA Development
5. Click Install

**Note:** CUDA 11.2 and 13.0 can coexist peacefully! The installer will put it in:
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2`

### Step 3: Download cuDNN 8.1 for CUDA 11.2
1. Go to: https://developer.nvidia.com/rdp/cudnn-archive
2. Find: **"Download cuDNN v8.1.1 for CUDA 11.0, 11.1, and 11.2"**
3. Choose: "cuDNN Library for Windows (x86)"
4. Extract the ZIP file

### Step 4: Copy cuDNN files to CUDA 11.2 folder
From the extracted ZIP, copy files to CUDA folder:
- **bin\cudnn64_8.dll** → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin\`
- **include\*.h** → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include\`
- **lib\x64\*.lib** → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib\x64\`

### Step 5: Update PATH Environment Variable
1. Search Windows: "Environment Variables"
2. Edit "System PATH"
3. **Add these to the TOP** (before CUDA 13.0 entries):
   - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin`
   - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp`
4. Click OK, OK, OK

### Step 6: RESTART PowerShell and Test
**Important:** Close and reopen PowerShell, then run:
```powershell
python test_gpu.py
```

Should now show: **"✓ GPU DETECTED!"**

## Option 2: Use CPU (Current Setup - Already Optimized)

The code is already optimized for CPU:
- Reduced batch size (128 instead of 256)
- Enabled CPU parallelism
- Reduced logging frequency
- Training every 8 steps instead of 4

Expected performance: ~10-15 episodes/second (acceptable for one-time training)

**Just run:** `.\run_phase1.bat` and wait ~10 minutes

## Quick Decision Guide

**Install GPU support if:**
- ✓ You plan to do multiple training runs
- ✓ You want to experiment with different parameters
- ✓ 20-30 minutes of setup is worth 15-30x speedup
- ✓ You enjoy the satisfaction of maxing out your GPU 

**Stay on CPU if:**
- ✓ This is a one-time training run
- ✓ You're okay waiting ~10 minutes vs ~30 seconds
- ✓ You don't want to download 2.5GB CUDA installer
- ✓ You just want to see the AI work without hassle

## Summary

**Current state:** TensorFlow 2.10 GPU installed, but needs CUDA 11.2 libraries to actually use your RTX 3060.

**To enable GPU:** Install CUDA 11.2 + cuDNN 8.1 (follow steps above, ~20-30 min)

**To skip GPU:** Just use CPU - it's already optimized and will work fine for 5000 episodes (~10 min total)

## Performance Comparison

| Setup | Episodes/sec | Time for 5000 eps |
|-------|--------------|-------------------|
| CPU (current) | ~5 | ~17 minutes |
| CPU (optimized) | ~10-15 | ~8-10 minutes |
| **GPU (RTX 3060)** | **~75-150** | **~30-60 seconds** |

## My Recommendation

For a one-time 5000 episode training, the optimized CPU version is acceptable (~10 minutes).

For multiple training runs or longer training (50k+ episodes), **definitely set up GPU support** - it's worth the 30 minutes of setup time.

