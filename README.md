# Installation
### 1. Create Virtual Environment
```
python -m venv .
```
### 2. Activate Virtual Environment

### 3. Install Appropriate Torch Version
* CPU-only + Python 3.10.x + Windows
```
pip install https://download.pytorch.org/whl/cpu/torch-2.2.2%2Bcpu-cp310-cp310-win_amd64.whl#sha256=fc29dda2795dd7220d769c5926b1c50ddac9b4827897e30a10467063691cdf54
```
* CPU-only + Python 3.11.x + Windows
```
pip install https://download.pytorch.org/whl/cpu/torch-2.2.2%2Bcpu-cp311-cp311-win_amd64.whl#sha256=88e63c916e3275fa30a220ee736423a95573b96072ded85e5c0171fd8f37a755
```
* CUDA + Python 3.10 + Windows
```
pip install https://download.pytorch.org/whl/cu121/torch-2.2.2%2Bcu121-cp310-cp310-win_amd64.whl#sha256=d300055aac0e2063f9a2659924e9766605db06d5683532c6eabbdef6bec865dd
```
* CUDA + Python 3.11 + Windows
```
pip install https://download.pytorch.org/whl/cu121/torch-2.2.2%2Bcu121-cp311-cp311-win_amd64.whl#sha256=efbcfdd4399197d06b32f7c0e1711c615188cdd65427b933648c7478fb880b3f
```
### 4. Install CUDA Dependencies
Only run this command if using a CUDA version of Torch above.
```
pip install nvidia-cublas-cu12==12.1.3.1 nvidia-cuda-nvrtc-cu12==12.1.105 nvidia-cuda-runtime-cu12==12.1.105 nvidia-cudnn-cu12==8.9.7.29 --no-deps
```
### 5. Install Other Dependencies
```
pip install --no-deps -r requirements.txt
```
# Usage
Navigate to the directory and activate the virtual environment, then run:
```
python keybert_pyside.py
```
