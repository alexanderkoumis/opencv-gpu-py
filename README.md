# Python OpenCV GPU Face Detection

Python wrapper for GPU CascadeClassifier, should work with OpenCV 2 and 3. Will fall back to CPU CascadeClassifier if CUDA isn't installed, but if the CPU version enough, just use stock [OpenCV Python](http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html). 

## Installation

First [install CUDA](https://developer.nvidia.com/cuda-downloads).

On Debian/Ubuntu, after downloading the `.deb` provided by NVIDIA:

```bash
sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb # Filename will vary
sudo apt-get update && sudo apt-get install cuda
echo "export CUDA_HOME=/usr/local/cuda-7.5" >> ~/.bashrc
echo "export PATH=$PATH:$CUDA_HOME/bin" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64" >> ~/.bashrc
```

Build/install OpenCV 3 from source:

```bash
sudo apt-get update && sudo apt-get install -y \
    build-essential cmake git libgtk2.0-dev pkg-config \
    libavcodec-dev libavformat-dev libswscale-dev python-dev \
    python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev
git clone https://github.com/Itseez/opencv
mkdir opencv/build && cd opencv/build
# Build OpenCV 2.4 instead of 3:
# mkdir opencv/build && cmake opencv && git checkout 2.4 && cd build
cmake .. && make -j$(nproc) && sudo make install
```

Install opencv-gpu-py:

```bash
git clone https://github.com/alexanderkoumis/opencv-gpu-py
pip install --user -e opencv-gpu-py
```

Test:

```bash
cd opencv-gpu-py
python test/test.py
```

## Usage

```python
# Full example in test/test.py
import cv2gpu

if cv2gpu.is_cuda_compatible():
    cv2gpu.init_gpu_detector('cascade_frontalface_gpu.xml')
else:
    cv2gpu.init_cpu_detector('cascade_frontalface_cpu.xml')

for (x, y, w, h) in cv2gpu.find_faces('image.jpg'):
    # Do something with face rectangle
```
