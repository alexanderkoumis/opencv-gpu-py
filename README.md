# Python OpenCV GPU Face Detection

Python wrapper for GPU CascadeClassifier, should work with OpenCV 2 and 3.

## Installation

```bash
cd opencv-gpu-py
pip install -e . --user
```

## Usage

```python
if cv2gpu.is_cuda_compatible():
    cv2gpu.init_gpu_detector(cascade_file_gpu)
else:
    cv2gpu.init_cpu_detector(cascade_file_cpu)

for (x, y, w, h) in cv2gpu.find_faces(image_file):
    # Do something with face rectangle
```

See `test/test.py` for full example.
