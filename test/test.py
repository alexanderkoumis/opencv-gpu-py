#!/usr/bin/env python

import os
import cv2gpu

this_dir = os.path.dirname(os.path.realpath(__file__))
image_file = os.path.join(this_dir, 'obama.jpg')
image_file_b = os.path.join(this_dir, 'lilb.jpg')
cascade_file_cpu = os.path.join(this_dir, 'haarcascade_frontalface_default.xml')
cascade_file_gpu = os.path.join(this_dir, 'haarcascade_frontalface_default_cuda.xml')

def main():

    if cv2gpu.is_cuda_compatible():
        cv2gpu.init_gpu_detector(cascade_file_gpu)
    else:
        cv2gpu.init_cpu_detector(cascade_file_cpu)

    print 'pic 1'
    
    faces = cv2gpu.find_faces(image_file)
    for (x, y, w, h) in faces:
        print x, y, w, h

    print 'pic 2'

    faces = cv2gpu.find_faces(image_file_b)
    for (x, y, w, h) in faces:
        print x, y, w, h


if __name__ == '__main__':
    main()
