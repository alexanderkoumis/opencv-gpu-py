#!/usr/bin/env python

import os
from distutils.core import setup, Extension
from subprocess import Popen, PIPE

# Pkg-config function from https://gist.github.com/abergmeier/9488990

def pkgconfig(*packages, **kw):
    flag_map = {
        '-I': 'include_dirs',
        '-L': 'library_dirs',
        '-l': 'libraries'
    }
    
    env = os.environ.copy()
    kw = {}
    for key, value in flag_map.iteritems():
        kw[value] = []

    command = ['pkg-config', '--libs', '--cflags', ' '.join(packages)]
    for token in Popen(command, stdout=PIPE, env=env).communicate()[0].split():
        key = token[:2]
        try:
            arg = flag_map[key]
            value = token[2:]
        except KeyError:
            arg = 'extra_link_args'
            value = token
        
        kw.setdefault(arg, []).append(value)
    for key, value in kw.iteritems(): # remove duplicated
        kw[key] = list(set(value))
    return kw

opencv_deps = pkgconfig('opencv')

this_dir = os.path.dirname(os.path.realpath(__file__))

cv2gpumodule = Extension('cv2gpu', 
                  define_macros = [('MAJOR_VERSION', '1'),
                                   ('MINOR_VERSION', '0')],
                  sources = [os.path.join('src', 'face_detector.cpp'), os.path.join('src', 'cv2gpu.cpp')],
                  include_dirs = opencv_deps['include_dirs'],
                  libraries = opencv_deps['libraries'],
                  library_dirs = opencv_deps['library_dirs'],
                  extra_compile_args = ['-std=c++11'])

setup (name = 'cv2gpu',
       version = '1.0',
       description = 'OpenCV GPU Bindings',
       author = 'Alexander Koumis and Matthew Carlis',
       author_email = 'alexander.koumis@sjsu.edu, matthew.carlis@sjsu.edu',
       url = 'https://docs.python.org/extending/building',
       long_description = '''
OpenCV GPU Bindings
''',
       ext_modules = [cv2gpumodule])
