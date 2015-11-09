#ifndef OPENCVVERSION_HPP
#define OPENCVVERSION_HPP

#include <opencv2/core/version.hpp>

#ifdef CV_VERSION_EPOCH
  #define OPENCV_2
#else
  #define OPENCV_3
#endif

#endif
