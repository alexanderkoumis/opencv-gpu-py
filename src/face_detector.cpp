#include "face_detector.hpp"

// #include <functional>                        // []()
#include <iostream>                          // cout
#include <memory>                            // unique_ptr
#include <mutex>                             // call_once
#include <thread>                            // call_once
#include <vector>                            // vector

#ifdef OPENCV_2
  // OpenCV 2 includes
  #include <opencv2/contrib/contrib.hpp>
  #include <opencv2/core/core.hpp>           // Mat, Point, Scalar, Size
  #include <opencv2/core/gpumat.hpp>         // GpuMat
  #include <opencv2/highgui/highgui.hpp>     // imread, imshow, waitKey
  #include <opencv2/imgproc/imgproc.hpp>     // cvtColor, equalizeHist
  #include <opencv2/objdetect/objdetect.hpp> // CascadeClassifier
#else
  // OpenCV 3 includes
  #include <opencv2/core.hpp>                // Mat, Point, Scalar, Size
  #include <opencv2/core/cuda.hpp>           // cuda::isCompatible
  #include <opencv2/highgui.hpp>             // imshow, waitKey
  #include <opencv2/imgcodecs.hpp>           // imread
  #include <opencv2/imgproc.hpp>             // cvtColor, equalizeHist
  #include <opencv2/objdetect.hpp>           // CascadeClassifier
#endif

FaceDetector::FaceDetector() : gpu_(false)
{}

void FaceDetector::Init(const std::string& cascade_file, bool gpu)
{
  cascade_file_ = cascade_file;
  gpu_ = gpu;
}

bool FaceDetector::Detect(const char* image_file,
                          std::vector<cv::Rect>& face_rects)
{
  cv::Mat frame = cv::imread(image_file, 0);
  cv::equalizeHist(frame, frame);
  static std::once_flag load_flag;
  if (cascade_file_.empty())
  {
    std::cout << "Error: Need to load cascade file" << std::endl;
    return false;
  }
  if (gpu_)
  {
    #ifdef OPENCV_2
      static cv::gpu::CascadeClassifier_GPU cascade_gpu;
      std::call_once(load_flag, [&]() {
        cascade_gpu.load(cascade_file_);
      });
      cv::gpu::GpuMat frame_gpu(frame);
      cv::gpu::GpuMat faces_gpu;
      cv::Mat faces_cpu;
      std::cout << "hi" << std::endl;
      int num_faces = cascade_gpu.detectMultiScale(frame_gpu, faces_gpu, 1.2);
      faces_gpu.colRange(0, num_faces).download(faces_cpu);
      cv::Rect* faces = faces_cpu.ptr<cv::Rect>();
      for (int i = 0; i < num_faces; ++i)
      {
        face_rects.push_back(faces[i]);
      }
    #else
      static
      auto cascade_gpu_ptr = cv::cuda::CascadeClassifier::create(cascade_file_);
      cv::cuda::GpuMat frame_gpu(frame);
      cv::cuda::GpuMat faces_gpu;
      cascade_gpu_ptr->detectMultiScale(frame_gpu, faces_gpu);
      cascade_gpu_ptr->convert(faces_gpu, face_rects);
    #endif
  }
  else
  {
    // CPU detector
    static cv::CascadeClassifier cascade_cpu;
    std::call_once(load_flag, [&]() {
      cascade_cpu.load(cascade_file_);
    });
    cascade_cpu.detectMultiScale(frame, face_rects, 1.1, 2,
                                  0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
  }
  return true;
}
