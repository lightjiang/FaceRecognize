#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <dlib/opencv.h>

void mark_face(cv::Mat & image, dlib::rectangle & rect);

void mark_face(cv::Mat & image, dlib::rectangle & rect) {
    cv::rectangle(image, cv::Point(rect.left(), rect.top()), cv::Point(rect.right(), rect.bottom()),cv::Scalar(255,255,0),1,1,0);
}