#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <sys/time.h>

#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>

#include "mybase.hpp"

using namespace cv;
using namespace std;

struct face {
    string name = "";
    dlib::rectangle position;
};

int main()
{
    VideoCapture capture(0); //如果是笔记本，0打开的是自带的摄像头，1 打开外接的相机
    // cv::VideoCapture capture("out.avi");
    Mat frame;
    namedWindow("cam");

    int fps = capture.get(CAP_PROP_FPS); //获取摄像头的帧率
    // capture.set(CV_CAP_PROP_FPS, 30);
    // capture.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
    // capture.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
    // capture.set(CV_CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'));
    cout << capture.get(CAP_PROP_FRAME_WIDTH) << capture.get(CAP_PROP_FRAME_HEIGHT) << fps << endl;
    // if(fps <= 0 )fps = 25;
    // //设置视频的格式

    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    std::vector<dlib::correlation_tracker> trackers(5);
    cout << trackers.size()<< endl;
    if (!capture.isOpened()) //判断摄像头是否打开
    {
        cout << "open video faild";
        return 0;
    }
    cout << "open video success" << endl;
    double t = 0;
    bool run_status = true;
    unsigned int update_status = 0;
    unsigned int track_status = 0;
    std::vector<face> faces;
    while (run_status)
    {
        double temp_t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        t = (double)cv::getTickCount();
        // capture.read(frame);  //读取视频帧
        switch (update_status)
        {
        case 0:{
            capture.read(frame);
            if (track_status==1){
                dlib::cv_image<dlib::rgb_pixel> img(frame);
                for (unsigned int i = 0; i<faces.size(); i++){
                    trackers[i].update(img);
                    dlib::drectangle rect = trackers[i].get_position();
                    cv::rectangle(frame, cv::Point(rect.left(), rect.top()), cv::Point(rect.right(), rect.bottom()),cv::Scalar(255,255,0),1,1,0);
                }
            }
            break;
        }
        case 1:
            break;
        default:
            break;
        }
        if (frame.empty())
            break;
        switch (waitKey(10))
        {
        case 'q':
            run_status = false;
            break;
        case 'f':
            std::cout << capture.get(CAP_PROP_FPS) << "1: " << 1.0 / temp_t << endl;
            break;
        case 'p':
            update_status = (update_status == 1) ? 0 : 1;
            break;
        case 'd':
        {
            faces.clear();
            track_status = 1;
            dlib::cv_image<dlib::rgb_pixel> img(frame);
            std::vector<dlib::rectangle> dets = detector(img);
            for (unsigned int i = 0; i<dets.size(); i++){
                trackers[i].start_track(img, dets[i]);
                face temp_face;
                temp_face.name = (char) i;
                temp_face.position = dets[i];
                faces.push_back(temp_face);
            }
            for (auto &temp_point : dets)
            {
                mark_face(frame, temp_point);
            }
        }
        default:
            break;
        }
        imshow("cam", frame); //显示视频帧
    }
    return 0;
}
