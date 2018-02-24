#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "mybase.hpp"

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    try
    {
        if (argc == 1)
        {
            cout << "Give some image files as arguments to this program." << endl;
            return 0;
        }

        frontal_face_detector detector = get_frontal_face_detector();

        // Loop over all the images provided on the command line.
        for (int i = 1; i < argc; ++i)
        {
            cout << "processing image " << argv[i] << endl;
            cv::Mat image;
            image = cv::imread(argv[i], cv::IMREAD_COLOR);
            dlib::cv_image<rgb_pixel> img(image);
            std::vector<rectangle> dets = detector(img);
            cout << "Number of faces detected: " << dets.size() << dets[0].left() <<dets[1] << image.size() << endl<< image;
            for (auto temp_point: dets){
                cout << temp_point<< temp_point.left() <<endl;
                mark_face(image, temp_point);
                // cv::rectangle(image, cv::Point(temp_point.left(), temp_point.top()), cv::Point(temp_point.right(), temp_point.bottom()),cv::Scalar(255,0,0),1,1,0);
            }
            // cv::rectangle(image, cv::Point(dets[0][0][0], dets[0][0][1]));
            cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); // Create a window for display.
            cv::imshow("Display window", image);                // Show our image inside it.
            cv::waitKey(0);                                     // Wait for a keystroke in the window
        }
    }
    catch (exception &e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}
