cmake_minimum_required(VERSION 2.8)  
project(test)  
find_package(OpenCV REQUIRED)  
add_subdirectory(~/program/dlib/ dlib_build)

add_executable(face_detector face_detector.cpp mybase.hpp)
target_link_libraries(face_detector dlib::dlib)
target_link_libraries(face_detector ${OpenCV_LIBS})


add_executable(video_tracker video_tracker.cpp)
target_link_libraries(video_tracker dlib::dlib)
target_link_libraries(video_tracker ${OpenCV_LIBS})

add_executable(video video.cpp)
target_link_libraries(video ${OpenCV_LIBS})

add_definitions("-Wall -g")
#define BASEPATH "/home/light/PycharmProjects/FaceRecognize/"
