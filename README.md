# FaceRecognize

## 目录结构
``` bash
.
├── cpp_src     # cpp 源码
│   ├── CMakeLists.txt
│   ├── face_detector.cpp
│   ├── mybase.hpp
│   └── video_tracker.cpp
├── data        # 图片数据
│   ├── auto_download_img
│   ├── crowd
│   ├── frontal
│   ├── profile
│   └── star
├── model       # 模型 下载链接见 http://dlib.net/files/
│   ├── dlib_face_recognition_resnet_model_v1.dat
│   ├── mmod_human_face_detector.dat
│   ├── shape_predictor_5_face_landmarks.dat
│   └── shape_predictor_68_face_landmarks.dat
├── results     # 存储人脸识别数据结果
│   ├── known   # 人脸图片
│   │   ├── fee26871-0f30-11e8-93de-a0c589189417.jpg
|   |   ..... 
│   │   └── fef129db-0da9-11e8-93de-a0c589189417.jpg
│   ├── known_faces     # 存人脸数据，如名称，来源，图片保存路径
│   ├── known_vectors   # 存人脸特征
│   └── vedio
├── script
│   ├── download_baidu_img.py
│   ├── download_star_img.py
│   ├── __init__.py
│   ├── reform_star_data.py
│   └── rename_img_in_dir.py
├── base.py
├── face_alignment.py
├── face_clustering.py
├── face_detector.py
├── face_features.py
├── face_recognition.py
├── face_recognize_in_vedio.py
├── settings.py
├── uming.ttc
├── utils.py
├── README.md
└── vedio_base.py
```
