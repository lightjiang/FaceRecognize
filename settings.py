# coding = utf-8
import os
BASEPATH = os.path.abspath(os.path.dirname(__file__))


face_recognition_model_v1 = os.path.join(BASEPATH, 'model/dlib_face_recognition_resnet_model_v1.dat')
mmod_human_face_detector = os.path.join(BASEPATH, 'model/mmod_human_face_detector.dat')
shape_predictor_5_face_landmarks = os.path.join(BASEPATH, 'model/shape_predictor_5_face_landmarks.dat')
shape_predictor_68_face_landmarks = os.path.join(BASEPATH, 'model/shape_predictor_68_face_landmarks.dat')

known_faces_path = os.path.join(BASEPATH, 'results/known_faces')
known_vectors_path = os.path.join(BASEPATH, 'results/known_vectors')
