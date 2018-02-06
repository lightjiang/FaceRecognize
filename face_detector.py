# coding=utf-8
import dlib
import cv2
import os
import time
from base import Base


class FaceDetector(Base):
    """
    based on HOG
    """
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        super().__init__()

    def detect_face(self):
        if self.img is None:
            raise AttributeError('please load img before detect')
        t = time.time()
        dets, scores, idx = self.face_detector.run(self.img, 1, -1)
        res = []
        for index, face in enumerate(dets):
            temp = {
                'position': face,
                'score': scores[index]
            }
            if scores[index] > 0:
                res.append(temp)
            # else:
            #     self.add_face_on_img(temp, color=(255, 0, 0))
        print("Number of faces detected: {}\n takes:{}s".format(len(res), time.time() - t))
        return res


class FaceDetectorCNN(FaceDetector):
    """
    use CNN model
    too long to detect, normally in 10s, but sometimes in decades seconds

    """
    def __init__(self):
        super().__init__()
        self.face_detector = dlib.cnn_face_detection_model_v1(
            '/home/light/PycharmProjects/ImageProcess/model/mmod_human_face_detector.dat')

    def detect_face(self):
        t = time.time()
        dets= self.face_detector(self.img, 1)
        print("Number of faces detected: {}\n takes:{}s".format(len(dets), time.time()-t))
        res = []
        for index, face in enumerate(dets):
            temp = {
                'position': face.rect,
                'score': face.confidence
            }
            if face.confidence > -0.1:
                res.append(temp)
            else:
                self.add_face_on_img(temp, color=(255, 0, 0))
        return res


def detect_imgs():
    dir_path = '/home/light/PycharmProjects/ImageProcess/data/profile'
    temp = os.listdir(dir_path)
    temp.sort()
    f = FaceDetector()
    for i in temp:
        f.load_img('data/profile/%s' % i)
        faces = f.detect_face()
        f.add_faces(faces)
        f.show()
        # type 0 to continue
        cv2.waitKey(0)


if __name__ == '__main__':
    f = FaceDetector()
    f.load_img('/home/light/PycharmProjects/ImageProcess/data/frontal/0001.jpg')
    faces = f.detect_face()
    f.add_faces(faces)
    f.show()
    cv2.waitKey(0)
