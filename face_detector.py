# coding=utf-8
"""
light
人脸检测
"""
import dlib
import cv2
import os
import time
import settings
import uuid
from base import Base


class FaceDetector(Base):
    """
    based on HOG
    """
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        super().__init__()

    def detect_face(self, log_status=True):
        if self.img is None:
            raise AttributeError('please load img before detect')
        t = time.time()
        dets, scores, idx = self.face_detector.run(self.img, 1, -1)
        res = []
        for index, face in enumerate(dets):
            temp = {
                'id': str(uuid.uuid1()),
                'position': face,
                'score': scores[index],
                'src': self.img_path
            }
            if scores[index] > 0:
                res.append(temp)
        if log_status:
            print("Detecting faces takes: {}s\nNumber of faces detected: {}".format(time.time() - t, len(res)))
        return res

    def detect_faces_from_imgs(self, imgs: list):
        t = time.time()
        res = []
        for img_path in imgs:
            self.load_img(img_path)
            res += self.detect_face(log_status=False)
        self.img = None
        print("Detecting faces takes: {}s\nNumber of faces detected: {}".format(time.time() - t, len(res)))
        return res


class FaceDetectorCNN(FaceDetector):
    """
    use CNN model
    too long to detect, normally in 10s,  sometimes in decades seconds

    """
    def __init__(self):
        super().__init__()
        self.face_detector = dlib.cnn_face_detection_model_v1(settings.mmod_human_face_detector)

    def detect_face(self, log_status=True):
        t = time.time()
        dets= self.face_detector(self.img, 1)
        res = []
        for index, face in enumerate(dets):
            temp = {
                'id': str(uuid.uuid1()),
                'position': face.rect,
                'score': face.confidence,
                'src': self.img_path
            }
            if face.confidence > -0.1:
                res.append(temp)
        if log_status:
            print("Detecting face takes: {}s\nNumber of faces detected: {}".format(time.time() - t, len(res)))
        return res


def detect_imgs():
    dir_path = 'data/profile'
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
    f.load_img('data/crowd/0001.jpg')
    faces = f.detect_face()
    f.add_faces(faces)
    f.show()
    cv2.waitKey(0)
