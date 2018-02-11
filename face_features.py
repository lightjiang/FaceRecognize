# coding=utf-8
import dlib
import cv2
import settings
from face_detector import FaceDetector


class FaceFeatures(FaceDetector):
    def __init__(self, ):
        super().__init__()
        self.marks_detector = dlib.shape_predictor(settings.shape_predictor_68_face_landmarks)

    def detect_features(self, rect):
        """
        :param rect:  must bu dlib.rect object, not 2d tuple
        :return:
        """
        features = self.marks_detector(self.img, rect)
        return features

    def get_marks(self, rect):
        """
        :param rect: must have method: left, top, right, bottom
        :return: marks: list, contains face features' position like (x, y)
        """
        features = self.detect_features(rect)
        marks = []
        for index, pt in enumerate(features.parts()):
            pos = (pt.x, pt.y)
            marks.append(pos)
        return marks


if __name__ == '__main__':
    f = FaceFeatures()
    f.load_img('results/known/e7420ac0-0d91-11e8-93de-a0c589189417.jpg')
    faces = f.detect_face()
    for face in faces:
        marks = f.get_marks(face['position'])
        f.add_marks(marks)
        f.add_faces(faces)
    f.show()
    cv2.waitKey(0)
