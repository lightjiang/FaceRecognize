# coding=utf-8
import dlib
import cv2
from face_detector import FaceDetector


class FaceLandMarks(FaceDetector):
    def __init__(self, ):
        super().__init__()
        self.marks_detector = dlib.shape_predictor(self.base_path + '/model/shape_predictor_68_face_landmarks.dat')

    def detect_marks(self, rect):
        """

        :param rect: must have properties: left, top, right, bottom
        :return: marks: list, contains face features' position like (x, y)
        """
        shape = self.marks_detector(self.img, rect)
        marks = []
        for index, pt in enumerate(shape.parts()):
            pos = (pt.x, pt.y)
            marks.append(pos)
        return marks


if __name__ == '__main__':
    f = FaceLandMarks()
    f.load_img('/home/light/PycharmProjects/ImageProcess/data/crowd/1.jpg')
    faces = f.detect_face()
    for face in faces:
        marks = f.detect_marks(face['position'])
        f.add_marks(marks)
        f.add_faces(faces)
    f.show()
    cv2.waitKey(0)
