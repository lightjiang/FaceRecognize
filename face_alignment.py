# coding=utf-8
import dlib
import cv2
from face_landmarks import FaceLandMarks


class FaceAlignment(FaceLandMarks):
    def __init__(self):
        super().__init__()

    def alignment_face(self, rect):
        faces = dlib.full_object_detections()
        faces.append(self.marks_detector(self.img, rect))
        image = dlib.get_face_chip(self.img, faces[0])
        self.show(image, 'alignment_face')


if __name__ == '__main__':
    f = FaceAlignment()
    f.load_img('/home/light/PycharmProjects/ImageProcess/data/crowd/0001.jpg')
    faces = f.detect_face()
    for face in faces:
        f.alignment_face(face['position'])
        cv2.waitKey(0)
