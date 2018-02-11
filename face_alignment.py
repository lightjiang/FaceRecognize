# coding=utf-8
import dlib
import cv2
from face_features import FaceFeatures


class FaceAlignment(FaceFeatures):
    def alignment_face(self, features):
        image = dlib.get_face_chip(self.img, features)
        return image

    def jitter_face(self, image):
        jittered_images = dlib.jitter_image(image, num_jitters=5)
        for image in jittered_images:
            self.show(image, 'jitter_face')
            cv2.waitKey(0)


if __name__ == '__main__':
    f = FaceAlignment()
    f.load_img('/home/light/PycharmProjects/ImageProcess/data/frontal/0017.jpg')
    f.show(name='src')
    faces = f.detect_face()
    for face in faces:
        face_img = f.alignment_face(f.detect_features(face['position']))
        f.show(face_img, 'alignment_face')
        cv2.waitKey(0)
        f.jitter_face(face_img)
