# coding=utf-8
import os
import cv2
import uuid
import dlib
import time
BASEPATH = os.path.abspath(os.path.dirname(__file__))


class CamerTracker(object):
    base_path = BASEPATH

    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.cap = cv2.VideoCapture(0)
        self.frame = ''
        self.last_second = time.time()
        self.update_hz = 0
        self.track_face = {}
        self.tracker = dlib.correlation_tracker()
        self.start_track = False

    def detect_face(self, img):
        dets, scores, idx = self.face_detector.run(img, 1, -1)
        res = []
        for index, face in enumerate(dets):
            temp = {
                'id': str(uuid.uuid1()),
                'position': face,
                'score': scores[index]
            }
            if scores[index] > 0:
                res.append(temp)
        return res

    def process_image(self):
        ret, self.frame = self.cap.read()
        print(dir(self.frame))
        if self.track_face:
            height, width, _ = self.frame.shape
            self.tracker.update(self.frame)
            p = self.tracker.get_position()
            centre_x = (p.right() + p.left()) / 2
            centre_y = (p.bottom() + p.top()) / 2
            print(centre_x, centre_y, p)
            print(centre_x - width / 2, centre_y - height / 2)
        if self.start_track:
            temp = time.time()
            self.track_face = {}
            faces = self.detect_face(img=cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
            score = 1.2
            for face in faces:
                if face['score'] > score:
                    score = face['score']
                    self.track_face = face
            if self.track_face:
                self.tracker.start_track(self.frame, self.track_face['position'])
            self.start_track = False
            print(time.time() - temp)

    def run(self):
        k = 0
        sig = 0
        while 1:
            k += 1
            self.process_image()
            now = time.time()
            if now - self.last_second > 1:
                self.last_second = now
                self.update_hz = k
                k = 0
                sig += 1
                if sig > 2:
                    self.start_track = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        cv2.destroyAllWindows()

if __name__ == '__main__':
    CamerTracker().run()
