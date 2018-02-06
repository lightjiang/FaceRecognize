# coding=utf-8
import cv2
from utils import load_img
import os
BASEPATH = os.path.abspath(os.path.dirname(__file__))


class Base(object):
    base_path = BASEPATH

    def __init__(self):
        self.img = None

    def load_img(self, path):
        self.img = load_img(path, flags=cv2.IMREAD_COLOR)

    def add_marks(self, pos, color=None):
        if isinstance(pos, tuple):
            pos = [pos]
        elif isinstance(pos, list):
            pos = pos
        else:
            raise AttributeError

        if not color:
            color = (0, 255, 0)
        for p in pos:
            cv2.circle(self.img, p, 2, color, 1)

    def add_faces(self, faces, show_score=True, color=None):
        if isinstance(faces, dict):
            faces = [faces]
        elif isinstance(faces, list):
            faces = faces
        else:
            raise AttributeError
        for face in faces:
            rect = face['position']
            if not color:
                color = (255, 0, 0)
            cv2.rectangle(self.img, (rect.left(), rect.top()), (rect.right(), rect.bottom()), color, 3)
            if show_score and 'score' in face:
                score = face['score']
                width = rect.right() - rect.left()
                cv2.putText(self.img, str(round(score, 3)), (rect.left() + 10, rect.bottom() - 10), cv2.FONT_HERSHEY_SIMPLEX, width/120,
                            (255, 255, 255), 1)

    def show(self, img=None, name=None):
        if img is None:
            img = self.img
        if not name:
            name = 'main'
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def __exit__(self, exc_type, exc_val, exc_tb):
        cv2.destroyAllWindows()
