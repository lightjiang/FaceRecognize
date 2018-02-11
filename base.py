# coding=utf-8
import cv2
import os
from urllib import request
import numpy as np
import uuid
BASEPATH = os.path.abspath(os.path.dirname(__file__))


class Base(object):
    base_path = BASEPATH

    def __init__(self):
        self.img = None
        self.img_path = ''

    def load_img(self, path, relative=True, flags=cv2.IMREAD_COLOR):
        if path.startswith('http'):
            path = self.download_web_img(path)

        img_path = path
        if relative:
            path = os.path.join(self.base_path, path)
        if os.path.exists(path):
            res = cv2.imread(path, flags=flags)
            self.img = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            self.img_path = img_path
            return self.img
        else:
            raise FileNotFoundError(path)

    def download_web_img(self, url):
        path = 'data/auto_download_img/%s.jpg' % uuid.uuid1()
        request.urlretrieve(url, path)
        print('download complete')
        return path

    def save_img(self, img, path, relative=True):
        if relative:
            path = os.path.join(self.base_path, path)
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

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

    def add_faces(self, faces, show_score=True, color=None, add_text=None):
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
            if add_text:
                width = rect.right() - rect.left()
                cv2.putText(self.img, str(add_text), (rect.left() + 10, rect.bottom() - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, width / 120,
                            (255, 255, 255), 1)

    def show(self, img=None, name=None):
        if img is None:
            img = self.img
        if not name:
            name = 'main'
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    @staticmethod
    def type_any_key_to_continue():
        print('type_any_key_to_continue')
        return cv2.waitKey(0)

    def __exit__(self, exc_type, exc_val, exc_tb):
        cv2.destroyAllWindows()
