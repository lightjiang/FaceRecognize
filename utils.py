# coding=utf-8
import cv2
import os


def load_img(path, flags=None):
    base_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(base_path, path)
    if os.path.exists(path):
        res = cv2.imread(path, flags=flags)
        return cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    else:
        raise "path not exist: %s" % path
