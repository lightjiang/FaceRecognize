# coding = utf-8
import cv2
import os
import time
BASEPATH = os.path.abspath(os.path.dirname(__file__))


class VedioBase(object):
    base_path = BASEPATH

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.run_status = 1
        self.show_status = 'running'
        self.frame = ''
        self.last_second = time.time()

    def process_image(self):
        if self.show_status == 'running':
            ret, self.frame = self.cap.read()
            print(self.frame.shape)
            cv2.imshow('main', self.frame)
        elif self.show_status == 'pause':
            cv2.imshow('main', self.frame)

    def key_event(self):
        keyValue = cv2.waitKey(5)
        if keyValue != -1:
            print(keyValue, [keyValue])
        # esc
        if keyValue == 104:
            print('%s  hz' % self.cap.get(cv2.CAP_PROP_FPS))
        if keyValue == 27:
            self.run_status = False
        # s(save_img)
        if keyValue == 115:
            cv2.imwrite(self.base_path + '/results/vedio/vedio_now.jpg', self.frame)
            print('saved')
        # space : pause
        if keyValue == 32:
            if self.show_status == 'pause':
                self.show_status = 'running'
            else:
                self.show_status = 'pause'

    def run(self):
        while self.run_status:
            self.key_event()
            self.process_image()


if __name__ == '__main__':
    v = VedioBase()
    v.run()
