# coding = utf-8
"""
视频中识别人脸
"""

import cv2
import dlib
from face_recognition import FaceRecognition
from vedio_base import VedioBase
from utils import PutChineseText
import time


class VedioFaceRecognize(FaceRecognition, VedioBase):
    put_chinese_text = PutChineseText().draw_text

    def __init__(self):
        start = time.time()
        FaceRecognition.__init__(self)
        VedioBase.__init__(self)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # track 5 faces in the sametime
        self.trackers = [dlib.correlation_tracker() for _ in range(7)]
        self.track_names = []
        self.detectd_faces = []
        print('init', time.time() - start)

    def process_image(self):
        if self.show_status == 'running':
            ret, self.frame = self.cap.read()
            if self.track_names:
                positions = []
                for index, name in enumerate(self.track_names):
                    tracker = self.trackers[index]
                    tracker.update(self.frame)
                    positions.append(tracker.get_position())
                for index, box_predict in enumerate(positions):
                    print(box_predict)
                    text_position = (int(box_predict.left()) + 20, int(box_predict.bottom()) - 20)
                    try:
                        self.frame = self.put_chinese_text(self.frame, self.track_names[index], text_position,
                                                           text_size=20, text_color=(255, 0, 0))
                    except:
                        pass
                    cv2.rectangle(self.frame,
                                  (int(box_predict.left()), int(box_predict.top())),
                                  (int(box_predict.right()), int(box_predict.bottom())),
                                  (0, 0, 255), 1)
            cv2.imshow('main', self.frame)
        elif self.show_status == 'pause':
            cv2.imshow('main', self.frame)

    def key_event(self):
        keyValue = cv2.waitKey(5)

        # h: print hz
        if keyValue == 104:
            print('%s  hz' % self.cap.get(cv2.CAP_PROP_FPS))
            print('%s  hz' % self.update_hz)

        # esc： 退出
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
        # d: dump model data，将该次识别结果保存的已知数据库里
        if keyValue == 100:
            for index, face in enumerate(self.detectd_faces):
                name = face.get('name', None)
                if not name:
                    continue
                self.update_known_data(face, name, face['vector'])
            self.save_known_data()
            print('dumped')
        # i : init index
        # 重新建立索引
        if keyValue == 105:
            self.init_flann_index()
        # t: train model
        # 在暂停模式下可以训练模型，需要手动输入识别脸的名字
        if keyValue == 116 and self.show_status == 'pause':
            self.update_img()
            faces = self.detect_face()
            for face in faces:
                self.show(self.cut_face(face), name='test')
                features = self.detect_features(face['position'])
                vector = self.compute_face_vector(features)
                self.type_any_key_to_continue()
                name = str(input('type in name: '))
                if name:
                    self.update_known_data(face, name, vector)
                self.type_any_key_to_continue()
            self.save_known_data()
            cv2.destroyWindow('test')
        # r: face_recognize
        # 第一次按r ，识别人脸并追踪，按d保存这次结果， 按i重新建立索引，第二次按r取消追踪
        if keyValue == 114:
            if not self.track_names:
                self.update_img()
                self.detectd_faces = self._recognize()
                for index, face in enumerate(self.detectd_faces):
                    results = face['recognize_results']
                    name, probable_cov, _ = self.knn_filter(results)
                    if not name:
                        continue
                    p = face['position']
                    face['name'] = name
                    self.trackers[index].start_track(self.frame, p)
                    self.track_names.append(name + ' : %s' % round(probable_cov, 3))
            else:
                self.track_names = []
                self.detectd_faces = []

    def update_img(self):
        self.img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
    v = VedioFaceRecognize()
    v.run()



