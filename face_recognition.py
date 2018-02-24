# coding = utf-8
"""
light
人脸识别
"""
import dlib
import settings
import numpy as np
import json
import base64
import time
import cv2
import traceback
from face_features import FaceFeatures
from scipy.spatial import KDTree


class FaceRecognition(FaceFeatures):
    def __init__(self):
        super().__init__()
        # 加载model
        self.face_predictor = dlib.face_recognition_model_v1(settings.face_recognition_model_v1)
        # 使用kdtree做最邻近搜索
        self.flann = KDTree(np.array([[0, 0]]))
        # 存已知脸
        self.known_faces = {}
        # 存已知脸特征向量
        self.known_vectors = {}
        # 加载已知数据
        self.init_know_data()

        # 存已知脸特征向量
        self.index_vectors = []
        # 存已知脸特征id
        self.index_ids = []
        # 初始化索引
        self.init_flann_index()

    def init_know_data(self):
        with open(settings.known_faces_path, 'r') as f:
            self.known_faces = json.loads(f.read())

        with open(settings.known_vectors_path, 'r') as f:
            self.known_vectors = json.loads(f.read())

    def init_flann_index(self):
        self.index_vectors = []
        self.index_ids = []
        for _id, v in self.known_vectors.items():
            self.index_vectors.append(np.loads(base64.b64decode(v)))
            self.index_ids.append(_id)
        self.build_index(self.index_vectors)

    def compute_face_vector(self, features: object, img=None):
        """
        convert face features (68 marks) to an 128D vector
        :param features:
        :param img:
        :return:
        """
        if not img:
            img = self.img
        vector = self.face_predictor.compute_face_descriptor(img, features)
        return vector

    def build_index(self, vectors):
        if isinstance(vectors, list):
            vectors = np.array(vectors)
        del self.flann
        # self.flann = cyflann.FLANNIndex()
        # self.flann.build_index(vectors, algorithm="autotuned", target_precision=0.9)
        self.flann = KDTree(vectors)

    def search(self, vector, threshold=0.6):
        """
        所有最邻近点
        """
        b, a = self.flann.query(x=vector, k=10)
        res = []
        convince = []
        for index, cov in enumerate(b):
            score = np.sqrt(cov)
            if score < threshold:
                res.append(self.index_ids[a[index]])
                convince.append(score)
        return res, convince

    @staticmethod
    def compare_two_face(vector_a, vector_b, threshold=0.6):
        t = np.linalg.norm(vector_a - vector_b)
        return bool(t < threshold)

    def save_face(self, face: dict, name: str, vector):
        """
        if this face exists, the data will be updated
        :param face:
        :return:
        """
        self.update_known_data(face, name, vector)
        self.save_known_data()

    def update_known_data(self, face: dict, name: str, vector):
        """
        将新识别脸数据保存到本地文件
        """
        # 序列化特征向量
        vector = base64.b64encode(np.array(vector).dumps()).decode()
        # 剪裁脸部图片
        img = self.cut_face(face)
        img_path = 'results/known/%s.jpg' % face['id']
        face['name'] = name
        temp_face = {
            'id': face['id'],
            'path': img_path,
            'name': name,
            'position': str(face['position']),
            'src': face['src'],
            'score': face['score']
        }
        # 保存数据
        self.save_img(img, img_path)

        # 更新到内存
        self.known_faces[temp_face['id']] = temp_face

        self.known_vectors[face['id']] = vector

    def save_known_data(self):
        with open(settings.known_faces_path, 'w') as f:
            f.write(json.dumps(self.known_faces))

        with open(settings.known_vectors_path, 'wb') as f:
            f.write(json.dumps(self.known_vectors).encode())

    def delete_one(self, name):
        """
        删除默认数据
        """
        ids = []
        for _id, item in self.known_faces.items():
            if item['name'] == name:
                ids.append(_id)
        for _id in ids:
            del self.known_faces[_id]
            del self.known_vectors[_id]
        self.save_known_data()

    def cut_face(self, face, img=None):
        if not img:
            img = self.img
        p = face['position']
        return img[p.top():p.bottom(), p.left():p.right()]

    def recognize(self, path:str):
        self.load_img(path)
        faces = self._recognize()
        for face in faces:
            self.show(self.cut_face(face), name='face')
            results = face['recognize_results']
            if len(results.keys()) == 0:
                print('sorry, i can\'t recognize him/her.')
            else:
                name, probable_cov, _ = self.knn_filter(results)
                print('%s : %s' % (name, probable_cov))
                self.type_any_key_to_continue()
                cv2.destroyWindow('face')
                if probable_cov > 0.80:
                    choice = str(input('is it right? y/n'))
                    if choice.lower() == 'y':
                        print('updating')
                        vector = face['vector']
                        self.save_face(face, name=name, vector=vector)
                        print('updating')
                        self.init_flann_index()
                        print('updated')
            # input('type any key to continue')
            self.type_any_key_to_continue()

    def knn_filter(self, results):
        """
        对搜索到的临近点使用knn，得出评分最高的点
        影响因素：
        与数量成正相关，与距离成负相关
        """
        if not results:
            return '', 0, {}
        probable_id = ''
        probable_name = ''
        probable_cov = 1
        # names存脸名称和其评分，越大越好
        names = {}
        d_sum = len(results) / 1.3
        for _id, cov in results.items():
            # 此时cov为欧氏距离，越近越想死
            if cov < probable_cov:
                # 确定最近点
                probable_id = _id
                probable_cov = cov
            # name 为该张脸的名称
            name = self.known_faces[_id]['name']
            # 前一个5 是距离因子，值越大距离对评分贡献越大
            # 后一个5 是指欧式距离在0.5是基础评分为 1 
            # d为该张脸评分，与距离成反比
            d = 5 ** (5 - cov / 0.1)
            # d_sum为所有临近的脸评分之和
            d_sum += d

            # names[name]为该名字脸评分
            if name in names:
                names[name] += d
            else:
                names[name] = d
        # 最邻近脸所属名字 加权，避免数据库里属于该人的数据只有一条而造成的误差
        names[self.known_faces[probable_id]['name']] += 10
        d_sum += 10


        probable_cov = 0
        for name in names:
            # names[name] 为人的可能性，用该人评分除以总评分
            names[name] = names[name] / d_sum
            score = names[name]
            # 确定最高评分 人
            if score > probable_cov:
                probable_cov = score
                probable_name = name
        return probable_name, probable_cov, names

    def _recognize(self, log_status=True):
        t = time.time()
        faces = self.detect_face()
        for face in faces:
            features = self.detect_features(face['position'])
            vector = self.compute_face_vector(features)
            res, convince = self.search(vector)
            temp_result = {}
            for i, j in enumerate(res):
                temp_result[j] = convince[i]
            face['recognize_results'] = temp_result
            face['vector'] = vector
        if log_status:
            print("recognize faces takes: {}s\nNumber of faces detected: {}".format(time.time() - t, len(faces)))
        return faces


def train_faces(img):
    f = FaceRecognition()
    f.load_img(img)
    faces = f.detect_face()
    for face in faces:
        f.show(f.cut_face(face))
        features = f.detect_features(face['position'])
        vector = f.compute_face_vector(features)
        f.type_any_key_to_continue()
        name = str(input('type in name: '))
        f.save_face(face, name, vector)
        f.type_any_key_to_continue()


def start_recognize_faces():
    f = FaceRecognition()
    while 1:
        cv2.destroyAllWindows()
        path = str(input("请输入图片路径或链接:"))
        if path:
            try:
                f.recognize(path)
            except:
                traceback.print_exc()


if __name__ == '__main__':
    # start_recognize_faces()
    f = FaceRecognition()
    r = f.recognize('data/star/乔振宇/6.jpg')
