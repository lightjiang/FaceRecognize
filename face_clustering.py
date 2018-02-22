# coding=utf-8
import dlib
import os
import time
from face_recognition import FaceRecognition


class FaceCluster(FaceRecognition):

    def cluster(self, imgs: list, log=True) -> dict:
        t = time.time()
        response = {}
        vectors = []
        vectors_id = []
        for img in imgs:
            img_path = os.path.join(self.base_path, img)
            if not os.path.isfile(img_path):
                raise FileNotFoundError(img_path)
            self.load_img(img_path)
            faces = self.detect_face(log_status=False)
            for face in faces:
                # get face features
                features = self.detect_features(face['position'])

                # convert face features to a vector
                face_vector = self.compute_face_vector(features)
                vectors.append(face_vector)
                vectors_id.append(face['id'])
                response[face['id']] = face
        labels = self.__cluster(vectors)
        for index, label in enumerate(labels):
            response[vectors_id[index]]['label'] = label
        if log:
            print("Clustering faces takes: {}s\nNumber of categories: {}".format(time.time() - t, len(set(labels))))
        return response

    @staticmethod
    def __cluster(faces_vectors: list, threshold=0.5):
        return dlib.chinese_whispers_clustering(faces_vectors, threshold)


if __name__ == '__main__':
    from face_alignment import FaceAlignment

    # use FaceAlignment to cut face from imgs and show faces
    class TempClass(FaceCluster, FaceAlignment):
        pass

    f = TempClass()
    res = f.cluster([
        'data/star/乔振宇/1.jpg',
        'data/star/乔振宇/6.jpg',
        'data/star/乔振宇/4.jpg',
        'data/star/俞灏明/1.jpg',
        'data/star/俞灏明/2.jpg',
        'data/star/俞灏明/3.jpg',
        'data/star/俞灏明/4.jpg',
        # BY2 双胞胎傻傻分不清
        'data/star/BY2/1.jpg',
        'data/star/BY2/2.jpg',
        'data/star/BY2/3.jpg',
        'data/star/BY2/4.jpg',

    ])
    for face_id in res:
        face = res[face_id]
        print(face)
        print(face['label'], face['src'], face['position'])
        f.load_img(os.path.join(f.base_path, face['src']))
        features = f.detect_features(face['position'])
        img = f.alignment_face(features)
        f.show(img, name=str(face['id']))
        f.type_any_key_to_continue()
