# coding = utf-8
import os
from face_clustering import FaceCluster


def compare_two_face_data(a, b):
    if abs(a['score'] - b['score']) < 0.2:
        pa = a['position']
        pb = b['position']
        if abs(pa.right() - pb.right()) < 3 \
                and abs(pa.left() - pb.left()) < 3 \
                and abs(pa.top() - pb.top()) < 3 \
                and abs(pa.bottom() - pb.bottom()) < 3:
            return True

    return False


f = FaceCluster()
dir_path = '/home/light/PycharmProjects/ImageProcess/data/star'
temp = os.listdir(dir_path)
temp.sort()
for index, name in enumerate(temp):
    if os.path.isdir(os.path.join(dir_path, name)):
        paths = [
            os.path.join(dir_path, name, i) for i in os.listdir(os.path.join(dir_path, name))
        ]
        paths.sort()
        res = f.cluster(imgs=paths, log=False)
        labels = [0 for _ in range(20)]
        for face_id in res:
            face = res[face_id]
            labels[face['label']] += 1
        if len(set(labels)) > 5:
            continue
        label = labels.index(max(labels))
        a_face = {'score': 0.7}
        cluster_imgs = set()
        for face_id in res:
            face = res[face_id]
            if face['label'] == label:
                if face['score'] > a_face['score']:
                    a_face = face
                if face['score'] > 0.7:
                    cluster_imgs.add(face['src'])
        for img_path in cluster_imgs:
            f.load_img(img_path)
            faces = f.detect_face(log_status=False)
            for face in faces:
                if compare_two_face_data(a_face, face):
                    features = f.detect_features(face['position'])
                    vector = f.compute_face_vector(features)
                    #f.show()
                    #f.show(img=f.cut_face(face), name='face')
                    f.save_face(face, name=name, vector=vector)
                    # key = f.type_any_key_to_continue()
                    # print([key])
        print(name)
        print(float(index)/len(temp))
        print('\n'*4)

