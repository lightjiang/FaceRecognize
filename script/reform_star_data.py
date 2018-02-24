# coding = utf-8
"""
light
20180202
对获取到的明星图片聚类，认为包含最多张脸数的类别为该目录名的明星
并把结果保存，以便识别引擎识别
"""
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
    # name 为明星名字
    if os.path.isdir(os.path.join(dir_path, name)):
        # paths 为该明星路径下所有图片路径
        paths = [
            os.path.join(dir_path, name, i) for i in os.listdir(os.path.join(dir_path, name))
        ]
        paths.sort()
        # 对该明星相关图片进行聚类
        res = f.cluster(imgs=paths, log=False)
        labels = [0 for _ in range(20)]
        for face_id in res:
            face = res[face_id]
            labels[face['label']] += 1
        if len(set(labels)) > 5:
            continue
        # label 为分类里识别数目最多的脸分类名
        label = labels.index(max(labels))
        # 取人脸评分0.7以上图片
        a_face = {'score': 0.7}
        cluster_imgs = set()
        for face_id in res:
            face = res[face_id]
            if face['label'] == label:
                # 寻找评分最高的脸图片
                if face['score'] > a_face['score']:
                    a_face = face
                # 把评分高于0.7的图片添加到聚类结果里
                if face['score'] > 0.7:
                    cluster_imgs.add(face['src'])
        for img_path in cluster_imgs:
            f.load_img(img_path)
            faces = f.detect_face(log_status=False)
            for face in faces:
                # 讲识别的face 与已知最清晰的脸对比，通过添加到引擎识别库里
                if compare_two_face_data(a_face, face):
                    features = f.detect_features(face['position'])
                    vector = f.compute_face_vector(features)
                    #f.show()
                    #f.show(img=f.cut_face(face), name='face')

                    # 保存face图片，特征，人名
                    f.save_face(face, name=name, vector=vector)
                    # key = f.type_any_key_to_continue()
                    # print([key])
        print(name)
        # 打印进度
        # 总共需要10来mins
        print(float(index)/len(temp))
        print('\n'*4)

