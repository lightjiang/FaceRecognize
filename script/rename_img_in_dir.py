# coding=utf-8
import os

dir_path = '/home/light/PycharmProjects/ImageProcess/data/crowd'
temp = os.listdir(dir_path)
temp.sort()
for index, f in enumerate(temp):
    print(index, f)
    if os.path.isfile(os.path.join(dir_path, f)):
        os.rename(os.path.join(dir_path, f), os.path.join(dir_path, '%s.jpg' % str(index + 1).zfill(4)))
