# coding = utf-8
"""
light
20180202
下载明星图片
"""

import os
from urllib import request
import re
# 明星图片路径
BASEPATH = os.path.abspath(os.path.dirname(__file__)) + '/data/star/'


class DownloadImg(object):
    def __init__(self):
        pass

    def parse_names(self):
        # 搜索以a....等字母开头明星
        for k in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            page = request.urlopen('http://www.27270.com/star/0_0_%s/' % k).read().decode('GBK')
            seq = re.findall(r'(<img[^>]*uploads/tu/mx/[^>]*/>)', page)
            # 明星列表
            for item in seq:
                name = re.findall(r'alt="([^"]*)"', item)[0]
                star_id = re.findall(r'uploads/tu/mx/(\d+)/', item)[0]
                # 下载相关图片
                self.download_picture(star_id, name)


    def download_picture(self, star_id, name):
        url_base = 'http://t2.hddhhn.com/uploads/tu/mx/'
        img_dir = os.path.join(BASEPATH, str(name))
        print(img_dir)
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        for i in range(1, 10):
            img_url = url_base + str(star_id) + '/%s.jpg' % i
            imgpath = os.path.join(img_dir, '%s.jpg' % i)
            if not os.path.exists(imgpath):
                print('downloading:     %s' % img_url)
                try:
                    imgdata = request.urlopen(img_url, timeout=2).read()
                    imgfile = open(imgpath, 'wb')
                    imgfile.write(imgdata)
                    imgfile.close()
                except:
                    break
        return True

if __name__ == '__main__':
    d = DownloadImg()
    d.parse_names()