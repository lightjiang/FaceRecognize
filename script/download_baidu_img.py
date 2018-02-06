# coding=utf-8
import urllib.request
import os
import re

url = r'https://image.baidu.com/search/index?tn=baiduimage&ct=201326592&lm=-1&cl=2&ie=gbk&word=%D5%FD%C1%B3&fr=ala&ala=1&alatpl=adress&pos=0&hs=2&xthttps=111111'
BASEPATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/data/frontal/'

imgHtml = urllib.request.urlopen(url).read().decode('utf-8')
# test html
# print(imgHtml)
urls = re.findall(r'"objURL":"(.*?)"', imgHtml)

if not os.path.isdir(BASEPATH):
    os.mkdir(BASEPATH)

index = 1
for url in urls:
    print("下载:", url)

    # 未能正确获得网页 就进行异常处理
    try:
        res = urllib.request.urlopen(url)

        if str(res.status) != '200':
            print('未下载成功：', url)
            continue
    except Exception as e:
        print('未下载成功：', url)

    filename = os.path.join(BASEPATH, str(index).zfill(4) + '.jpg')
    with open(filename, 'wb') as f:
        f.write(res.read())
        print('下载完成\n')
        index += 1
print("下载结束，一共下载了 %s 张图片" % (index - 1))