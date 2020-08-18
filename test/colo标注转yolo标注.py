# import cv2
# import os
# path = r'C:\Users\pc\Desktop\tmp\tmp_20200707\detect_cut'
# save_path = r'C:\Users\pc\Desktop\tmp\tmp_20200707\cut'
# images = os.listdir(path)
# idx = 1
# for image in images:
#     image_full_path = os.path.join(path,image)
#     img = cv2.imread(image_full_path)
#     height,width,_ = img.shape
#     h1 = int(height / 3)
#     h2 = int(height / 3 * 2)
#     img1 = img[0:h1,:]
#     img2 = img[h1:h2,:]
#     img3 = img[h2:height,:]
#     save_image = os.path.join(save_path,'{}.jpg'.format(idx))
#     cv2.imwrite(save_image,img1)
#     idx += 1
#     save_image = os.path.join(save_path, '{}.jpg'.format(idx))
#     cv2.imwrite(save_image, img2)
#     idx += 1
#     save_image = os.path.join(save_path, '{}.jpg'.format(idx))
#     cv2.imwrite(save_image, img3)
#     idx += 1

import os
import xml.etree.ElementTree as ET

dirpath = r'C:\Users\pc\Desktop\tmp\tmp\colo_xml'  # 原来存放xml文件的目录
newdir = r'C:\Users\pc\Desktop\tmp\tmp\yolo_txt'  # 修改label后形成的txt目录

if not os.path.exists(newdir):
    os.makedirs(newdir)

for fp in os.listdir(dirpath):

    root = ET.parse(os.path.join(dirpath, fp)).getroot()

    xmin, ymin, xmax, ymax = 0, 0, 0, 0
    sz = root.find('size')
    width = float(sz[0].text)
    height = float(sz[1].text)
    filename = root.find('filename').text
    for child in root.findall('object'):  # 找到图片中的所有框
        cls = child.find('name').text
        sub = child.find('bndbox')  # 找到框的标注值并进行读取
        xmin = float(sub[0].text)
        ymin = float(sub[1].text)
        xmax = float(sub[2].text)
        ymax = float(sub[3].text)
        try:  # 转换成yolov3的标签格式，需要归一化到（0-1）的范围内
            x_center = (xmin + xmax) / (2 * width)
            y_center = (ymin + ymax) / (2 * height)
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
        except ZeroDivisionError:
            print(filename, '的 width有问题')

        with open(os.path.join(newdir, fp.split('.')[0] + '.txt'), 'a+') as f:
            f.write(' '.join([str(cls), str(x_center), str(y_center), str(w), str(h) + '\n']))