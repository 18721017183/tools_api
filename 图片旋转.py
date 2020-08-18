# -*- coding: UTF-8 -*-

import os
import numpy as np
import cv2
from PIL import Image

'''方法一：对于一般图片可以自动旋转矫正。对于黑色背景效果不太好'''
# ## 图片旋转
# def rotate_bound(image, angle):
#     # 获取宽高
#     (h, w) = image.shape[:2]
#     (cX, cY) = (w // 2, h // 2)
#
#     # 提取旋转矩阵 sin cos
#     M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
#     cos = np.abs(M[0, 0])
#     sin = np.abs(M[0, 1])
#
#     # 计算图像的新边界尺寸
#     nW = int((h * sin) + (w * cos))
#     #     nH = int((h * cos) + (w * sin))
#     nH = h
#
#     # 调整旋转矩阵
#     M[0, 2] += (nW / 2) - cX
#     M[1, 2] += (nH / 2) - cY
#
#     return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#
#
# ## 获取图片旋转角度
# def get_minAreaRect(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.bitwise_not(gray)
#     thresh = cv2.threshold(gray, 0, 255,
#                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#     coords = np.column_stack(np.where(thresh > 0))
#     return cv2.minAreaRect(coords)
#
#
# image_path = r"C:\Users\pc\Desktop\tmp\12_1.jpg"
# image = cv2.imread(image_path)
# angle = get_minAreaRect(image)[-1]
# rotated = rotate_bound(image, angle)
#
# cv2.putText(rotated, "angle: {:.2f} ".format(angle),
#             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#
# # show the output image
# print("[INFO] angle: {:.3f}".format(angle))
# cv2.namedWindow('imput',0)
# cv2.namedWindow('output',0)
# # cv2.resizeWindow('imput',200,300)
# cv2.imshow("imput", image)
# cv2.imshow("output", rotated)
# cv2.waitKey(0)


#解决cv2.imread没法读取中文路径图片的问题
def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    return cv_img

'''方法二：对于黑色背景的图片，将其转换为白色背景，计算旋转角，再进行矫正'''
## 图片旋转
def rotate_bound(image, angle):
    # 获取宽高
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # 提取旋转矩阵 sin cos
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 计算图像的新边界尺寸
    nW = int((h * sin) + (w * cos))
    #     nH = int((h * cos) + (w * sin))
    nH = h

    # 调整旋转矩阵
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


## 获取图片旋转角度
def get_minAreaRect(image):
    #将黑底替换成白底，方便计算旋转角度
    img = image.copy()
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = img

    image = np.asarray(hsv)  # (422, 903, 3)

    lower_blue = np.array([0, 0, 0])
    upper_blue = np.array([50, 50, 50])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # cv2.namedWindow('Mask', 0)
    # cv2.resizeWindow('Mask', 300, 500)
    # cv2.imshow('Mask', mask)



    rows, cols, channels = img.shape
    for i in range(rows):
        for j in range(cols):
            if mask[i, j] == 255:
                img[i, j] = (255, 255, 255)  # 此处替换颜色，为BGR通道


    #计算旋转角度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    return cv2.minAreaRect(coords)

# '''批量教程文件夹图片，并保存如另一个文件夹'''
# input_dir = r'F:\datas\原始数据\20200630_拍摄（OCR-辣条缺陷）\缺陷检测'
# save_dir = r'F:\datas\tmp_image'
# image_names = os.listdir(input_dir)
# for image_name in image_names:
#     print(image_name)
#     image_full_name = os.path.join(input_dir,image_name)
#     # image  =cv2.imread(image_full_name)
#     image = cv_imread(image_full_name)
#     angle = get_minAreaRect(image)[-1]
#     rotated = rotate_bound(image, angle + 90)
#     rotated = np.asarray(rotated)
#     cv2.imwrite(os.path.join(save_dir,image_name),rotated)
#     print("[INFO] angle: {:.3f}".format(angle+90))
#     cv2.namedWindow('input',0)
#     cv2.resizeWindow('input',300,500)
#     cv2.namedWindow('output',0)
#     cv2.resizeWindow('output',300,500)
#     cv2.imshow("input", image)
#     cv2.imshow("output", rotated)
#     cv2.waitKey(0)



image_path = r'F:\datas\原始数据\20200630_拍摄（OCR-辣条缺陷）\缺陷检测\2020_6_30_11_12_36_10.bmp'
image = cv_imread(image_path)
angle = get_minAreaRect(image)[-1]
print(angle+90)
rotated = rotate_bound(image,angle + 90)

#打印旋转的角度
cv2.putText(rotated, "angle: {:.2f} ".format(angle),
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# show the output image
# print("[INFO] angle: {:.3f}".format(angle))
cv2.namedWindow('input',0)
cv2.resizeWindow('input',300,500)
cv2.namedWindow('output',0)
cv2.resizeWindow('output',300,500)
cv2.imshow("input", image)
cv2.imshow("output", rotated)
cv2.waitKey(0)