#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import torch
import crnn
from PIL import Image
from config import opt
from torchvision import transforms
device = torch.device('cpu')
import cv2
import numpy as np
import onnxruntime as rt

#数据转换模块
class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        # img.sub_(0.5).div_(0.5)
        return img

#编码函数
def decode(preds, char_set):
    pred_text = ''
    for i in range(len(preds)):
        if preds[i] != 0 and ((i==0) or (i != 0 and preds[i] != preds[i-1])):
            pred_text += char_set[int(preds[i]-1)]

    return pred_text


def standardization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma

# flag = 1---谭   2---有bn   3---无bn
flag = 2    #onnx选择模型

#test_tpye = 1---类别文件夹   2---混合文件夹
test_type = 1

dir_path = r'F:\pc\work\test_python\test\OCR\Train\data\lv_generate_image\test'

def onnx_predict():
    '''所有文件夹'''
    # dirs = os.listdir(dir_path)
    # print(dirs)
    # # b. 遍历所有子文件夹
    # for dir_name in dirs:
    #     # b1. 构建子文件夹的路径
    #     c_dir_path = os.path.join(dir_path, dir_name)
    #     print("加载文件夹'{}'中的图像数据!!!".format(c_dir_path))
    #     # b2. 获取子文件夹中的图像文件的名称

    '''单个文件夹'''
    image_names = os.listdir(dir_path)
    # image_names = os.listdir(c_dir_path)
    # b3. 遍历所有图像，读取图像数据构建X和Y
    for image_name in image_names:
        # b31. 构建图像的路径
        image_path = os.path.join(dir_path, image_name)

        image = cv2.imread(image_path, 0);
        cv2.imshow('image',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # 构造一个3×3的结构元素
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilate = cv2.dilate(image, element)
        erode = cv2.erode(image, element)

        # 将两幅图像相减获得边，第一个参数是膨胀后的图像，第二个参数是腐蚀后的图像
        result = cv2.absdiff(dilate, erode);

        # 上面得到的结果是灰度图，将其二值化以便更清楚的观察结果
        retval, result = cv2.threshold(result, 40, 255, cv2.THRESH_BINARY);
        # 反色，即对二值图每个像素取反
        result = cv2.bitwise_not(result);
        # 显示图像
        cv2.imshow("result", result);
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # img = cv2.imread(image_path)
        # gray = cv2.resize(img, (256, 32), interpolation=cv2.INTER_CUBIC)
        # gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray',gray)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #
        # ret, gray = cv2.threshold(src=gray, thresh=90, maxval=255, type=cv2.THRESH_BINARY)
        #
        # # 反色，即对二值图每个像素取反
        # gray = cv2.bitwise_not(gray)
        # cv2.imshow('gray',gray)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #
        # # b. 膨胀操作
        # # kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (5, 5))
        # # # b. 膨胀操作
        # # gray = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
        #
        # kernel = np.ones((2, 2), np.uint8)
        # # b. 膨胀操作
        # gray = cv2.dilate(gray, kernel, iterations=1)
        #
        # gray = cv2.bitwise_not(gray)
        # cv2.imshow('gray',gray)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #
        # # 转换为numpy
        # x = np.array(gray).astype(np.float32)
        # x /= 255.
        #
        # if flag == 2:
        #     x = standardization(x)
        #
        # x = x.reshape((1, 1, 32, 256))

if __name__ == '__main__':
   onnx_predict()
   # pth_predict1()
   # pth_predict2()
