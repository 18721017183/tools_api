#!usr/bin/python
# -*- coding: utf-8 -*-
import cv2
from imgaug import augmenters as iaa
import os
import numpy as np

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# 定义一组变换方法.
seq = iaa.Sequential([

    # 选择0到5种方法做变换
    iaa.SomeOf((0, 5),
               [
                   # iaa.Fliplr(0.5),  # 对50%的图片进行水平镜像翻转
                   # iaa.Flipud(0.5),  # 对50%的图片进行垂直镜像翻转

                   # Convert some images into their superpixel representation,  将一些图像转换成超像素表示,
                   # sample between 20 and 200 superpixels per image, but do
                   # not replace all superpixels with their average, only
                   # some of them (p_replace).
                   # sometimes(
                   #     iaa.Superpixels(
                   #         p_replace=(0, 1.0),
                   #         n_segments=(20, 200)
                   #     )
                   # ),

                   # Blur each image with varying strength using    使用不同的强度模糊每个图像
                   # gaussian blur (sigma between 0 and 3.0),
                   # average/uniform blur (kernel size between 2x2 and 7x7)
                   # median blur (kernel size between 3x3 and 11x11).
                   # iaa.OneOf([
                   #     iaa.GaussianBlur((0, 3.0)),
                   #     iaa.AverageBlur(k=(2, 7)),
                   #     iaa.MedianBlur(k=(3, 11)),
                   # ]),
                   iaa.OneOf([
                       iaa.GaussianBlur((0, 3.0)),
                       iaa.AverageBlur(k=(2, 5)),
                       iaa.MedianBlur(k=(3, 7)),
                   ]),

                   # Sharpen each image, overlay the result with the original   锐化每个图像，将结果与原始图像叠加
                   # image using an alpha between 0 (no sharpening) and 1
                   # (full sharpening effect).
                   iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                   # Same as sharpen, but for an embossing effect.  和锐化一样，不过是为了浮雕效果。
                   # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                   # Add gaussian noise to some images.
                   # In 50% of these cases, the noise is randomly sampled per
                   # channel and pixel.
                   # In the other 50% of all cases it is sampled once per
                   # pixel (i.e. brightness change).
                   # """
                   # 高斯噪声
                   # """
                   iaa.AdditiveGaussianNoise(
                       loc=0, scale=(0.0, 0.05 * 255)
                   ),

                   # Invert each image's chanell with 5% probability.   以5% 的概率反转每个图像的频率。
                   # This sets each pixel value v to 255-v.
                   # iaa.Invert(0.05, per_channel=True),  # invert color channels

                   # Add a value of -10 to 10 to each pixel.   每个像素增加一个 -10到10的值。
                   iaa.Add((-10, 10), per_channel=0.5),

                   # Add random values between -40 and 40 to images, with each value being sampled per pixel:
                   # 在图像中添加-40到40之间的随机值，每个值按像素进行取样:
                   # iaa.AddElementwise((-40, 40)),

                   # Change brightness of images (50-150% of original value). 改变图像的亮度(原始值的50-150%)。
                   iaa.Multiply((0.5, 1.5)),

                   # Multiply each pixel with a random value between 0.5 and 1.5.  用0.5到1.5之间的随机值乘以每个像素。
                   # iaa.MultiplyElementwise((0.5, 1.5)),

                   # Improve or worsen the contrast of images.  改善或恶化图像的对比度。
                   iaa.ContrastNormalization((0.5, 2.0)),

               ],
               # do all of the above augmentations in random order
               random_order=True
               )

], random_order=True)  # apply augmenters in random order

# 图片文件相关路径
path = r'C:\Users\pc\Desktop\tmp\make_ocr_image\number_image\Y'   #原图片文件夹
saved_dir = r'C:\Users\pc\Desktop\tmp\make_ocr_image\number_image_expand'   #图片保存位置
dir_name = os.path.split(path)[-1]
dir_name = os.path.join(saved_dir,dir_name)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
savedpath = dir_name
imglist = []
filelist = os.listdir(path)
print(filelist)
# 遍历要增强的文件夹，把所有的图片保存在imglist中
for item in filelist:
    img_dir = os.path.join(path,item)
    img = cv2.imread(img_dir)
    # print('item is ',item)
    # print('img is ',img)
    # images = load_batch(batch_idx)
    imglist.append(img)
# print('imglist is ' ,imglist)
print('all the picture have been appent to imglist')

# 对文件夹中的图片进行增强操作，循环100次
for count in range(6):
    images_aug = seq.augment_images(imglist)
    for index in range(len(images_aug)):
        filename = str(count) + str(index) + '.jpg'
        # 保存图片
        img_name = os.path.join(savedpath,filename)
        cv2.imwrite(img_name, images_aug[index])
        print('image of count%s index%s has been writen' % (count, index))