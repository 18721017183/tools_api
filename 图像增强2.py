# 在训练模型的时候，经常会出现数据不够多，如此就会出现过拟合等问题，通过对训练图片进行变换可以得到泛化能力更强的网络，更好的适应应用场景。博主用自己项目中常用的一些方法代码写出来。
#
# 一、数据增强方法总结
#
#  1、平移。在图像平面上对图像以一定方式进行平移。
#
# 2、翻转图像。沿着水平或者垂直方向翻转图像。
#
# 3、旋转角度。随机旋转图像一定角度;
# 改变图像内容的朝向。
#
# 4、随机颜色。对图像进行颜色抖动，对图像的每个像素RGB进行随机扰动, 常用的噪声模式是椒盐噪声和高斯噪声。
#
# 5、对比度增强。增强图像对比度，也可以用直方图均衡化。
#
# 6、亮度增强。将整个图像亮度调高。
#
# 7、颜色增强。
#
# 8、还有随机裁剪、尺度变换等代码就不赘述了。
#
# 二、数据增强方法代码
# 1、平移


from PIL import Image
from PIL import ImageEnhance
import os
import cv2
import numpy as np


def move(root_path, img_name, off):  # 平移，平移尺度为off
    img = Image.open(os.path.join(root_path, img_name))
    offset = img.offset(off, 0)
    return offset

# 2、翻转图像

def flip(root_path, img_name):  # 翻转图像
    img = Image.open(os.path.join(root_path, img_name))
    filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # filp_img.save(os.path.join(root_path,img_name.split('.')[0] + '_flip.jpg'))
    return filp_img

#  3、旋转角度

def rotation(root_path, img_name):
    img = Image.open(os.path.join(root_path, img_name))
    rot = np.random.randint(-3,3)
    rotation_img = img.rotate(rot)  # 旋转角度
    # rotation_img.save(os.path.join(root_path,img_name.split('.')[0] + '_rotation.jpg'))
    return rotation_img


# 4、随机颜色 

def randomColor(root_path, img_name):  # 随机颜色
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    image = Image.open(os.path.join(root_path, img_name))
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度


# 5、对比度增强

def contrastEnhancement(root_path, img_name):  # 对比度增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_con = ImageEnhance.Contrast(image)
    contrast = 1.5
    image_contrasted = enh_con.enhance(contrast)
    return image_contrasted


# 6、亮度增强 
def brightnessEnhancement(root_path, img_name):  # 亮度增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 1.5
    image_brightened = enh_bri.enhance(brightness)
    return image_brightened


# 7、颜色增强

def colorEnhancement(root_path, img_name):  # 颜色增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_col = ImageEnhance.Color(image)
    color = 1.5
    image_colored = enh_col.enhance(color)
    return image_colored


# 三、工程具体实现代码
# 1、将下列代码中flip函数换成你要实现的数据增强方法的名字即可。
# 2、包含的库必须要包含

from PIL import Image
from PIL import ImageEnhance
import os
import cv2
import numpy as np

imageDir = "C:/Users/Administrator/Desktop/right/"  # 要改变的图片的路径文件夹
saveDir = "C:/Users/Administrator/Desktop/save/"  # 要保存的图片的路径文件夹
i = 0
for name in os.listdir(imageDir):
    i = i + 1
    saveName = "car" + str(i) + ".jpg"
    # saveImage = flip(imageDir, name)
    saveImage = flip(imageDir, name)
    saveImage.save(os.path.join(saveDir, saveName))