# import numpy as np
# import cv2 as cv
# path = './2.jpg'
# image = cv.imread(path)
# h,w,_ = image.shape
# print(image[14][210])
# print(image[27][166])
# for i in range(h):
#     for j in range(w):
#         a,b,c = image[i][j]
#         if a < 30 and b< 30 and c > 40:
#         # if (c-a) > 20 and (c - b) > 20:
#             image[i][j] = 255
# cv.imshow('image',image)
# cv.waitKey(0)
# cv.destroyAllWindows()


import cv2
import os
import numpy as np

path = r"JPEGImages/"

template = cv2.imread(r"E:/muban.jpg", 0)

files = os.listdir(path)

for filepath in files:
    target = cv2.imread(path + filepath, 0)

    if target is None:
        continue

    theight, twidth = template.shape[:2]

    w, h = template.shape[::-1]
    # 使用matchTemplate对原始灰度图像和图像模板进行匹配
    res = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
    # 设定阈值
    threshold = 0.6
    # res大于70%
    loc = np.where(res >= threshold)

    # 使用灰度图像中的坐标对原始RGB图像进行标记
    for pt in zip(*loc[::-1]):
        cv2.rectangle(target, pt, (pt[0] + w, pt[1] + h), (7, 249, 151), 2)
    # 显示图像
    cv2.imshow('Detected', target)
    cv2.waitKey(0)

    # #执行模板匹配，采用的匹配方式cv2.TM_SQDIFF_NORMED
    # result = cv2.matchTemplate(target,template,cv2.TM_SQDIFF_NORMED)
    #
    # cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1 )
    # #寻找矩阵（一维数组当做向量，用Mat定义）中的最大值和最小值的匹配结果及其位置
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    # #匹配值转换为字符串
    # #对于cv2.TM_SQDIFF及cv2.TM_SQDIFF_NORMED方法min_val越趋近与0匹配度越好，匹配位置取min_loc
    # #对于其他方法max_val越趋近于1匹配度越好，匹配位置取max_loc
    # #绘制矩形边框，将匹配区域标注出来
    # #min_loc：矩形定点
    # #(min_loc[0]+twidth,min_loc[1]+theight)：矩形的宽高
    # #(0,0,225)：矩形的边框颜色；2：矩形边框宽度
    # cv2.rectangle(target,min_loc,(min_loc[0]+twidth,min_loc[1]+theight),(0,0,225),2)
    # #显示结果,并将匹配值显示在标题栏上
    # print(min_val,max_val)
    # cv2.imshow("MatchResult",target)
    #
    # cv2.waitKey()
cv2.imwrite(r"e:/new/13.jpg", target)
cv2.destroyAllWindows()
