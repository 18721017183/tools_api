import numpy as np
import cv2

img_path = r'C:\Users\pc\Desktop\img\1.jpg'
#要resize的图像
H = 128
W = 512

# bg = np.ones((512,128,3)) * 255
img = cv2.imread(img_path)

# img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
height,weight,_ = img.shape
print(weight,height)
if weight >= (W/H) * height:
    w = W
    h = int(height / ( weight/ w))
    img = cv2.resize(img,(w,h))
else:
    h = H
    w = int(weight / (height / h))
    img = cv2.resize(img,(w,h))
height,weight,_ = img.shape

#左上角点的坐标
x = int((W - weight) / 2)
y = int((H - height) / 2)

join_bg = np.asarray(np.ones((H,W,3))*200,dtype=np.uint8)

join_bg[y:y+height,x:x+weight] = img
image_data = np.array(join_bg, dtype='float32')
image_data /= 255.
image_data = np.transpose(image_data, [2, 0, 1])
image_data = np.expand_dims(image_data, 0)
