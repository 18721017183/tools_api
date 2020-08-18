
from keras.preprocessing.image import ImageDataGenerator

path = r'C:\Users\pc\Desktop\tmp\make_data\LTSJ_LV_20200603_data_to_enhance\train_work_data\test\0' # 类别子文件夹的上一级

dst_path = r'C:\Users\pc\Desktop\tmp\make_data\LTSJ_LV_20200603_data_to_enhance\train_work_data\test\0\0'


datagen = ImageDataGenerator(

    rotation_range=5,

    width_shift_range=0.02,

    height_shift_range=0.02,

    shear_range=0.02,

    horizontal_flip=True,

    vertical_flip=True

)

gen = datagen.flow_from_directory(

                          path,

                          target_size=(224, 224),

                          batch_size=2,

                          save_to_dir=dst_path,#生成后的图像保存路径

                          save_prefix='xx',

                          save_format='jpg')

for i in range(10):

    gen.next()