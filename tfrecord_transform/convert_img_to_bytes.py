# conding=utf-8

"""
convert image dataset to raw bytes
"""
# =======================================================

import numpy as np
import skimage.io as io

# step 1: 读入图片
dog_img = io.imread("dog.jpg")
io.imshow(dog_img)
io.show()

# step 2: 转换成字符串类型
# 利用ndarray.tostring()将图片转成字符串表示，type(dog_string)是bytes
dog_string = dog_img.tostring()

# step 3: 从字符串类型恢复, 需要指定数据类型，reconstructed_dog_1d是1维的；
reconstructed_dog_1d = np.fromstring(dog_string, dtype=np.uint8)
# reshape
reconstructed_dog_img = reconstructed_dog_1d.reshape(dog_img.shape)

# step 4: 比较两者是否相同
is_same = np.allclose(dog_img, reconstructed_dog_img)
print(is_same)  # True
