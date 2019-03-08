# coding=utf-8

"""
convert image dataset to tfrecords, and recovery;

    注：原博文中使用的测试图片来自VOC2007，读取图片的方式是PIL；
        而本次测试代码没有使用PIL，而是skimage.ios
"""
# =========================================================================
import skimage.io as io
import numpy as np
import tensorflow as tf
import os


test_data_path = "./test_images/"
output_tfrecord_filename = "./test_images/test_images.tfrecords"


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def convert_img_to_tfrecords(filename, labels):
    """将图片数据转换为tfrecord字节类型

    Parameter
        filename: 测试图片的路径名列表
        labels: 图片标签列表

    Return:
        original_images: 保存的是image_array, label_array,用于测试
    """
    writer = tf.python_io.TFRecordWriter(output_tfrecord_filename)
    original_images = []

    for img_path, label in zip(filename, labels):
        img_array = io.imread(img_path)  # 读入类型要求ndarray
        label = int(label)

        original_images.append((img_array, label))  # 实际转换的时候不需要保存，这里是为了测试，最后使用tfrecords恢复的是否一致

        height = img_array.shape[0]
        width = img_array.shape[1]

        img_raw = img_array.tostring()

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "height": _int64_feature(height),
                    "width": _int64_feature(width),
                    "image_raw": _bytes_feature(img_raw),
                    "label": _int64_feature(label)
                }
            )
        )
        writer.write(example.SerializeToString())

    writer.close()
    return original_images


def convert_tfrecords_to_img():
    """将tfrecords数据转换成图片数据

    Return:
        reconstructed_images:
    """
    reconstructed_images = []

    record_iterator = tf.python_io.tf_record_iterator(path=output_tfrecord_filename)

    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        height = int(example.features.feature['height'].int64_list.value[0])
        width = int(example.features.feature['width'].int64_list.value[0])
        img_string = (example.features.feature['image_raw'].bytes_list.value[0])
        label = int(example.features.feature['label'].int64_list.value[0])

        img_1d = np.fromstring(img_string, dtype=np.uint8)
        reconstructed_img = img_1d.reshape((height, width, -1))

        reconstructed_images.append((reconstructed_img, label))

    return reconstructed_images


def check_result_recovery(original_images, reconstructed_images):
    """检查由tfrecord类型恢复的数据与原始数据是否完全一致

    Parameter:
        original_images: 原始图像数据以及标签数据
        reconstructed_images: 由tfrecord类型恢复的图像数据与标签数据

    Return:
        result: 每张图片的检测结果
            True: 表示原始与恢复的图像数据与标签数一致
            False: 表示不一致
    """
    for original_pair, reconstructed_pair, i in zip(original_images, reconstructed_images, range(5)):
        img_pair_to_compare, label_pair_to_compare = zip(original_pair, reconstructed_pair)

        print("test{}.jpg: img_data: {}, label:{}".format(i+1, np.allclose(*img_pair_to_compare), np.allclose(*label_pair_to_compare)))

def get_dataset_filename():
    """ 获取测试数据集的图片路径，以及对应的标签

    Return:
        filename: 数据集图片路径列表
        labels: 图片标签列表
    """
    img_data_path = os.path.join(test_data_path, "data/")
    files_dir = os.listdir(img_data_path)

    filename = []
    for img_name in files_dir:
        filename.append(os.path.join(img_data_path, img_name))

    labels = []
    file_label = open(os.path.join(test_data_path, "labels.txt"))
    for line in file_label.readlines():
        line = line.strip('\n')
        labels.append(line)

    return filename, labels

if __name__ == '__main__':
    filename, labels = get_dataset_filename()

    original_images = convert_img_to_tfrecords(filename, labels)
    reconstructed_images = convert_tfrecords_to_img()

    check_result_recovery(original_images, reconstructed_images)
