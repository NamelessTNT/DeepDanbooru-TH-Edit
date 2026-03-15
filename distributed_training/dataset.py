import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

import skimage.transform


class DatasetBuilder:
    def __init__(self, data_list, img_size=(512, 512)):
        self.file_paths = data_list
        self.img_size = img_size

    def _load_and_preprocess(self, path):
        """
        核心处理函数：结合 TF 算子和 PyFunction
        """
        # 1. 加载图片 (使用 TF 原生算子，速度快且支持并行)
        image_raw = tf.io.read_file(path)
        try:
            image = tf.io.decode_png(image_raw, channels=3)
        except:
            image = tfio.image.decode_webp(image_raw)
            image = tfio.experimental.color.rgba_to_rgb(image)

        img = tf.image.resize(image, self.img_size)
        img = tf.cast(img, tf.float32) / 255.0  # 归一化到 [0, 1]
        
        # 3. 关键：手动设置形状
        img.set_shape([self.img_size[0], self.img_size[1], 3])

        return img

    def build(self, batch_size=32):
        # 从列表创建基础数据集
        ds = tf.data.Dataset.from_tensor_slices(self.file_paths)

        # 使用 num_parallel_calls 实现并行化
        ds = ds.map(
            self._load_and_preprocess, 
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True
        )

        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds


class DatasetWithTags:
    def __init__(self, data_list, tags, width=512, height=512):
        self.data_list = data_list
        self.tag_all_array = np.array(tags)
        self.width = width
        self.height = height

    def get_dataset(self, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices(self.data_list)
        dataset = dataset.map(
            self.map_load_image, num_parallel_calls=tf.data.AUTOTUNE
        )   # get image data from path
        dataset = dataset.ignore_errors()
        dataset = dataset.map(
            self.map_transformation_image_and_label,
            num_parallel_calls=tf.data.AUTOTUNE
        )   # scale image and get multi-hot tag array
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset
    
    def map_load_image(self, image_path, tag_string):
        image_raw = tf.io.read_file(image_path)
        try:
            image = tf.io.decode_png(image_raw, channels=3)
        except:
            image = tfio.image.decode_webp(image_raw)
            image = tfio.experimental.color.rgba_to_rgb(image)

        return (image, tag_string)
    
    def map_transformation_image_and_label(self, image, tag_string):
        return tf.py_function(
            self.map_transformation_image_and_label_py,
            (image, tag_string),
            (tf.float32, tf.float32)
        )
    
    def map_transformation_image_and_label_py(self, image, tag_string):
        image = image.numpy()
        t = skimage.transform.AffineTransform(
            translation=(-image.shape[1] * 0.5, -image.shape[0] * 0.5)
        )   # centerize
        image = skimage.transform.warp(
            image, (t).inverse, output_shape=(self.height, self.width), order=1, mode="edge"
        )
        image = image / 255.0

        tag_string = tag_string.numpy().decode()
        tag_array = np.array(tag_string.split(" "))

        labels = np.where(np.isin(self.tag_all_array, tag_array), 1, 0).astype(np.float32)

        return (image, labels)