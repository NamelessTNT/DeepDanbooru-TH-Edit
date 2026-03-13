import tensorflow as tf
import tensorflow_io as tfio


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