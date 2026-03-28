import tensorflow as tf
import numpy as np

base_model_path = "../deepdanbooru-v3-20211112-sgd-e28/model-resnet_custom_v3.h5"
old_tags_path = "../deepdanbooru-v3-20211112-sgd-e28/tags.txt"
new_tags_path = "../th-project/tags.txt"

with open(old_tags_path, 'r', encoding='utf-8') as file:
    old_tags = file.read().split('\n')
with open(new_tags_path, 'r', encoding='utf-8') as file:
    new_tags = file.read().split('\n')

index_map = []
for tag in new_tags:
    if tag in old_tags:
        index_map.append(old_tags.index(tag))
    else:
        index_map.append(None)

base_model = tf.keras.models.load_model(base_model_path)

old_conv_layer = base_model.get_layer('conv2d_178')
old_weights = old_conv_layer.get_weights()[0]

new_weights = np.zeros((1, 1, 4096, 182), dtype=np.float32)

initializer = tf.keras.initializers.GlorotNormal()
rand_weights = initializer(shape=(1, 1, 4096, 182)).numpy()

for i, old_idx in enumerate(index_map):
    if old_idx is not None:
        # 继承已有角色的知识
        new_weights[:, :, :, i] = old_weights[:, :, :, old_idx]
    else:
        # 新角色使用随机初始化
        new_weights[:, :, :, i] = rand_weights[:, :, :, i]

np.save("new_weights.npy", new_weights)