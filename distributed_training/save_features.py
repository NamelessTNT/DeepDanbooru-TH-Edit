import tqdm

import numpy as np
import tensorflow as tf
import deepdanbooru as dd

from dataset import DatasetBuilder

source_model_path = '/home/aya/github_repos/DeepDanbooru-TH-Edit/deepdanbooru-v3-20211112-sgd-e28/model-resnet_custom_v3.h5'
project_path = '/home/aya/github_repos/DeepDanbooru-TH-Edit/th-project'
database_path = '/home/aya/backup_files/Github_repos/touhou_dataset_1/danbooru_minority_filtered.sqlite'

def serialize_example(feature, tag):

    feature = tf.io.serialize_tensor(feature)
    tag = tf.io.serialize_tensor(tag)

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "feature": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[feature.numpy()])
                ),
                "tag": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[tag.numpy()])
                )
            }
        )
    )

    return example.SerializeToString()

@tf.function
def extract_feature(images, model):
    return model.predict(images, verbose=False)

def extract_and_save_chunks(dataset, feature_model, save_prefix, max_items_per_file=2000):
    batched_ds = dataset
    file_idx = 0
    chunk_features = []
    total = 0
    
    for images in tqdm.tqdm(batched_ds, desc=f"Extracting {save_prefix}"):
        features = feature_model.predict(images, verbose=0)
        chunk_features.append(features)
        total += len(images)
        
        if total >= max_items_per_file:
            # 保存当前块
            feat_arr = np.concatenate(chunk_features, axis=0)
            np.save(f'{save_prefix}_features_{file_idx}.npy', feat_arr)
            file_idx += 1
            chunk_features = []
            total = 0
    
    # 保存最后不足一块的数据
    if chunk_features:
        feat_arr = np.concatenate(chunk_features, axis=0)
        np.save(f'{save_prefix}_features_{file_idx}.npy', feat_arr)


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    print("Loading tags ... ")
    tags = dd.project.load_tags_from_project(project_path)
    output_dim = len(tags)

    source_model = tf.keras.models.load_model(source_model_path)
    print(
        f"Model : {source_model.input_shape} -> {source_model.output_shape} (loaded from {source_model_path})"
    )
    feature_layer = source_model.get_layer("activation_171").output
    feature_model = tf.keras.Model(
        inputs=source_model.input,
        outputs=feature_layer
    )
    feature_model.trainable=False
    # cut the original model

    print(f"Loading database ... ")
    image_records = dd.data.load_image_records(database_path, 5)
    image_paths = [item[0] for item in image_records]
    tag_strings = [item[1] for item in image_records]

    dataset_wrapper = DatasetBuilder(image_paths)
    train_dataset = dataset_wrapper.build(batch_size=32)

    all_tags = []
    for tag_string in tag_strings:
        tag_array = np.array(tag_string.split(" "))

        labels = np.where(np.isin(tags, tag_array), 1, 0).astype(np.float32)
        all_tags.append(labels)
    np.save('tags.npy', all_tags)
    print("Finish processing tags.")

    feature_file = np.memmap(
        "features.dat",
        dtype="float32",
        mode="w+",
        shape=(len(image_records),4,4,4096)
    )
    
    # features = feature_model.predict(train_dataset, verbose=1)
    # index = 0
    # for images in tqdm.tqdm(train_dataset):
    #     features = feature_model(images, training=False)

    #     features = features.numpy()
    #     batch = features.shape[0]

    #     feature_file[index:index+batch] = features
    #     index += batch

    # np.save("features.npy", features)
    extract_and_save_chunks(train_dataset, feature_model, save_prefix="features")
    