import os
import tqdm

import numpy as np
import tensorflow as tf
import deepdanbooru as dd

from dataset import DatasetBuilder, DatasetWithTags

source_model_path = '/home/aya/github_repos/DeepDanbooru-TH-Edit/deepdanbooru-v3-20211112-sgd-e28/model-resnet_custom_v3.h5'
project_path = '/home/aya/github_repos/DeepDanbooru-TH-Edit/th-project'
database_path = '/home/aya/backup_files/Github_repos/touhou_dataset_1/danbooru.sqlite'

def serialize_example(input_feature, tag):
    features = {
        'logits': tf.train.Feature(float_list=tf.train.FloatList(value=input_feature)),
        'tags':tf.train.Feature(float_list=tf.train.FloatList(value=tag))
    }
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()

@tf.function
def extract_feature(images, model):
    return model(images, training=False, verbose=False)

def run_inference_to_tfrecord(model, dataset, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Starting inference pipeline")

    global_idx = 0
    shard_idx = 0

    writer = None
    
    for img_batch, tag_batch in tqdm.tqdm(dataset):
        logits_batch = extract_feature(img_batch, model)

        logits_numpy = logits_batch.numpy()
        tag_numpy = tag_batch.numpy()

        batch_size = logits_numpy.shape[0]

        for i in range(batch_size):
            # 每 20,000 条数据一切片
            if global_idx % 20000 == 0:
                if writer: writer.close()
                shard_path = os.path.join(output_dir, f"shard_{shard_idx:03d}.tfrecord")
                writer = tf.io.TFRecordWriter(shard_path)
                print(f"Writing Shard {shard_idx}...")
                shard_idx += 1
            
            # 序列化
            example_str = serialize_example(logits_numpy[i], tag_numpy[i])
            writer.write(example_str)
            global_idx += 1
            
        if global_idx % 5000 == 0:
            print(f"Processed {global_idx} / 900,000 samples...")

    if writer:
        writer.close()

def extract_and_save_chunks(dataset, feature_model, save_prefix, max_items_per_file=32000):
    batched_ds = dataset
    file_idx = 0
    chunk_features = []
    total = 0
    
    for images in tqdm.tqdm(batched_ds, desc=f"Extracting {save_prefix}"):
        features = feature_model.predict(images, verbose=0)
        features = np.mean(features, axis=(1, 2))
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

    print(f"Loading database ... ")
    image_records = dd.data.load_image_records(database_path, 5)
    image_paths = [item[0] for item in image_records]
    tag_strings = [item[1] for item in image_records]
    print("Successfully extracted sqlite file")

    dataset_wrapper = DatasetWithTags((image_paths, tag_strings), tags, 512, 512)
    dataset = dataset_wrapper.get_dataset(batch_size=64)

    source_model = tf.keras.models.load_model(source_model_path)
    print(
        f"Model : {source_model.input_shape} -> {source_model.output_shape} (loaded from {source_model_path})"
    )
    feature_layer = source_model.get_layer("activation_171").output
    output_layer = tf.keras.layers.GlobalAveragePooling2D()(feature_layer)
    feature_model = tf.keras.Model(
        inputs=source_model.input,
        outputs=output_layer
        # outputs=feature_layer
    )
    feature_model.trainable=False
    # cut the original model

    
    # dataset_wrapper = DatasetBuilder(image_paths)
    # train_dataset = dataset_wrapper.build(batch_size=32)

    # all_tags = []
    # for tag_string in tag_strings:
    #     tag_array = np.array(tag_string.split(" "))

    #     labels = np.where(np.isin(tags, tag_array), 1, 0).astype(np.float32)
    #     all_tags.append(labels)
    # np.save('tags_large.npy', all_tags)
    # print("Finish processing tags.")

    
    # extract_and_save_chunks(train_dataset, feature_model, save_prefix="squished")

    # writer = tf.io.TFRecordWriter("squished_feature.tfrecord")

    # for images, tags in tqdm.tqdm(dataset):
    #     features = extract_feature(images, feature_model)
    #     features = features.numpy()
    #     features = np.mean(features, axis=(1,2))
    #     tags = tags.numpy()

    #     for f, t in zip(features, tags):
    #         example = serialize_example(f, t)
    #         writer.write(example)

    # writer.close()

    run_inference_to_tfrecord(feature_model, dataset, "./TFRecords")
    