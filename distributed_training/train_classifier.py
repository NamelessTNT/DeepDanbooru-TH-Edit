import os
import glob

import numpy as np
import tensorflow as tf
import deepdanbooru as dd

from sklearn.model_selection import train_test_split

OUTPUT_FEATURES = 182

def concatenate_records():
    dataslice = []
    for i in range(0, 11):
        content = np.load(f"features_features_{i}.npy")
        dataslice.append(content)
    dataslice = np.concatenate(dataslice)
    np.save("features.npy", dataslice)

def model_construct():
    """build a new classifier"""

    inputs = tf.keras.Input(shape=(4, 4, 4096))
    x = dd.model.layers.conv_gap(inputs, OUTPUT_FEATURES)
    outputs =  tf.keras.layers.Activation("sigmoid", dtype="float32")(x)

    return tf.keras.Model(inputs, outputs)

def model_construct_ver_1():

    inputs = tf.keras.Input(shape=(4096,))
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(OUTPUT_FEATURES, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs)

# def data_generator():
#     """yield data one by one to avoid oom"""
#     batch_size = 512

#     feature_files = sorted(glob.glob("squished_features_*.npy"))
#     tags = np.load("tags_large.npy", mmap_mode="r")
    
#     tag_index = 0
#     for f in feature_files:
#         features = np.load(f, mmap_mode='r')
#         length = features.shape[0]

#         for index in range(0, length, batch_size):
#             feature_batch = features[index : index+batch_size]
#             tag_batch = tags[tag_index : tag_index + len(feature_batch)]

#             tag_index += len(feature_batch)

#             yield feature_batch, tag_batch

def data_generator(target_indices, feature_files, samples_per_file, all_tags):
    """data generator with splitting"""

    idx_set = set(target_indices)

    for file_idx, file_path in enumerate(feature_files):
        start_idx = file_idx * samples_per_file
        end_idx = start_idx + samples_per_file

        current_file_indices = [i for i in range(start_idx, end_idx) if i in idx_set]
        if not current_file_indices:
            continue

        fearures_chunk = np.load(file_path)

        for i in current_file_indices:
            relative_idx = i - start_idx
            yield fearures_chunk[relative_idx], all_tags[i]




if __name__ == "__main__":
    batch_size = 512

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # output_signature = (
    #     tf.TensorSpec(shape=(None, 4096), dtype=tf.float32),
    #     tf.TensorSpec(shape=(None, OUTPUT_FEATURES), dtype=tf.float32)
    # )

    # dataset = tf.data.Dataset.from_generator(
    #     data_generator,
    #     output_signature=output_signature
    # )
    feature_files = sorted(glob.glob("squished_features_*.npy"))
    samples_per_file = 32000
    tags = np.load("tags_large.npy", mmap_mode='r')

    indices = np.arange(len(tags))
    train_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=42)

    ds = tf.data.Dataset.from_generator(
        lambda: data_generator(train_idx,
                               feature_files,
                               samples_per_file,
                               tags),
        output_signature=(
            tf.TensorSpec(shape=(4096,), dtype=tf.float32),
            tf.TensorSpec(shape=(OUTPUT_FEATURES,), dtype=tf.float32)
        )
    )
    dataset = (
        ds
        .shuffle(buffer_size=10000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


    # features = np.concatenate(
    #     [np.load(file, mmap_mode='r') for file in feature_files],
    #     axis=0
    # )
    
    # dataset = (
    #     tf.data.Dataset.from_tensor_slices((features, tags))
    #     .shuffle(buffer_size=1000)
    #     .cache()
    #     .batch(batch_size)
    #     .prefetch(tf.data.AUTOTUNE)
    # )
    # ds_length = 9e5
    # train_dataset = dataset.take(int(0.8 * ds_length))
    # test_dataset = dataset.skip(int(0.8 * ds_length))

    model = model_construct_ver_1()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        # loss=dd.model.losses.focal_loss(),
        metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )

    model.fit(dataset,
            verbose=1,
            # validation_data = test_dataset,
            epochs=150)
    
    model.save('all_data_dense.keras', include_optimizer=False)