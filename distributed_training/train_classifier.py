import os

import numpy as np
import tensorflow as tf
import deepdanbooru as dd

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

def train_with_features(data_dir, batch_size=128):
    """train the classifier with output features"""

    features = np.load(os.path.join(data_dir, 'features.npy'))
    features = np.mean(features, axis=(1,2))
    tags = np.load(os.path.join(data_dir, 'tags.npy'))

    train_dataset = tf.data.Dataset.from_tensor_slices((features, tags))
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    model = model_construct_ver_1()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )

    history = model.fit(train_dataset,
                        verbose=1,
                        epochs=150)
    
    model.save('dense_after_pooling.keras', include_optimizer=False)

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    train_with_features("/home/aya/github_repos/DeepDanbooru-TH-Edit/distributed_training/")