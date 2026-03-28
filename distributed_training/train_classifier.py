import os
import glob
import tomllib

import numpy as np
import tensorflow as tf
import deepdanbooru as dd
from sys import argv

from sklearn.model_selection import train_test_split

OUTPUT_FEATURES = 182

def concatenate_records():
    dataslice = []
    for i in range(0, 11):
        content = np.load(f"features_features_{i}.npy")
        dataslice.append(content)
    dataslice = np.concatenate(dataslice)
    np.save("features.npy", dataslice)

def build_unsquished_classifier(weight_path=None):
    """build a new classifier"""

    inputs = tf.keras.Input(shape=(4, 4, 4096))
    x = dd.model.layers.conv_gap(inputs, OUTPUT_FEATURES)
    if weight_path:
        new_weights = np.load(weight_path)
        x.set_weights(new_weights)
    outputs =  tf.keras.layers.Activation("sigmoid", dtype="float32")(x)

    return tf.keras.Model(inputs, outputs)

def build_custom_head(input_shape=(4, 4, 4096), num_classes=182):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # 1. 局部特征提取 (1x1 Conv 替代原版，但增加非线性)
    # 这一步是为了将 DeepDanbooru 的通用特征映射到你特定 IP 的语义空间
    x = tf.keras.layers.Conv2D(1024, (1, 1), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # 2. 引入空间注意力机制 (非常重要)
    # 因为 4x4 的 grid 虽然小，但每个点代表了原图很大的感受野
    # 我们通过一个简单的注意力分支，让模型决定 16 个点中哪些点对识别这 182 个角色最重要
    attention = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    x = tf.keras.layers.Multiply()([x, attention])

    # 3. 混合池化 (结合 GAP 和 GMP)
    # GAP 提取全局平滑特征，GMP (Global Max Pooling) 提取显著的局部特征（如某个角色的徽章）
    gap = tf.keras.layers.GlobalAveragePooling2D()(x)
    gmp = tf.keras.layers.GlobalMaxPooling2D()(x)
    
    merged = tf.keras.layers.Concatenate()([gap, gmp]) # 2048 维
    
    # 4. 分类器
    x = tf.keras.layers.Dropout(0.5)(merged)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # 多标签分类输出层
    outputs = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x)
    
    return tf.keras.models.Model(inputs, outputs)

def model_construct_ver_1():

    inputs = tf.keras.Input(shape=(4096,))
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(OUTPUT_FEATURES, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs)

def parse_fn(example_proto):
    """解析单条 TFRecord 记录"""
    feature_description = {
        'logits': tf.io.FixedLenFeature([4096], tf.float32),
        'tags': tf.io.FixedLenFeature([182], tf.float32),
    }

    parsed_features = tf.io.parse

def parse_unsquished_records(example_proto):
    """解析未经压缩的 (4,4,4096) feature"""
    feature_description = {
        'logits_blob': tf.io.FixedLenFeature([], tf.string),
        'labels_blob': tf.io.FixedLenFeature([], tf.string),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)

    logits = tf.io.parse_tensor(parsed['logits_blob'], out_type=tf.float32)
    labels = tf.io.parse_tensor(parsed['labels_blob'], out_type=tf.float32)
    
    logits = tf.reshape(logits, [-1, 4, 4, 4096])
    labels = tf.reshape(labels, [-1, 182])

    return logits, labels

def split_files(tfrecord_pattern, train_ratio=0.9):
    """split dataset on file level"""
    tfrecord_paths = tf.data.Dataset.list_files(tfrecord_pattern)
    
    n_total = len(tfrecord_paths)
    n_train = int(n_total * train_ratio)

    train_files = tfrecord_paths.take(n_train)
    val_files = tfrecord_paths.skip(n_train)

    return train_files, val_files

def load_unsquished_dataset(file_paths, batch_size=128, training=True):
    """construct dataset from tfrecords"""
    raw_dataset = file_paths.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        cycle_length=32,
        num_parallel_calls=32,
        deterministic=not training
    )
    
    mapped_dataset = raw_dataset.map(parse_unsquished_records,
                                     num_parallel_calls=tf.data.AUTOTUNE)
    final_dataset = (
        mapped_dataset
        .unbatch()
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return final_dataset


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

def main(config_path):
    """main train logic"""

    with open(config_path, 'rb') as f:
        train_config = tomllib.load(f)

    batch_size = train_config["batch_size"]
    initial_learning_rate = train_config["learning_rate"]
    minimum_learning_rate = train_config["min_lr"]
    lr_patience = train_config["lr_patience"]
    early_stopping_patience = train_config["early_stopping_patience"]
    tb_log_dir = train_config["tb_log_dir"]
    checkpoint_path = train_config["checkpoint_path"]

    weight_path = train_config["weight_path"]

    train_metrics = [
        tf.keras.metrics.BinaryAccuracy(name='acc'),
        tf.keras.metrics.AUC(multi_label=True, name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]

    val_metrics = [
        tf.keras.metrics.BinaryAccuracy(name='acc'),
        tf.keras.metrics.AUC(multi_label=True, name='auc')
    ]

    tfrecord_pattern = 'TFRecords_Full/shard_*.tfrecord'
    train_files, val_files = split_files(tfrecord_pattern)

    train_ds = load_unsquished_dataset(train_files, batch_size=batch_size)
    val_ds = load_unsquished_dataset(val_files,batch_size=batch_size, training=False)

    model = build_unsquished_classifier(weight_path=weight_path)
    # model = build_custom_head()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(initial_learning_rate),
        loss=tf.keras.losses.BinaryFocalCrossentropy(),
        metrics=train_metrics
    )

    callbacks = [
        # 自动保存最佳模型
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_path, "best.keras"),
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1,
            save_weights_only=True
        ),
        # 学习率衰减：当 val_loss 不再下降时自动减小学习率
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=lr_patience,
            min_lr=minimum_learning_rate,
            verbose=1
        ),
        # 早停：防止过拟合
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=early_stopping_patience,
            mode='max',
            restore_best_weights=True
        ),
        # TensorBoard 可视化
        tf.keras.callbacks.TensorBoard(log_dir=tb_log_dir)
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=150,
        callbacks=callbacks,
        verbose=1
    )


def npy_train_procedure():
    batch_size = 512

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

if __name__ == "__main__":
    default_config_path = "./train_configs/train_config.toml"
    main(default_config_path)