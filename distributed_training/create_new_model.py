import os
import six

import tensorflow as tf
import deepdanbooru as dd

from typing import Any, Iterable, List, Tuple, Union

BACKBONE_PATH = "../deepdanbooru-v3-20211112-sgd-e28/model-resnet_custom_v3.h5"
CLASSIFIER_PATH = "dense_after_pooling.keras"

def evaluate_image(
    image_input: Union[str, six.BytesIO], model: Any, tags: List[str], threshold: float
) -> Iterable[Tuple[str, float]]:
    width = model.input_shape[2]
    height = model.input_shape[1]

    image = dd.data.load_image_for_evaluate(image_input, width=width, height=height)

    image_shape = image.shape
    image = image.reshape((1, image_shape[0], image_shape[1], image_shape[2]))
    y = model.predict(image)[0]

    result_dict = {}

    for i, tag in enumerate(tags):
        result_dict[tag] = y[i]

    for tag in tags:
        if result_dict[tag] >= threshold:
            yield tag, result_dict[tag]

def combine_backbone_and_classifier(
        backbone_path,
        classifier_path
    ):
    backbone = tf.keras.models.load_model(backbone_path)
    classifier = tf.keras.models.load_model(classifier_path)

    inputs = backbone.input
    x = backbone.get_layer("activation_171").output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = classifier(x)

    return tf.keras.Model(inputs, outputs)

def evaluate(target_path, tags_path, threshold, allow_folder=True):
    model = combine_backbone_and_classifier(
        BACKBONE_PATH,
        CLASSIFIER_PATH
    )

    target_images_paths = []
    if allow_folder and not os.path.isfile(target_path):
        items = os.listdir(target_path)
        for item in items:
            path = os.path.join(target_path, item)
            target_images_paths.append(path)
    
    with open(tags_path, 'r') as file:
        tags = file.read().split('\n')

    for image_path in target_images_paths:
        print(f"Tags of {image_path}:")
        for tag, score in evaluate_image(image_path,
                                        model,
                                        tags,
                                        threshold):
            print(f"({score:05.3f}) {tag}")
        
if __name__ == "__main__":
    evaluate("../testpics", "../th-project/tags.txt", 0.5)