from math import e
import numpy as np
import matplotlib.pyplot as plt
from tf_keras.src.backend import one_hot
from tqdm import tqdm

import pandas as pd
import tensorflow as tf
import tf_keras as keras
import tensorflow_hub as hub

from pathlib import Path
from datasets import Dataset

from sc4000.train.models import Model
from sc4000.utils.logger import setup_logger

logger = setup_logger(__name__)


class MobileNetV3(Model):
    def __init__(
        self,
        *,
        pretrained: str = "https://tfhub.dev/google/cropnet/classifier/cassava_disease_V1/2",
        image_size=224,
        num_classes: int = 5,
        resize_scale: float = 1.5,
        **kwargs,
    ):
        super().__init__("MobileNetV3")
        self.image_size = image_size
        self.pretrained_layer = hub.KerasLayer(pretrained, trainable=True)
        self.model = keras.Sequential(
            [
                keras.Input(shape=(self.image_size, self.image_size, 3)),
                self.pretrained_layer,
            ]
        )
        self.resize_scale = resize_scale
        self.image_resize_shape = int(self.image_size * self.resize_scale)
        self.num_classes = num_classes + 1
        self.train_transforms = [
            tf.image.random_flip_left_right,
            tf.image.random_flip_up_down,
            lambda img: tf.image.random_brightness(img, 0.2),
            lambda img: tf.image.random_saturation(img, 5, 10),
            lambda img: tf.clip_by_value(img, 0.0, 255.0),
            lambda img: tf.image.resize(
                img, [self.image_resize_shape, self.image_resize_shape]
            ),
            lambda img: tf.image.random_crop(
                img, size=[self.image_size, self.image_size, 3]
            ),
            lambda img: img / 255.0,
        ]
        self.val_transforms = [
            # lambda img: tf.cast(img, tf.float32),
            lambda img: tf.image.resize(
                img, [self.image_resize_shape, self.image_resize_shape]
            ),
            lambda img: tf.image.resize_with_crop_or_pad(
                img, target_height=self.image_size, target_width=self.image_size
            ),
            lambda img: img / 255.0,
        ]

    def collate_fn(self, batch):
        images = [item["image"] for item in batch]
        labels = [item["one_hot_label"] for item in batch]
        logger.info(labels)
        logger.info(tf.stack(labels))
        return {"image": tf.stack(images), "one_hot_label": tf.stack(labels)}

    def get_target(self, label):
        return tf.one_hot(label, self.num_classes)

    def train_image_transforms(self, image):
        image = keras.utils.img_to_array(image)
        for fn in self.train_transforms:
            image = fn(image)
        return image

    def val_image_transforms(self, image):
        image = keras.utils.img_to_array(image)
        for fn in self.val_transforms:
            image = fn(image)
        return image

    def train_image_transforms_batch(self, images):
        for fn in self.train_transforms:
            images = tf.map_fn(fn, images)
        # print(images)
        return images

    def val_image_transforms_batch(self, images):
        for fn in self.val_transforms:
            # image = fn(image)
            images = tf.map_fn(fn, images)
        return images

    def train_map(self, item):
        return (
            self.train_image_transforms(item["image"]),
            self.get_target(item["label"])
            # item["label"],
        )

    def val_map(self, item):
        return (
            self.val_image_transforms(item["image"]),
            # item["label"],
            self.get_target(item["label"]),
        )

    def train(
        self,
        train_ds: Dataset,
        val_ds: Dataset,
        output_dir: str,
        lr: float = 1e-5,
        early_stopping_patience: int = 5,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        lr_reduce_patience: int = 3,
        image_size: int = 224,
        lr_reduce_min_delta: float = 1e-3,
        **kwargs,
    ):
        train_ds = train_ds
        val_ds = val_ds

        train_steps = len(train_ds) // train_batch_size

        self.kwargs = kwargs
        self.image_size = image_size

        train_list = [
            self.train_map(item)
            for item in tqdm(train_ds, desc="Converting train data")
        ]
        val_list = [
            self.val_map(item) for item in tqdm(val_ds, desc="Converting val data")
        ]

        tf_train_ds = (
            tf.data.experimental.from_list(train_list)
            .batch(train_batch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        tf_val_ds = (
            tf.data.experimental.from_list(val_list)
            .batch(eval_batch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )

        for images, labels in tf_train_ds.take(1):
            logger.info(f"Sample train batch images: {images}")
            logger.info(f"Sample train batch labels: {labels}")

        for images, labels in tf_val_ds.take(1):
            logger.info(f"Sample val batch images shape: {images.shape}")
            logger.info(f"Sample val batch labels shape: {labels.shape}")

        self.model.fit(
            tf_train_ds,
            validation_data=tf_val_ds,
            epochs=500,
            steps_per_epoch=train_steps,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=early_stopping_patience,
                    mode="min",
                    verbose=1,
                    restore_best_weights=True,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.3,
                    patience=lr_reduce_patience,
                    min_delta=lr_reduce_min_delta,
                    mode="min",
                    verbose=1,
                ),
            ],
        )

        model_dir = Path(output_dir) / "model"
        if not model_dir.exists():
            model_dir.mkdir(parents=True)

        self.model.save(model_dir)

        logger.info(f"Model saved to {model_dir}")
