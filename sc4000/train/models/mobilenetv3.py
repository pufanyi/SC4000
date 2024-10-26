from math import e
import numpy as np
import matplotlib.pyplot as plt
from tf_keras.src.backend import one_hot
from torch import optim
import wandb

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
        batch_norm_momentum: float = 0.997,
        num_classes: int = 5,
        **kwargs,
    ):
        super().__init__("MobileNetV3")
        self.pretrained_layer = hub.KerasLayer(pretrained, trainable=True)
        self.model = keras.Sequential(
            [
                keras.Input(shape=(224, 224, 3)),
                self.pretrained_layer,
            ]
        )
        self.image_size = image_size
        self.num_classes = num_classes
        self.train_transforms = [
            tf.image.random_flip_left_right,
            tf.image.random_flip_up_down,
            lambda img: tf.image.random_brightness(img, 0.2),
            lambda img: tf.image.random_crop(
                img, size=[self.image_size, self.image_size, 3]
            ),
        ]
        self.val_transforms = [
            lambda img: tf.image.resize_with_crop_or_pad(
                img, target_height=self.image_size, target_width=self.image_size
            ),
        ]

    def collate_fn(self, batch):
        images = [item["image"] for item in batch]
        labels = [item["one_hot_label"] for item in batch]
        logger.info(labels)
        logger.info(tf.stack(labels))
        return {"image": tf.stack(images), "one_hot_label": tf.stack(labels)}

    def get_target(self, label):
        one_hot_label = [0.0] * (self.num_classes + 1)
        one_hot_label[label] = 1.0
        return one_hot_label

    def train_image_transforms(self, images):
        for fn in self.train_transforms:
            images = tf.map_fn(fn, images)
        return images

    def val_image_transforms(self, images):
        for fn in self.val_transforms:
            images = tf.map_fn(fn, images)
        return images

    def train_map(self, item):
        logger.info(item["one_hot_label"])
        return (
            self.train_image_transforms(item["image"]),
            # self.get_target(item["one_hot_label"])
            item["one_hot_label"],
        )

    def val_map(self, item):
        return (
            self.val_image_transforms(item["image"]),
            item["one_hot_label"],
            # self.get_target(item["one_hot_label"]),
        )

    def train(
        self,
        train_ds: Dataset,
        val_ds: Dataset,
        output_dir: str,
        optimizer: str = "rmsprop",
        lr: float = 1e-4,
        early_stopping_patience: int = 5,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        lr_reduce_patience: int = 3,
        image_size: int = 224,
        lr_reduce_min_delta: float = 1e-4,
        **kwargs,
    ):
        train_steps = len(train_ds) // train_batch_size

        self.kwargs = kwargs
        self.image_size = image_size

        # train_one_hot_labels = [tf.one_hot(label, self.num_classes) for label in train_ds["label"]]
        # val_one_hot_labels = [tf.one_hot(label, self.num_classes) for label in val_ds["label"]]

        train_one_hot_labels = [self.get_target(label) for label in train_ds["label"]]
        val_one_hot_labels = [self.get_target(label) for label in val_ds["label"]]

        train_ds = train_ds.add_column("one_hot_label", train_one_hot_labels)
        val_ds = val_ds.add_column("one_hot_label", val_one_hot_labels)

        tf_train_ds = train_ds.to_tf_dataset(
            columns=["image", "one_hot_label"],
            batch_size=train_batch_size,
            collate_fn=self.collate_fn,
        )
        tf_val_ds = val_ds.to_tf_dataset(
            columns=["image", "one_hot_label"],
            batch_size=eval_batch_size,
            collate_fn=self.collate_fn,
        )

        self.model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )

        tf_train_ds = tf_train_ds.map(
            self.train_map, num_parallel_calls=tf.data.AUTOTUNE
        )
        tf_val_ds = tf_val_ds.map(self.val_map, num_parallel_calls=tf.data.AUTOTUNE)

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

        self.model.save(str(dir))

        logger.info(f"Model saved to {model_dir}")
