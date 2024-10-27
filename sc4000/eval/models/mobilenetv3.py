from typing import List
from PIL import Image

from sc4000.eval.utils.results import Result
from sc4000.eval.models.base import Model
from huggingface_hub import from_pretrained_keras
import tf_keras as keras
from pathlib import Path
import tensorflow as tf


class MobileNetV3(Model):
    def __init__(
        self,
        *,
        pretrained: str = "pufanyi/SC4000-MobileNetV3",
        image_size: int = 224,
        resize_scale: float = 1.5,
        **kwargs,
    ):
        super().__init__("MobileNetV3")
        self.model = from_pretrained_keras(pretrained)
        self.image_size = image_size
        self.resize_scale = resize_scale
        self.image_resize_shape = int(resize_scale * image_size)
        self.eval_transforms = [
            lambda img: tf.image.resize(
                img, (self.image_resize_shape, self.image_resize_shape)
            ),
            lambda img: tf.image.resize_with_crop_or_pad(
                img, target_height=image_size, target_width=image_size
            ),
            lambda img: img / 255.0,
        ]

    def process_image(self, image: Image.Image):
        image_tf = keras.utils.img_to_array(image)
        for fn in self.eval_transforms:
            image_tf = fn(image_tf)
        return image_tf

    def predict(self, images: List[Image.Image]) -> List[Result]:
        inputs = tf.stack([self.process_image(image) for image in images])
        outputs = self.model.predict(inputs)
        predictions = tf.argmax(outputs, axis=-1).numpy().tolist()
        logits_list = outputs.tolist()
        return [
            Result(prediction=pred, logs=logits)
            for pred, logits in zip(predictions, logits_list)
        ]
