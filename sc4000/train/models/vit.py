from sc4000.train.models import Model
from sc4000.train.utils.label_utils import label_mapping

import torch
from PIL import Image
from datasets import Dataset
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
)
from typing import List
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
    Resize,
)


class ViT(Model):
    def __init__(
        self, *, pretrained="google/vit-large-patch16-224", id2label=None, label2id=None
    ):
        super().__init__("ViT")
        self.image_processor = ViTImageProcessor.from_pretrained(pretrained)
        self.model = ViTForImageClassification.from_pretrained(
            pretrained,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

        self.image_mean, self.image_std = (
            self.image_processor.image_mean,
            self.image_processor.image_std,
        )
        size = self.image_processor.size["height"]
        normalize = Normalize(mean=self.image_mean, std=self.image_std)
        self.train_transforms = Compose(
            [
                RandomResizedCrop(size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )
        self.val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(size),
                ToTensor(),
                normalize,
            ]
        )

    def process_image_train(self, images: List[Image.Image]):
        return [self.train_transforms(image.convert("RGB")) for image in images]

    def process_image_val(self, images: List[Image.Image]):
        return [self.val_transforms(image.convert("RGB")) for image in images]

    def train(
        self,
        train_ds: Dataset,
        val_ds: Dataset,
        output_dir: str,
        report_to: str = "wandb",
        save_strategy="epoch",
        evaluation_strategy: str = "epoch",
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        num_train_epochs: int = 40,
        per_device_train_batch_size: int = 10,
        per_device_eval_batch_size: int = 4,
        load_best_model_at_end: bool = True,
        logging_dir: str = "logs",
        remove_unused_columns: bool = False,
        **kwargs
    ):
        train_ds.set_transform(self.process_image_train)
        val_ds.set_transform(self.process_image_val)

        train_args = TrainingArguments(
            output_dir=output_dir,
            report_to=report_to,
            save_strategy=save_strategy,
            evaluation_strategy=evaluation_strategy,
            learning_rate=lr,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            load_best_model_at_end=load_best_model_at_end,
            logging_dir=logging_dir,
            remove_unused_columns=remove_unused_columns,
            **kwargs
        )

        def collate_fn(examples):
            pixel_values = torch.stack(
                [example["pixel_values"] for example in examples]
            )
            labels = torch.tensor([example["label"] for example in examples])
            return {"pixel_values": pixel_values, "labels": labels}

        trainer = Trainer(
            self.model,
            train_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collate_fn,
            tokenizer=self.image_processor,
        )

        trainer.train()
