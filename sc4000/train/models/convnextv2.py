from sc4000.train.models.base import Model
from sc4000.train.trainer.sc4000_trainer import SC4000Trainer

import numpy as np
import torch
from datasets import Dataset
from collections import Counter
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
)
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
    RandomRotation,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomResizedCrop,
    ColorJitter,
    Resize,
    CenterCrop,
)

from sc4000.utils.logger import setup_logger


logger = setup_logger(__name__)


class ConvNeXtV2(Model):
    def __init__(
        self,
        *,
        pretrained="facebook/convnextv2-base-22k-384",
        # More models: https://huggingface.co/models?sort=trending&search=facebook+%2F+convnextv2
        id2label=None,
        label2id=None,
    ):
        super().__init__("ConvNeXtV2")
        self.image_processor = AutoImageProcessor.from_pretrained(pretrained)
        self.model = AutoModelForImageClassification.from_pretrained(
            pretrained,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.bfloat16,
        )

        self.image_mean, self.image_std = (
            self.image_processor.image_mean,
            self.image_processor.image_std,
        )
        size = self.image_processor.size["shortest_edge"]
        normalize = Normalize(mean=self.image_mean, std=self.image_std)
        self.train_transforms = Compose(
            [
                RandomRotation(degrees=45),
                RandomResizedCrop(size),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
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

    def apply_train_transforms(self, examples):
        examples["pixel_values"] = [
            self.train_transforms(image.convert("RGB")) for image in examples["image"]
        ]
        return examples

    def apply_val_transforms(self, examples):
        examples["pixel_values"] = [
            self.val_transforms(image.convert("RGB")) for image in examples["image"]
        ]
        return examples

    def train(
        self,
        train_ds: Dataset,
        val_ds: Dataset,
        output_dir: str,
        report_to: str = "wandb",
        save_strategy="steps",
        eval_strategy: str = "steps",
        logging_steps: int = 10,
        eval_steps: int = 100,
        save_steps: int = 1000,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        num_train_epochs: int = 50,
        per_device_train_batch_size: int = 16,
        per_device_eval_batch_size: int = 16,
        load_best_model_at_end: bool = True,
        logging_dir: str = "logs",
        label_smoothing: float = 0.06,
        remove_unused_columns: bool = False,
        **kwargs,
    ):
        label_counts = Counter(train_ds["label"])
        num_samples = sum(label_counts.values())
        num_classes = len(label_counts)
        class_weights = {
            label: num_samples / (num_classes * count)
            for label, count in label_counts.items()
        }
        logger.debug(f"Class weights: {class_weights}")

        train_ds.set_transform(self.apply_train_transforms)
        val_ds.set_transform(self.apply_val_transforms)

        train_args = TrainingArguments(
            output_dir=output_dir,
            report_to=report_to,
            save_strategy=save_strategy,
            save_steps=save_steps,
            eval_strategy=eval_strategy,
            logging_steps=logging_steps,
            eval_steps=eval_steps,
            learning_rate=lr,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            load_best_model_at_end=load_best_model_at_end,
            logging_dir=logging_dir,
            remove_unused_columns=remove_unused_columns,
            metric_for_best_model="accuracy",
            **kwargs,
        )

        def collate_fn(examples):
            pixel_values = torch.stack(
                [example["pixel_values"] for example in examples]
            ).to(torch.bfloat16)
            labels = torch.tensor([example["label"] for example in examples])
            return {"pixel_values": pixel_values, "labels": labels}

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = predictions.argmax(axis=1)
            return {
                "accuracy": (predictions == labels).astype(np.float32).mean().item()
            }

        trainer = SC4000Trainer(
            self.model,
            train_args,
            weights=class_weights,
            label_smoothing=label_smoothing,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collate_fn,
            tokenizer=self.image_processor,
            compute_metrics=compute_metrics,
            lr_scheduler="reduce_lr_on_plateau",
            lr_scheduler_kwargs={"factor": 0.5, "min_lr": 5e-6},
            # lr_scheduler="cosine_with_restarts",
        )

        trainer.train()
