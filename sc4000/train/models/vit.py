from sc4000.train.models.base import Model
from sc4000.utils.label_utils import label_mapping

import numpy as np
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

from sc4000.utils.logger import setup_logger


logger = setup_logger(__name__)


class ViT(Model):
    def __init__(
        self, *, pretrained="google/vit-base-patch16-224", id2label=None, label2id=None
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
        save_strategy="epoch",
        eval_strategy: str = "epoch",
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        num_train_epochs: int = 50,
        per_device_train_batch_size: int = 16,
        per_device_eval_batch_size: int = 16,
        load_best_model_at_end: bool = True,
        logging_dir: str = "logs",
        remove_unused_columns: bool = False,
        **kwargs,
    ):
        train_ds.set_transform(self.apply_train_transforms)
        val_ds.set_transform(self.apply_val_transforms)

        # Calculate class weights for imbalanced dataset
        label_counts = train_ds.features['label'].compute_class_weight()
        num_samples = sum(label_counts.values())
        num_classes = len(label_counts)
        class_weights = {label: num_samples / (num_classes * count) for label, count in label_counts.items()}
        
        train_args = TrainingArguments(
            output_dir=output_dir,
            report_to=report_to,
            save_strategy=save_strategy,
            eval_strategy=eval_strategy,
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
            )
            labels = torch.tensor([example["label"] for example in examples])
            # Apply class weights
            weights = torch.tensor([class_weights[label.item()] for label in labels])
            return {"pixel_values": pixel_values, "labels": labels, "weights": weights}

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = predictions.argmax(axis=1)
            return {
                "accuracy": (predictions == labels).astype(np.float32).mean().item()
            }

        # Custom loss function to handle class weights
        def weighted_cross_entropy(logits, labels, weights):
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            weighted_loss = (loss * weights).mean()
            return weighted_loss

        trainer = Trainer(
            self.model,
            train_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collate_fn,
            tokenizer=self.image_processor,
            compute_metrics=compute_metrics,
            loss_fct=weighted_cross_entropy,
        )

        trainer.train()