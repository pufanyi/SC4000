from datasets import load_dataset, Dataset
from typing import Tuple
import argparse

from sc4000.train.models import load_model, Model
from sc4000.train.utils.label_utils import label_mapping


def setup_args():
    parser = argparse.ArgumentParser(description="Train a model on a dataset")
    parser.add_argument("--model", type=str, help="Name of the model to train")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of the dataset to use",
        default="pufanyi/cassava-leaf-disease-classification",
    )
    parser.add_argument(
        "--model_args", type=str, help="Arguments for the model", default="{}"
    )
    parser.add_argument(
        "--training_args",
        type=str,
        help="Arguments for training the model",
        default="{}",
    )
    return parser.parse_args()


def get_dataset(dataset_name: str) -> Tuple[Dataset, Dataset]:
    data = load_dataset(dataset_name)
    train_ds, val_ds = data["train"], data["validation"]
    return train_ds, val_ds


if __name__ == "__main__":
    args = setup_args()

    train_ds, val_ds = get_dataset(args.dataset)
    id2label, label2id = label_mapping(train_ds)

    model_args = eval(args.model_args)

    model: Model = load_model(
        args.model, id2label=id2label, label2id=label2id, **model_args
    )
    print(model.name)
