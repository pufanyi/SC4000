import datasets

from typing import Dict, Tuple


def label_mapping(train_ds: datasets.Dataset) -> Tuple[Dict[int, str], Dict[str, int]]:
    id2label = dict(enumerate(train_ds.features["label"].names))
    label2id = {label: id for id, label in id2label.items()}
    return id2label, label2id
