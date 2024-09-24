import os
import json
import datasets
import pandas as pd

from PIL import Image
from tqdm import tqdm
from datasets import Dataset, DatasetDict, features


DATASET_PATH = ".download/cassava-leaf-disease-classification"


if __name__ == "__main__":
    test_path = os.path.join(DATASET_PATH, "test_images")
    train_path = os.path.join(DATASET_PATH, "train_images")
    train = []
    test = []
    train_csv_path = os.path.join(DATASET_PATH, "train.csv")

    with open(os.path.join(DATASET_PATH, "label_num_to_disease_map.json")) as f:
        label_map = json.load(f)

    def get_training_data():
        train_data = pd.read_csv(train_csv_path)
        for i, row in tqdm(
            train_data.iterrows(), total=len(train_data), desc="Processing train data"
        ):
            image_id = row["image_id"]
            label = row["label"]
            disease = label_map[str(label)]
            path = os.path.join(train_path, image_id)
            image = Image.open(path)
            yield {"image_id": image_id, "label": disease, "image": image}

    def get_testing_data():
        test_data_path = os.listdir(test_path)
        for id in tqdm(
            test_data_path, total=len(test_data_path), desc="Processing test data"
        ):
            path = os.path.join(test_path, id)
            image = Image.open(path)
            yield {"image_id": id, "image": image, "label": None}

    features = features.Features(
        {
            "image_id": datasets.Value("string"),
            "image": datasets.Image(),
            "label": datasets.ClassLabel(
                num_classes=len(label_map),
                names=[label_map[k] for k in sorted(label_map.keys())],
            ),
        }
    )

    hf_train_dataset = Dataset.from_generator(get_training_data, features=features)
    hf_test_dataset = Dataset.from_generator(get_testing_data, features=features)
    hf_dataset = DatasetDict({"train": hf_train_dataset, "test": hf_test_dataset})
    hf_dataset.save_to_disk(".download/hf/cassava-leaf-disease-classification")
    hf_dataset.push_to_hub("pufanyi/cassava-leaf-disease-classification")
