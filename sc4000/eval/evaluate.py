import json
import os

from datetime import datetime
from argparse import ArgumentParser
from datasets import load_dataset, Dataset
from tqdm import tqdm

from sc4000.eval.models import load_model, Model
from sc4000.utils.label_utils import label_mapping
from sc4000.utils.logger import enable_debug_mode, setup_logger
from sc4000.utils.format import format_args

logger = setup_logger(__name__)


def setup_args():
    parser = ArgumentParser(description="Train a model on a dataset")
    parser.add_argument(
        "--model", type=str, help="Name of the model to train", required=True
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of the dataset to use",
        default="pufanyi/cassava-leaf-disease-classification",
    )
    parser.add_argument(
        "--subset",
        type=str,
        help="Subset of the dataset to use",
        default="default",
    )
    parser.add_argument(
        "--split",
        type=str,
        help="Split of the dataset to use",
        default="validation",
    )
    parser.add_argument(
        "--model_args", type=str, help="Arguments for the model", default="{}"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Directory to save the model to",
        default="output/eval",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


def evaluate(model: Model, val_ds: Dataset, label2id, id2label, model_args):
    model = load_model(args.model, label2id=label2id, id2label=id2label, **model_args)
    result = []
    correct_num = 0
    for example in tqdm(val_ds, desc=f"Evaluating {model.name}"):
        id = example["image_id"]
        image = example["image"]
        prediction = model.predict(image)
        res = prediction.prediction
        logs = prediction.logs
        correct = example["label"] == res
        result.append(
            {
                "image_id": id,
                "answer": example["label"],
                "prediction": res,
                "logs": logs,
                "correct": correct,
            }
        )
        logger.debug(
            f"Image ID: {id}, Correct: {correct}, Prediction: {res}, Answer: {example['label']}, Logs: {logs}"
        )
        correct_num += correct
    acc = correct_num / len(val_ds)

    logger.info(f"Accuracy for {model.name}: {acc}")

    return {
        "model": model.name,
        "accuracy": acc,
        "model_args": model_args,
        "result": result,
    }


if __name__ == "__main__":
    args = setup_args()

    if args.debug:
        enable_debug_mode()
        logger = setup_logger(__name__)

    logger.debug(f"Model Arguments: {args.model_args}")
    model_args = format_args(args.model_args)
    val_ds = load_dataset(args.dataset, args.subset, split=args.split)
    id2label, label2id = label_mapping(val_ds)

    res = evaluate(
        args.model, val_ds, label2id=label2id, id2label=id2label, model_args=model_args
    )

    time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_folder = os.path.join(
        args.output_folder,
        f"{args.model}-{args.dataset}_{args.subset}_{args.split}_{time}",
    )
    os.makedirs(output_folder, exist_ok=True)

    output_file = os.path.join(output_folder, "result.json")
    with open(output_file, "w") as f:
        json.dump(res, f, indent=4)

    logger.info(f"Results saved to {os.path.abspath(output_file)}")
