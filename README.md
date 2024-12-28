# Cassava Leaf Disease Classification

[Kaggle Competition](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/overview) / [Report](https://pufanyi.github.io/SC4000/report/main.pdf) / [Submission](https://www.kaggle.com/code/pufanyi/final-submission)

## Installation

```sh
conda create --name SC4000 python=3.10
conda activate SC4000

git clone https://github.com/pufanyi/CassavaLeafDiseaseClassification.git
cd CassavaLeafDiseaseClassification
python -m pip install -e .

```

If you want to train the [CropNet](https://www.kaggle.com/models/google/cropnet/tensorFlow2/classifier-cassava-disease-v1/1), you need to install `tflite-model-maker-nightly` manually:

```sh
python -m pip install --use-deprecated=legacy-resolver tflite-model-maker-nightly
```

## Train

```sh
python -m sc4000.train.run --model=vit --subset=full
```

## Evaluation

```sh
python -m sc4000.eval.evaluate --model=vit --subset=full
```

For the `--model_args` options, should input like this:

```sh
python -m sc4000.eval.evaluate --model=vit --model_args="pretrained=output/models/checkpoint-124" --subset=full
```
