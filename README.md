# CassavaLeafDiseaseClassification

[Kaggle Competition](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/overview)

## Installation

```sh
conda create --name SC4000 python=3.12
conda activate SC4000

git clone https://github.com/pufanyi/CassavaLeafDiseaseClassification.git
cd CassavaLeafDiseaseClassification
python -m pip install -e .
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
