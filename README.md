# CassavaLeafDiseaseClassification

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
python -m sc4000.train.trainer --model=vit --subset=full
```
