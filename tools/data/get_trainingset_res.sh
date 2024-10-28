python -m sc4000.eval.evaluate --model=convnextv2 --model_args=pretrained=pufanyi/SC4000_ConvNeXtV2_base_full_9000 --subset=full --split=train

python -m sc4000.eval.evaluate --model=convnextv2 --model_args=pretrained=pufanyi/SC4000_ConvNeXtV2_base_balanced_12500 --subset=full --split=train

python -m sc4000.eval.evaluate --model=vit --model_args=pretrained=pufanyi/SC4000_vit_large_full_15900 --subset=full --split=train

python -m sc4000.eval.evaluate --model=mobilenetv3 --subset=full --split=train
