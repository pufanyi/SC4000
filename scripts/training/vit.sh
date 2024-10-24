python -m sc4000.train.run --model=vit --subset=full --model_args=pretrained=google/vit-base-patch16-384 --training_args=lr=5e-5 --wandb_run_name=vit_base_full
# python -m sc4000.train.run --model=vit --subset=full --model_args=pretrained=google/vit-large-patch16-384 --training_args=lr=5e-5 --wandb_run_name=vit_large
