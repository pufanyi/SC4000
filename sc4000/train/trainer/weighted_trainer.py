import torch
from transformers import Trainer, AdamW, Adafactor, get_scheduler
from peft import LoftQConfig, LoraConfig, get_peft_model


class WeightedTrainer(Trainer):
    def __init__(
        self,
        model,
        *args,
        weights,
        label_smoothing,
        optimizer: str = "adafactor",
        scheduler: str = "reduce_lr_on_plateau",
        use_lora: bool = False,
        **kwargs,
    ):
        if use_lora:
            self.loftq_config = LoftQConfig()
            self.lora_config = LoraConfig(
                init_lora_weights="loftq", loftq_config=self.loftq_config
            )
            self.model = get_peft_model(model, self.lora_config)
        weights = [weights[w] for w in sorted(list(weights.keys()))]
        self.weight = torch.tensor(weights, dtype=torch.float32)
        self.label_smoothing = label_smoothing
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        super().__init__(model, *args, **kwargs)

    def create_optimizer(self):
        if self.optimizer_name == "adamw":
            return AdamW(self.model.parameters(), lr=self.args.learning_rate)
        elif self.optimizer_name == "adafactor":
            return Adafactor(self.model.parameters(), lr=self.args.learning_rate, scale_parameter=False, relative_step=False)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    def create_scheduler(self, num_training_steps: int):
        if self.scheduler_name == "reduce_lr_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10)
        else:
            return get_scheduler(
                self.scheduler_name,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
            )

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.weight.to(logits.dtype).to(logits.device),
            label_smoothing=self.label_smoothing,
        )
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss