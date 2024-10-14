import torch
from transformers import Trainer, AdamW, Adafactor, get_scheduler
from peft import LoftQConfig, LoraConfig, get_peft_model

from sc4000.utils.logger import setup_logger

logger = setup_logger(__name__)


class WeightedTrainer(Trainer):
    def __init__(
        self,
        model,
        *args,
        weights,
        label_smoothing: float = 0.06,
        lr_scheduler: str = "reduce_lr_on_plateau",
        lr_scheduler_kwargs: dict = {},
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
        self.scheduler_name = lr_scheduler
        super().__init__(model, *args, **kwargs)

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
    ):
        self.lr_scheduler = get_scheduler(
            self.scheduler_name,
            optimizer=self.optimizer if optimizer is None else optimizer,
            num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
            scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
        )
        return self.lr_scheduler

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
