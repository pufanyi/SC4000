import torch
from transformers import Trainer


class WeightedTrainer(Trainer):
    def __init__(self, *args, weights, label_smoothing, **kwargs):
        weights = [weights[w] for w in sorted(list(weights.keys()))]
        self.weight = torch.tensor(weights, dtype=torch.float32)
        self.label_smoothing = label_smoothing
        super().__init__(*args, **kwargs)

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