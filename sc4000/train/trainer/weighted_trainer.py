import torch
from transformers import Trainer


class WeightedTrainer(Trainer):
    def __init__(self, *args, weights, **kwargs):
        weights = [weights[w] for w in sorted(list(weights.keys()))]
        self.weight = torch.tensor(weights)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.weight)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
