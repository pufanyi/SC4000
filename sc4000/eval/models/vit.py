from sc4000.eval.models.base import Model
from sc4000.eval.utils.results import Result

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image


class ViT(Model):
    def __init__(
        self, *, pretrained="google/vit-large-patch16-224", id2label=None, label2id=None
    ):
        super().__init__("ViT")
        self.image_processor = ViTImageProcessor.from_pretrained(pretrained)
        self.model = ViTForImageClassification.from_pretrained(
            pretrained,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

    def predict(self, image: Image.Image) -> Result:
        inputs = self.image_processor(image, return_tensors="pt")
        outputs = self.model(**inputs)
        logists_list = list(map(float, outputs.logits[0].detach().numpy()))
        return Result(
            prediction=outputs.logits.argmax(dim=-1).item(), logs=logists_list
        )
