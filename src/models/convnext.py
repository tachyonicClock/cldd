from transformers import ConvNextV2ForImageClassification
from transformers.models.convnextv2.configuration_convnextv2 import ConvNextV2Config
from torch import nn, Tensor


class ConvNextV2(nn.Module):
    """ConvNextV2 model for image classification.

    >>> import torch
    >>> model = ConvNextV2(num_classes=100)
    >>> input_tensor = torch.randn(2, 3, 32, 32)  # Example input tensor
    >>> model.features(input_tensor).shape
    torch.Size([2, 320])
    >>> model(input_tensor).shape
    torch.Size([2, 100])

    """

    def __init__(self, num_classes: int, pretrained=True):
        super(ConvNextV2, self).__init__()
        if pretrained:
            self.model = ConvNextV2ForImageClassification.from_pretrained(
                "facebook/convnextv2-atto-1k-224"
            )
        else:
            config = ConvNextV2Config.from_pretrained("facebook/convnextv2-atto-1k-224")
            self.model = ConvNextV2ForImageClassification(config)

        self.model.classifier = nn.Linear(
            self.model.classifier.in_features,  # type: ignore
            num_classes,
        )

    def features(self, x: Tensor) -> Tensor:
        return self.model.convnextv2(x).pooler_output

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x).logits
