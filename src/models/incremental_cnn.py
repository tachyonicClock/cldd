import torch
from torch import Tensor, nn


class IncrementalCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5),
        )
        self.classifier = nn.Linear(64 * 8 * 8, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 2:
            side = round((x.shape[1] / 3) ** 0.5)
            x = x.view(-1, 3, side, side)
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
