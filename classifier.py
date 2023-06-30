import torch
import torchvision.models as models
from timm import create_model, list_models


class BreastCancerModel(torch.nn.Module):
    def __init__(self, model_type, dropout=0.0):
        super().__init__()
        self.model = create_model(
            model_type, pretrained=True, num_classes=0, drop_rate=dropout
        )

        self.backbone_dim = self.model(torch.randn(1, 3, 256, 256)).shape[-1]

        self.nn_cancer = torch.nn.Sequential(
            torch.nn.Linear(self.backbone_dim, 1),
        )

    def forward(self, x):
        x = self.model(x)
        cancer = self.nn_cancer(x).squeeze()

        return cancer

    def predict(self, x):
        cancer = self.forward(x)
        return torch.sigmoid(cancer)
