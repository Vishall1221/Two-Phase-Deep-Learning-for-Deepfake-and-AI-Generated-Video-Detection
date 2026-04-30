import torch
import torch.nn as nn
from torchvision import models

from config import DEVICE, PHASE1_MODEL_PATH


class TwoBranchFusionNet(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__()

        self.residual_branch = models.resnet18(weights=None)
        residual_feat_dim = self.residual_branch.fc.in_features
        self.residual_branch.fc = nn.Identity()

        self.fft_branch = models.resnet18(weights=None)
        fft_feat_dim = self.fft_branch.fc.in_features
        self.fft_branch.fc = nn.Identity()

        fusion_dim = residual_feat_dim + fft_feat_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, residual_x: torch.Tensor, fft_x: torch.Tensor) -> torch.Tensor:
        residual_feat = self.residual_branch(residual_x)
        fft_feat = self.fft_branch(fft_x)
        fused = torch.cat([residual_feat, fft_feat], dim=1)
        return self.classifier(fused)


def load_phase1_model() -> nn.Module:
    if not PHASE1_MODEL_PATH.exists():
        raise FileNotFoundError(f"Phase 1 model file not found: {PHASE1_MODEL_PATH}")

    model = TwoBranchFusionNet(num_classes=2, dropout=0.3)
    state = torch.load(PHASE1_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model
