"""CartilAI network: MRNet-style per-slice 2D CNN with max-pooling across slices.

Architecture (Bien et al., 2018, "Deep-learning-assisted diagnosis for knee MRI"),
with ResNet18 in place of AlexNet::

    volume (S, 3, H, W)
      -> ResNet18 feature extractor, applied to every slice   (S, 512, h, w)
      -> global average pool over each slice                    (S, 512)
      -> max-pool across slices                                 (512,)
      -> dropout + linear layer                                 (3,) logits
      -> sigmoid                                                one probability per finding
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


class CartilAINet(nn.Module):
    """Multi-label exam classifier for knee MRI volumes."""

    def __init__(self, num_findings: int = 3, pretrained: bool = True, dropout: float = 0.3) -> None:
        """Build the network.

        Args:
            num_findings: Number of outputs (one logit per finding).
            pretrained: Load ImageNet weights for the ResNet18 backbone. Falls back
                to random initialisation (with a warning) if they cannot be downloaded.
            dropout: Dropout probability before the classifier.
        """
        super().__init__()
        weights = None
        if pretrained:
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        try:
            resnet = torchvision.models.resnet18(weights=weights)
        except Exception as exc:  # no internet: keep going with random weights
            print(f"[CartilAI] Could not load ImageNet weights ({exc}); using random initialisation.")
            resnet = torchvision.models.resnet18(weights=None)
        # Everything up to (and including) layer4; drop avgpool + fc.
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        use_exam_batchnorm(self.features)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(512, num_findings)

    @property
    def gradcam_layer(self) -> nn.Module:
        """The last convolutional block, used as the Grad-CAM target layer."""
        return self.features[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score one exam.

        Args:
            x: Preprocessed volume ``(S, 3, H, W)``.

        Returns:
            Logits of shape ``(num_findings,)``.
        """
        fmap = self.features(x)                   # (S, 512, h, w)
        per_slice = self.pool(fmap).flatten(1)    # (S, 512)
        exam = per_slice.max(dim=0).values        # (512,)
        return self.classifier(self.dropout(exam))


def use_exam_batchnorm(module: nn.Module) -> None:
    """Make every BatchNorm layer normalise with the statistics of the current exam.

    One training step sees the slices of a single exam, so BatchNorm's running
    averages (used in eval mode) never match the per-exam statistics the network
    was trained with, and predictions degrade at inference time. Dropping the
    running statistics makes training and inference behave identically: each
    exam is normalised over its own slices. The learned scale/shift are kept.

    Args:
        module: Network whose BatchNorm layers are converted in place.
    """
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            m.num_batches_tracked = None


def build_model(config: dict, pretrained: bool | None = None) -> CartilAINet:
    """Construct the model from the config dictionary.

    Args:
        config: Full CartilAI config.
        pretrained: Override ``config["model"]["pretrained"]``.

    Returns:
        A :class:`CartilAINet`.
    """
    mcfg = config["model"]
    return CartilAINet(
        num_findings=len(config["findings"]),
        pretrained=mcfg["pretrained"] if pretrained is None else pretrained,
        dropout=mcfg["dropout"],
    )
