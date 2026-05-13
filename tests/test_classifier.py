import torch

from src.classifier import LinearClassifier


def test_classifier_shape():
    clf = LinearClassifier(d_in=128, num_classes=21)
    x = torch.randn(10, 128)
    assert clf(x).shape == (10, 21)
