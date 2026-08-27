import os
import sys

import pytest
import torch

# add src to path so we can import from it directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import SimpleCNN, get_model


def test_resnet18_output_shape():
    model = get_model("resnet18", num_classes=10)
    model.eval()
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"


def test_simple_cnn_output_shape():
    model = SimpleCNN(num_classes=10)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 10)


def test_resnet18_num_classes_param():
    # make sure we actually change the head
    model_5 = get_model("resnet18", num_classes=5)
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out = model_5(x)
    assert out.shape == (1, 5)


def test_model_no_nan_in_output():
    model = get_model("resnet18", num_classes=10)
    model.eval()
    x = torch.randn(8, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert not torch.isnan(out).any(), "NaN found in model output"


def test_unknown_architecture_raises():
    with pytest.raises(ValueError):
        get_model("some_made_up_arch")


def test_resnet18_has_expected_param_count():
    model = get_model("resnet18", num_classes=10)
    n_params = sum(p.numel() for p in model.parameters())
    # adapted resnet18 should be around 11M params
    assert n_params > 1_000_000, f"Unexpectedly small model: {n_params} params"
