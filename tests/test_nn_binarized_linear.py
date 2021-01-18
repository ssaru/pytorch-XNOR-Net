import os
import sys

import pytest
import pytorch_lightning
import torch

from src.nn.binarized_linear import BinarizedLinear
from src.types import quantization


forward_test_case = [
    # (device, test_input, test_bias, test_mode, exptected_shape)
    ("cpu", torch.rand((1, 10)), False, quantization.QType.DETER, (1, 20)),
    (
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        torch.rand((1, 10)),
        False,
        quantization.QType.DETER,
        (1, 20),
    ),
    ("cpu", torch.rand((1, 10)), True, quantization.QType.DETER, (1, 20)),
    (
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        torch.rand((1, 10)),
        True,
        quantization.QType.DETER,
        (1, 20),
    ),
    ("cpu", torch.rand((1, 10)), False, quantization.QType.STOCH, (1, 20)),
    (
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        torch.rand((1, 10)),
        False,
        quantization.QType.STOCH,
        (1, 20),
    ),
    ("cpu", torch.rand((1, 10)), True, quantization.QType.STOCH, (1, 20)),
    (
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        torch.rand((1, 10)),
        True,
        quantization.QType.STOCH,
        (1, 20),
    ),
]


@pytest.mark.parametrize(
    "device, test_input, test_bias, test_mode, exptected_shape", forward_test_case
)
def test_forward(fix_seed, device, test_input, test_bias, test_mode, exptected_shape):
    test_input = test_input.to(device)
    model = BinarizedLinear(10, 20, bias=test_bias, mode=test_mode).to(device)

    assert model(test_input).shape == exptected_shape


clipping_test_case = [
    (
        "cpu",
        torch.rand((1, 10)),
        False,
        quantization.QType.DETER,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    ),
    (
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        torch.rand((1, 10)),
        False,
        quantization.QType.DETER,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    ),
    (
        "cpu",
        torch.rand((1, 10)),
        True,
        quantization.QType.DETER,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    ),
    (
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        torch.rand((1, 10)),
        True,
        quantization.QType.DETER,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    ),
    (
        "cpu",
        torch.rand((1, 10)),
        False,
        quantization.QType.STOCH,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    ),
    (
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        torch.rand((1, 10)),
        False,
        quantization.QType.STOCH,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    ),
    (
        "cpu",
        torch.rand((1, 10)),
        True,
        quantization.QType.STOCH,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    ),
    (
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        torch.rand((1, 10)),
        True,
        quantization.QType.STOCH,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    ),
]


@pytest.mark.parametrize(
    "device, test_input, test_bias, test_mode, exptected_max_value, exptected_min_value",
    clipping_test_case,
)
def test_clipping(
    fix_seed, device, test_input, test_bias, test_mode, exptected_max_value, exptected_min_value,
):

    test_input = test_input.to(device)
    model = BinarizedLinear(10, 20, bias=test_bias, mode=test_mode).to(device)

    with torch.no_grad():
        model.weight.mul_(100)

    model(test_input)

    with torch.no_grad():
        assert model.weight.min() >= exptected_min_value
        assert model.weight.max() <= exptected_max_value
