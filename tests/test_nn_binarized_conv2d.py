import logging
import os
import sys

import pytest
import pytorch_lightning
import torch

from src.nn.binarized_conv2d import BinarizedConv2d
from src.types import quantization

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


forward_test_case = [
    # (device, test_input, test_bias, test_mode, exptected_shape)
    ("cpu", torch.rand((1, 1, 3, 3)), False, quantization.QType.DETER, (1, 1, 1, 1)),
    (
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        torch.rand((1, 1, 3, 3)),
        False,
        quantization.QType.DETER,
        (1, 1, 1, 1),
    ),
    ("cpu", torch.rand((1, 1, 3, 3)), True, quantization.QType.DETER, (1, 1, 1, 1)),
    (
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        torch.rand((1, 1, 3, 3)),
        True,
        quantization.QType.DETER,
        (1, 1, 1, 1),
    ),
    ("cpu", torch.rand((1, 1, 3, 3)), False, quantization.QType.STOCH, (1, 1, 1, 1)),
    (
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        torch.rand((1, 1, 3, 3)),
        False,
        quantization.QType.STOCH,
        (1, 1, 1, 1),
    ),
    ("cpu", torch.rand((1, 1, 3, 3)), True, quantization.QType.STOCH, (1, 1, 1, 1)),
    (
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        torch.rand((1, 1, 3, 3)),
        True,
        quantization.QType.STOCH,
        (1, 1, 1, 1),
    ),
]


@pytest.mark.parametrize(
    "device, test_input, test_bias, test_mode, exptected_shape", forward_test_case
)
def test_foward(fix_seed, device, test_input, test_bias, test_mode, exptected_shape):
    test_input = test_input.to(device)
    model = BinarizedConv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=test_bias,
        padding_mode="zeros",
        mode=test_mode,
    ).to(device)

    model_output_shape = model(test_input).to("cpu").shape

    logger.debug(f"device : {device}")
    logger.debug(f"input : {test_input}")
    logger.debug(f"bias : {test_bias}")
    logger.debug(f"mode : {test_mode}")
    logger.debug(f"model output shape : {model_output_shape}")
    logger.debug(f"expected shape : {exptected_shape}")

    assert model_output_shape == exptected_shape


clipping_test_case = [
    (
        "cpu",
        torch.rand((1, 1, 3, 3)),
        False,
        quantization.QType.DETER,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    ),
    (
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        torch.rand((1, 1, 3, 3)),
        False,
        quantization.QType.DETER,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    ),
    (
        "cpu",
        torch.rand((1, 1, 3, 3)),
        True,
        quantization.QType.DETER,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    ),
    (
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        torch.rand((1, 1, 3, 3)),
        True,
        quantization.QType.DETER,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    ),
    (
        "cpu",
        torch.rand((1, 1, 3, 3)),
        False,
        quantization.QType.STOCH,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    ),
    (
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        torch.rand((1, 1, 3, 3)),
        False,
        quantization.QType.STOCH,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    ),
    (
        "cpu",
        torch.rand((1, 1, 3, 3)),
        True,
        quantization.QType.STOCH,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    ),
    (
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        torch.rand((1, 1, 3, 3)),
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
    fix_seed,
    device,
    test_input,
    test_bias,
    test_mode,
    exptected_max_value,
    exptected_min_value,
):

    test_input = torch.rand((1, 1, 3, 3)).to(device)
    model = BinarizedConv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=test_bias,
        padding_mode="zeros",
        mode=test_mode,
    ).to(device)

    with torch.no_grad():
        model.weight.mul_(100)

    model(test_input)

    with torch.no_grad():
        assert model.weight.min() >= exptected_min_value
        assert model.weight.max() >= exptected_max_value
