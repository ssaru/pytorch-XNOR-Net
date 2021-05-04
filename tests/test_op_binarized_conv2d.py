import logging
import os
import sys

import pytest
import pytorch_lightning
import torch

from src.ops.binarized_conv2d import BinarizedConv2d, binarized_conv2d
from src.types import quantization

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

mode_test_case = [
    # (test_input, test_weight, test_bias, test_mode)
    (
        torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        torch.tensor([[-1.0, 1.0, 1.0], [1.0, -0.8, 1.0], [1.0, -0.3, 1.0]]),
        None,
        "test",
    )
]


@pytest.mark.parametrize(
    "test_input, test_weight, test_bias, test_mode", mode_test_case
)
def test_supported_mode(fix_seed, test_input, test_weight, test_bias, test_mode):
    with pytest.raises(RuntimeError):
        binarized_conv2d(test_input, test_weight, test_bias, 1, 0, 1, 1, test_mode)


forward_test_case = [
    # (test_input, test_weight, test_bias, test_mode, expected)
    (
        torch.tensor([[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]]),
        torch.tensor([[[[-1.0, 1.0, 1.0], [1.0, -0.8, 1.0], [1.0, -0.3, 1.0]]]]),
        None,
        quantization.QType.DETER,
        torch.tensor([[[[2.7]]]]),
    ),
    (
        torch.tensor([[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]]),
        torch.tensor([[[[-1.0, 1.0, 1.0], [1.0, -0.8, 1.0], [1.0, -0.3, 1.0]]]]),
        torch.tensor([1.0]),
        quantization.QType.DETER,
        torch.tensor([[[[3.6]]]]),
    ),
    (
        torch.tensor([[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]]),
        torch.tensor([[[[-1.0, 0.5, 0.5], [0.5, -0.8, 1.0], [0.1, -0.3, 0.5]]]]),
        None,
        quantization.QType.STOCH,
        torch.tensor([[[[0.1778]]]]),
    ),
    (
        torch.tensor([[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]]),
        torch.tensor([[[[-1.0, 1.0, 1.0], [1.0, -0.8, 1.0], [1.0, -0.3, 1.0]]]]),
        torch.tensor([1.0]),
        quantization.QType.STOCH,
        torch.tensor([[[[0.0222]]]]),
    ),
]


@pytest.mark.parametrize(
    "test_input, test_weight, test_bias, test_mode, expected", forward_test_case
)
def test_forward(fix_seed, test_input, test_weight, test_bias, test_mode, expected):

    answer = binarized_conv2d(test_input, test_weight, test_bias, 1, 0, 1, 1, test_mode)

    logger.debug(f"answer: {answer}")
    logger.debug(f"test mode: {test_mode}")
    logger.debug(f"expected: {expected}")

    assert torch.allclose(
        input=answer, other=expected, rtol=1e-04, atol=1e-04, equal_nan=True,
    )


indirectly_backward_test_case = [
    # (test_input, test_weight, test_bias, test_mode, expected_weight_grad, expected_input_grad)
    (
        torch.tensor(
            [[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]], requires_grad=True
        ),
        torch.tensor(
            [[[[-1.0, 1.0, 1.0], [1.0, -0.8, 1.0], [1.0, -0.3, 1.0]]]],
            requires_grad=True,
        ),
        None,
        quantization.QType.DETER,
        torch.tensor([[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]]),
        torch.tensor([[[[-0.9, 0.9, 0.9], [0.9, -0.9, 0.9], [0.9, -0.9, 0.9]]]]),
    ),
    (
        torch.tensor(
            [[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]], requires_grad=True
        ),
        torch.tensor(
            [[[[-1.0, 1.0, 1.0], [1.0, -0.8, 1.0], [1.0, -0.3, 1.0]]]],
            requires_grad=True,
        ),
        torch.tensor([1.0]),
        quantization.QType.DETER,
        torch.tensor([[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]]),
        torch.tensor([[[[-0.9, 0.9, 0.9], [0.9, -0.9, 0.9], [0.9, -0.9, 0.9]]]]),
    ),
    (
        torch.tensor(
            [[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]], requires_grad=True
        ),
        torch.tensor(
            [[[[-1.0, 1.0, 1.0], [1.0, -0.8, 1.0], [1.0, -0.3, 1.0]]]],
            requires_grad=True,
        ),
        None,
        quantization.QType.STOCH,
        torch.tensor([[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]]),
        torch.tensor(
            [
                [
                    [
                        [0.0111, 0.0111, 0.0111],
                        [0.0111, -0.0111, -0.0111],
                        [0.0111, -0.0111, -0.0111],
                    ]
                ]
            ]
        ),
    ),
    (
        torch.tensor(
            [[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]], requires_grad=True
        ),
        torch.tensor(
            [[[[-1.0, 1.0, 1.0], [1.0, -0.8, 1.0], [1.0, -0.3, 1.0]]]],
            requires_grad=True,
        ),
        torch.tensor([1.0]),
        quantization.QType.STOCH,
        torch.tensor([[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]]),
        torch.tensor(
            [
                [
                    [
                        [0.0111, 0.0111, 0.0111],
                        [0.0111, -0.0111, -0.0111],
                        [0.0111, -0.0111, -0.0111],
                    ]
                ]
            ]
        ),
    ),
]


@pytest.mark.parametrize(
    "test_input, test_weight, test_bias, test_mode, expected_weight_grad, expected_input_grad",
    indirectly_backward_test_case,
)
def test_backward_indirectly(
    fix_seed,
    test_input,
    test_weight,
    test_bias,
    test_mode,
    expected_weight_grad,
    expected_input_grad,
):

    binarized_conv2d(
        test_input, test_weight, test_bias, 1, 0, 1, 1, test_mode
    ).backward()

    logger.info(f"input grad : {test_input.grad}")
    logger.info(f"expected input grad : {expected_input_grad}")

    assert torch.allclose(
        input=test_input.grad,
        other=expected_input_grad,
        rtol=1e-04,
        atol=1e-04,
        equal_nan=True,
    )

    assert torch.allclose(
        input=test_weight.grad,
        other=expected_weight_grad,
        rtol=1e-04,
        atol=1e-04,
        equal_nan=True,
    )


directly_backward_test_case = [
    # (saved_tensors, needs_input_grad, grad_output, expected_weight_grad, expected_input_grad, expected_bias_grad)
    (
        (
            torch.tensor(
                [[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]],
                requires_grad=True,
            ),
            torch.tensor(
                [[[[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, -1.0, 1.0]]]],
                requires_grad=True,
            ),
            torch.tensor([1]),
        ),
        (True, True, True, False),
        torch.tensor([[[[1.0]]]]),
        torch.tensor([[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]]),
        torch.tensor([[[[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, -1.0, 1.0]]]]),
        torch.tensor(1.0),
    ),
]


@pytest.mark.parametrize(
    "saved_tensors, needs_input_grad, grad_output, expected_weight_grad, expected_input_grad, expected_bias_grad",
    directly_backward_test_case,
)
def test_backward_directly(
    fix_seed,
    saved_tensors,
    needs_input_grad,
    grad_output,
    expected_weight_grad,
    expected_input_grad,
    expected_bias_grad,
):
    class CTX:
        def __init__(self, saved_tensors, needs_input_grad):
            self.saved_tensors = saved_tensors
            self.needs_input_grad = needs_input_grad

            self.stride = 1
            self.padding = 0
            self.dilation = 1
            self.groups = 1

    ctx = CTX(saved_tensors, needs_input_grad)

    input_grad, weight_grad, bias_grad, _, _, _, _, _ = BinarizedConv2d.backward(
        ctx, grad_output
    )

    assert torch.allclose(
        input=input_grad,
        other=expected_input_grad,
        rtol=1e-04,
        atol=1e-04,
        equal_nan=True,
    )

    assert torch.allclose(
        input=weight_grad,
        other=expected_weight_grad,
        rtol=1e-04,
        atol=1e-04,
        equal_nan=True,
    )

    assert torch.allclose(
        input=bias_grad,
        other=expected_bias_grad,
        rtol=1e-04,
        atol=1e-04,
        equal_nan=True,
    )
