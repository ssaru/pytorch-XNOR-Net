import logging
import os
import sys

import pytest
import pytorch_lightning
import torch

from src.ops.binarized_linear import BinarizedLinear, binarized_linear
from src.types import quantization
from src.utils import prod

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


mode_test_case = [
    # "test_input, test_weight, test_bias, test_mode"
    (
        (
            torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            torch.tensor(1.0),
        ),
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
        binarized_linear(test_input, test_weight, test_bias, test_mode)


forward_test_case = [
    # (test_input, test_weight, test_bias, test_mode, expected)
    (
        # [[1.0, 1.0, 1.0],   [[-1.0, 1.0, 1.0],                          [[1.0, 1.0, 1.0],
        #  [1.0, 1.0, 1.0], X  [1.0, -1.0, -1.0], X scale factor(0.9) =   [1.0, 1.0, 1.0], X 0.9
        #  [1.0, 1.0, 1.0]]    [1.0, 1.0, 1.0]]                            [1.0, 1.0, 1.0]]
        torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        torch.tensor([[-1.0, 1.0, 1.0], [1.0, -0.8, 1.0], [1.0, -0.3, 1.0]]),
        None,
        quantization.QType.DETER,
        torch.tensor([[0.9, 0.9, 0.9], [0.9, 0.9, 0.9], [0.9, 0.9, 0.9]]),
    ),
    (
        # [[1.0, 1.0, 1.0],   [[-1.0, 1.0, 1.0],                              [[1.0, 1.0, 1.0],
        #  [1.0, 1.0, 1.0], X  [1.0, -1.0, -1.0], X scale factor(0.9) + 1 =    [1.0, 1.0, 1.0], X 0.9 + 1
        #  [1.0, 1.0, 1.0]]    [1.0, 1.0, 1.0]]                                [1.0, 1.0, 1.0]]
        torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        torch.tensor([[-1.0, 1.0, 1.0], [1.0, -0.8, 1.0], [1.0, -0.3, 1.0]]),
        torch.tensor([1.0]),
        quantization.QType.DETER,
        torch.tensor([[1.9, 1.9, 1.9], [1.9, 1.9, 1.9], [1.9, 1.9, 1.9]]),
    ),
    (
        # [[1.0, 1.0, 1.0],   [[1.0, 1.0, 1.0],                             [[3.0, -1.0, -1.0],
        #  [1.0, 1.0, 1.0], X  [1.0, -1.0, -1.0], X scale factor(0.0111) =   [3.0, -1.0, -1.0], X 0.0111
        #  [1.0, 1.0, 1.0]]    [1.0, -1.0, -1.0]]                            [3.0, -1.0, -1.0]]
        torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        torch.tensor([[-1.0, 1.0, 1.0], [1.0, -0.8, 1.0], [1.0, -0.3, 1.0]]),
        None,
        quantization.QType.STOCH,
        torch.tensor(
            [
                [0.0333, -0.0111, -0.0111],
                [0.0333, -0.0111, -0.0111],
                [0.0333, -0.0111, -0.0111],
            ]
        ),
    ),
    (
        # [[1.0, 1.0, 1.0],   [[1.0, 1.0, 1.0],                                 [[3.0, -1.0, -1.0],
        #  [1.0, 1.0, 1.0], X  [1.0, -1.0, -1.0], X scale factor(0.0111) + 1 =   [3.0, -1.0, -1.0], X 0.0111 + 1
        #  [1.0, 1.0, 1.0]]    [1.0, -1.0, -1.0]]                                [3.0, -1.0, -1.0]]
        torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        torch.tensor([[-1.0, 1.0, 1.0], [1.0, -0.8, 1.0], [1.0, -0.3, 1.0]]),
        torch.tensor([1.0]),
        quantization.QType.STOCH,
        torch.tensor(
            [
                [1.0333, 0.9889, 0.9889],
                [1.0333, 0.9889, 0.9889],
                [1.0333, 0.9889, 0.9889],
            ]
        ),
    ),
]


@pytest.mark.parametrize(
    "test_input, test_weight, test_bias, test_mode, expected", forward_test_case
)
def test_forward(fix_seed, test_input, test_weight, test_bias, test_mode, expected):

    answer = binarized_linear(test_input, test_weight, test_bias, test_mode)

    logger.debug(f"answer: {answer}")
    logger.debug(f"expected: {expected}")
    assert torch.allclose(
        input=answer, other=expected, rtol=1e-04, atol=1e-04, equal_nan=True,
    )


indirectly_backward_test_case = [
    # (test_input, test_weight, test_bias, test_mode, expected_input_grad, expected_weight_grad, expected_bias_grad)
    (
        # [[1.0, -1.0, 1.0]], X  [[-1.0], [1.0], [-1.0]], X scale factor(1.35) =   [[-3.0]], X 1.35
        torch.tensor([[1.0, -1.0, 1.0]], requires_grad=True),
        torch.tensor([[-1.2, 1.5, -1.35]], requires_grad=True),
        None,
        quantization.QType.DETER,
        torch.tensor([[-1.35, 1.35, -1.35]]),
        torch.tensor([[1.0, -1.0, 1.0]]),
        None,
    ),
    (
        # [[1.0, -1.0, 1.0]], X  [[-1.0], [1.0], [-1.0]], X scale factor(1.35) =   [[-3.0]], X 1.35 + 1
        torch.tensor([[1.0, -1.0, 1.0]], requires_grad=True),
        torch.tensor([[-1.2, 1.5, -1.35]], requires_grad=True),
        torch.tensor([1.0], requires_grad=True),
        quantization.QType.DETER,
        torch.tensor([[-1.35, 1.35, -1.35]]),
        torch.tensor([[1.0, -1.0, 1.0]]),
        torch.tensor([1.0]),
    ),
    (
        # [[1.0, -1.0, 1.0]], X  [[1.0], [1.0], [-1.0]], X scale factor(-0.349) =   [[-3.0]], X -0.349
        torch.tensor([[1.0, -1.0, 1.0]], requires_grad=True),
        torch.tensor([[-1.2, 1.5, -1.35]], requires_grad=True),
        None,
        quantization.QType.STOCH,
        torch.tensor([[-0.35, -0.35, 0.35]]),
        torch.tensor([[1.0, -1.0, 1.0]]),
        None,
    ),
    (
        # [[1.0, -1.0, 1.0]], X  [[1.0], [1.0], [-1.0]], X scale factor(-0.349) =   [[-3.0]], X -0.349
        torch.tensor([[1.0, -1.0, 1.0]], requires_grad=True),
        torch.tensor([[-1.2, 1.5, -1.35]], requires_grad=True),
        torch.tensor([1.0], requires_grad=True),
        quantization.QType.STOCH,
        torch.tensor([[-0.35, -0.35, 0.35]]),
        torch.tensor([[1.0, -1.0, 1.0]]),
        torch.tensor([1.0]),
    ),
]


@pytest.mark.parametrize(
    "test_input, test_weight, test_bias, test_mode, expected_input_grad, expected_weight_grad, expected_bias_grad",
    indirectly_backward_test_case,
)
def test_backward_indirectly(
    fix_seed,
    test_input,
    test_weight,
    test_bias,
    test_mode,
    expected_input_grad,
    expected_weight_grad,
    expected_bias_grad,
):

    binarized_linear(test_input, test_weight, test_bias, test_mode).backward()

    logger.debug(f"input grad: {test_input.grad}")
    logger.debug(f"expected input grad: {expected_input_grad}")

    logger.debug(f"weight grad: {test_weight.grad}")
    logger.debug(f"expected weight grad: {expected_weight_grad}")

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

    if expected_bias_grad:
        logger.debug(f"bias grad: {test_bias.grad}")
        logger.debug(f"expected bias grad: {expected_bias_grad}")

        assert torch.allclose(
            input=test_bias.grad,
            other=expected_bias_grad,
            rtol=1e-04,
            atol=1e-04,
            equal_nan=True,
        )


directly_backward_test_case = [
    # (saved_tensors, needs_input_grad, grad_output, expected_input_grad, expected_weight_grad, expected_bias_grad)
    (
        (
            torch.tensor([[1.0, -1.0, 1.0]]),
            torch.tensor([[-1.35, 1.35, -1.35]]),
            torch.tensor([1]),
        ),
        (True, True, True, False),
        torch.tensor([[1.0]]),
        torch.tensor([[-1.35, 1.35, -1.35]]),
        torch.tensor([[1.0, -1.0, 1.0]]),
        torch.tensor([1.0]),
    )
]


@pytest.mark.parametrize(
    "saved_tensors, needs_input_grad, grad_output, expected_input_grad, expected_weight_grad, expected_bias_grad",
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

    ctx = CTX(saved_tensors, needs_input_grad)

    input_grad, weight_grad, bias_grad, _ = BinarizedLinear.backward(ctx, grad_output)

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
