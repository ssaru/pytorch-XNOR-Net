import os
import sys

import pytest
import pytorch_lightning
import torch

from src.transforms.binarized_input import BinarizeWithScaleFactor

input_transform_test_case = [
    # (input, expected_value)
    # (-5, 5)
    (
        torch.tensor(
            [
                [[5.0, -3.0, 2.0], [3.0, -3.0, 3.0], [5.0, 5.0, -5.0]],
                [[1.0, 1.0, 1.0], [-1.0, -3.0, 0.0], [5.0, 4.0, -4.0]],
                [[-4.0, 4.0, -3.0], [-2.0, 2.0, -1.0], [5.0, -5.0, -3.0]],
            ]
        ),
        (
            torch.tensor(
                [
                    [[1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]],
                    [[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [1.0, 1.0, -1.0]],
                    [[-1.0, 1.0, -1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]],
                ]
            ),
            torch.tensor(9.2222),
        ),
    ),
]


@pytest.mark.parametrize("input, expected_value", input_transform_test_case)
def test_transforms(fix_seed, input, expected_value):
    binarzied_input, scale_factor = BinarizeWithScaleFactor()(tensor=input)
    expected_binarized_input, expected_scale_factor = expected_value

    assert torch.allclose(
        input=binarzied_input,
        other=expected_binarized_input,
        rtol=1e-04,
        atol=1e-04,
        equal_nan=True,
    )

    assert torch.allclose(
        input=scale_factor,
        other=expected_scale_factor,
        rtol=1e-04,
        atol=1e-04,
        equal_nan=True,
    )
