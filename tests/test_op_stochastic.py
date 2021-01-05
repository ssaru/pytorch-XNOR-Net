import os
import sys

import pytest
import pytorch_lightning
import torch

from src.ops.utils import stochastic_quantize


@pytest.fixture(scope="module")
def fix_seed():
    pytorch_lightning.seed_everything(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


stochastic_quantize_test_case = [
    # weights, num_iters,
    (10000, torch.tensor(0.0001), torch.tensor(0.0)),
]


@pytest.mark.parametrize("num_iters, test_weight, expected_value", stochastic_quantize_test_case)
def test_stochastic_quantize(num_iters, test_weight, expected_value):
    # 1. -1과 1을 ``sigmoid(0.0001)=0.5=50%``확률로 뽑는다.
    # 2. 뽑은 모든 값을 더한다.
    # 3. 더한 값을 뽑은 횟수만큼 나눈다.
    # 4. 50%확률로 잘 샘플링이 되었다면 이 값은 0에 근사해야한다.
    s = 0
    with torch.no_grad():
        for _ in range(num_iters):
            s += stochastic_quantize(test_weight)

    s /= num_iters

    assert torch.allclose(input=s, other=expected_value, rtol=1e-02, atol=1e-02, equal_nan=True)
