import logging
import sys

import pytest
import pytorch_lightning
import torch

"""
XXX
This Error refer to https://github.com/pytest-dev/pytest/issues/5502

--- Logging error ---
Traceback (most recent call last):
  File "/usr/lib/python3.6/logging/__init__.py", line 996, in emit
    stream.write(msg)
ValueError: I/O operation on closed file.
Call stack:
  File "/home/ubuntu/Documents/dev/Hephaestus-project/pytorch-XNOR-Net/env/lib/python3.6/site-packages/wandb/internal/internal.py", line 137, in handle_exit
    logger.info("Internal process exited")
Message: 'Internal process exited'
Arguments: ()
"""

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def fix_seed():
    pytorch_lightning.seed_everything(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.debug("Fix SEED")
