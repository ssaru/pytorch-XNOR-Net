import logging
import os
import sys

import pytest
import pytorch_lightning
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from src.engine.train_jig import TrainingContainer
from src.nn.binarized_conv2d import BinarizedConv2d
from src.utils import build_model, get_config, get_data_loaders

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def tearup_mlp_config() -> DictConfig:
    config_path = {
        "--data-config": "conf/mlp/data/data.yml",
        "--model-config": "conf/mlp/model/model.yml",
        "--training-config": "conf/mlp/training/training.yml",
    }
    config_list = ["--data-config", "--model-config", "--training-config"]
    config: DictConfig = get_config(hparams=config_path, options=config_list)

    return config


def tearup_conv_config() -> DictConfig:
    config_path = {
        "--data-config": "conf/conv/data/data.yml",
        "--model-config": "conf/conv/model/model.yml",
        "--training-config": "conf/conv/training/training.yml",
    }
    config_list = ["--data-config", "--model-config", "--training-config"]
    config: DictConfig = get_config(hparams=config_path, options=config_list)

    return config


train_test_case = [
    # (config, gpus)
    (tearup_mlp_config(), None),
    (tearup_mlp_config(), 0),
    (tearup_conv_config(), None),
    (tearup_conv_config(), 0),
]


@pytest.mark.parametrize("config, gpus", train_test_case)
def test_train_pipeline(fix_seed, config, gpus):
    config = OmegaConf.create(config)

    train_dataloader, test_dataloader = get_data_loaders(config=config.data)
    model = build_model(model_conf=config.model)
    training_container = TrainingContainer(
        model=model, config=config.training_container
    )

    trainer_params = dict(config.trainer.params)
    trainer_params["limit_train_batches"] = 0.1
    trainer_params["limit_val_batches"] = 0.1
    trainer_params["max_epochs"] = 2
    trainer_params["gpus"] = gpus
    if not gpus:
        trainer_params["accelerator"] = None

    trainer = Trainer(**trainer_params)

    trainer.fit(
        model=training_container,
        train_dataloader=train_dataloader,
        val_dataloaders=test_dataloader,
    )
