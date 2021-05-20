"""
Usage:
    main.py train [options] [--dataset-config=<dataset config path>] [--model-config=<model config path>] [--runner-config=<runner config path>]
    main.py train (-h | --help)
Options:
    --dataset-config <dataset config path>  Path to YAML file for dataset configuration  [default: conf/mlp/data/data.yml] [type: path]
    --model-config <model config path>  Path to YAML file for model configuration  [default: conf/mlp/model/model.yml] [type: path]
    --runner-config <runner config path>  Path to YAML file for model configuration  [default: conf/mlp/training/training.yml] [type: path]
    -h --help  Show this.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch.nn as nn
import torchvision.transforms as transforms
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.utils.data import DataLoader

import wandb
from src.engine.train_jig import TrainingContainer
from src.model import net as Net
from src.model.net import BinaryConv, BinaryLinear
from src.utils import (
    build_model,
    get_checkpoint_callback,
    get_config,
    get_data_loaders,
    get_early_stopper,
    get_log_dir,
    get_next_version,
    get_wandb_logger,
    load_class,
)

# TODO. Hyperparameter search
# Search space
# Batch size: [32(ing), 64, 128, 256]
# optimizer: [Adam, SGD, etc...]
# lr : [1e-4(ing), 1e-3, 1e-2, 1e-1]
# scheduler gamma: [0.1(ing), 0.2, 0.4, 0.6, 0.8, 0.9]


def train(hparams: dict):
    config_list = ["--dataset-config", "--model-config", "--runner-config"]
    config: DictConfig = get_config(hparams=hparams, options=config_list)

    # TODO. 임시방편
    OmegaConf.set_readonly(config, False)
    for key, value in config.hyperparameter.items():
        config.runner.experiments.name += "-" + str(key) + "_" + str(value)
    OmegaConf.set_readonly(config, True)

    log_dir = get_log_dir(config=config)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_dataloader, test_dataloader = get_data_loaders(config=config)

    model: nn.Module = build_model(model_conf=config.model)
    training_container: pl.LightningModule = TrainingContainer(
        model=model, config=config
    )

    checkpoint_callback = get_checkpoint_callback(log_dir=log_dir, config=config)
    wandb_logger = get_wandb_logger(log_dir=log_dir, config=config)
    wandb_logger.watch(model, log="gradients", log_freq=100)

    lr_logger = LearningRateMonitor()
    early_stop_callback = get_early_stopper(
        early_stopping_config=config.runner.earlystopping.params
    )

    with (log_dir / Path("config.yaml")).open("w") as f:
        OmegaConf.save(config=config, f=f)

    trainer = Trainer(
        accelerator=config.runner.trainer.distributed_backend,
        fast_dev_run=False,
        gpus=config.runner.trainer.params.gpus,
        amp_level="O2",
        logger=wandb_logger,
        callbacks=[early_stop_callback, lr_logger],
        checkpoint_callback=checkpoint_callback,
        max_epochs=config.runner.trainer.params.max_epochs,
        weights_summary="top",
        reload_dataloaders_every_epoch=False,
        resume_from_checkpoint=None,
        benchmark=False,
        deterministic=True,
        num_sanity_val_steps=0,
        overfit_batches=0.0,
        precision=32,
        profiler=True,
        limit_train_batches=1.0,
    )
    trainer.fit(
        model=training_container,
        train_dataloader=train_dataloader,
        val_dataloaders=test_dataloader,
    )
