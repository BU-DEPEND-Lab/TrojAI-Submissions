from copy import deepcopy
from dataclass import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union
from pydantic import BaseModel, Field

import yaml
import json
from abc import ABC, abstractmethod

def merge(base: Dict, update: Dict, updated: Set) -> Dict:
    "Recursively updates a nested dictionary with new values"
    for k, v in base.items():
        if k in update and isinstance(v, dict):
            base[k] = merge(v, update[k], updated)
            updated.add(k)
        elif k in update:
            base[k] = update[k]
            updated.add(k)

    return base

def _merge_dicts(base: Dict, update: Dict) -> Dict:
    "Merge two dictionaries recursively, returning a new dictionary."

    base = deepcopy(base)

    for k, v in update.items():
        if isinstance(v, dict):
            base[k] = _merge_dicts(base.get(k, {}), v)
        else:
            base[k] = v

    return base


class BaseConfig(BaseModel, ABC):
    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)


class AlgorithmConfig(BaseConfig):
    """
    Config for an optimizer.

    :param name: Name of the optimizer
    :type name: str

    :param kwargs: Keyword arguments for the optimizer (e.g. lr, betas, eps, weight_decay)
    :type kwargs: Dict[str, Any]
    """

    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)



class ModelConfig(BaseConfig):
    """
    Config for an optimizer.

    :param name: Name of the optimizer
    :type name: str

    :param kwargs: Keyword arguments for the optimizer (e.g. lr, betas, eps, weight_decay)
    :type kwargs: Dict[str, Any]
    """

    name: str
    input_size: int
    output_size: int
    kwargs: Dict[str, Any] = field(default_factory=dict)


class OptimizerConfig(BaseConfig):
    """
    Config for an optimizer.

    :param name: Name of the optimizer
    :type name: str

    :param kwargs: Keyword arguments for the optimizer (e.g. lr, betas, eps, weight_decay)
    :type kwargs: Dict[str, Any]
    """

    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)



class TrainConfig(BaseConfig):
    """
    Config for train job on model.

    :param episodes: Total number of training episodes
    :type episodes: int
 
    :param batch_size: Batch size for training
    :type batch_size: int

    :param tracker: Tracker to use for logging. Default: "wandb"
    :type tracker: str

    :param checkpoint_interval: Save model every checkpoint_interval steps.
        Each checkpoint is stored in a sub-directory of the `TrainConfig.checkpoint_dir`
        directory in the format `checkpoint_dir/checkpoint_{step}`.
    :type checkpoint_interval: int

    :param eval_interval: Evaluate model every eval_interval steps
    :type eval_interval: int

    :param pipeline: Pipeline to use for training. One of the registered pipelines present in trlx.pipeline
    :type pipeline: str

    :param trainer: Trainer to use for training. One of the registered trainers present in trlx.trainer
    :type trainer: str

    :param trainer_kwargs: Extra keyword arguments for the trainer
    :type trainer: Dict[str, Any]

    :param project_name: Project name for wandb
    :type project_name: str

    :param entity_name: Entity name for wandb
    :type entity_name: str

    :param group_name: Group name for wandb (used for grouping runs)
    :type group_name: str

    :param checkpoint_dir: Directory to save checkpoints
    :type checkpoint_dir: str
 
    :param save_best: Save best model based on mean reward
    :type save_best: bool

    :param seed: Random seed
    :type seed: int

    :param minibatch_size: Size of model input during one forward pass. Must divide batch size
    :type minibatch_size: int
    """
    name: str
    epochs: int
    batch_size: int

    checkpoint_interval: int
    eval_interval: int

    pipeline: str  # One of the pipelines in framework.pipeline
    trainer_kwargs: Dict[str, Any] = field(default_factory=dict)  # Extra keyword arguments for the trainer

    project_name: str = "dependent"
    entity_name: Optional[str] = None
    group_name: Optional[str] = None

    checkpoint_dir: str = "ckpts"
    save_best: bool = True
    save_optimizer: bool = True

    tracker: Optional[str] = "wandb"
    logging_dir: Optional[str] = None
   
    seed: int = 1000

    minibatch_size: Optional[int] = None

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)



class DPConfig(BaseConfig):
    algorithm: AlgorithmConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    train: TrainConfig

    def load_file(cls, fp: str):
        """
        Load file as DPConfig
        
        :param fp: Path to file
        : type fp: str
        """
        if fp.split('.')[-1] == 'json':
            return cls.load_json(fp)
        elif fp.split('.')[-1] == 'yaml' or 'yml':
            return cls.load_yaml(fp)

    def load_json(cls, json_fp: str):
        """
        Load json file as DPConfig
        
        :param json_fp: Path to json file
        : type json_fp: str
        """
        with open(json_fp, mode='r') as file:
            config = json.safe_load(file)
        return cls.from_dict(config)


    def load_yaml(cls, yml_fp: str):
        """
        Load yaml file as DPConfig.

        :param yml_fp: Path to yaml file
        :type yml_fp: str
        """
        with open(yml_fp, mode="r") as file:
            config = yaml.safe_load(file)
        return cls.from_dict(config)

    def to_dict(self):
        """
        Convert DPConfig to dictionary.
        """
        config = {
            "algorithm": self.algorithm.__dict__,
            "model": self.model.__dict__,
            "optimizer": self.optimizer.__dict__,
            "train": self.train.__dict__,
        }
        return config
    
 
    def from_dict(cls, config: Dict):
        """
        Convert dictionary to DPConfig.
        """
        return cls(
            algorithm = AlgorithmConfig.from_dict(config.get("algorithm")),
            model = ModelConfig.from_dict(config["model"]),
            optimizer = OptimizerConfig.from_dict(config["optimizer"]),
            train = TrainConfig.from_dict(config["train"])
        )

    @classmethod
    def update(cls, baseconfig: Dict, config: Dict):
        update = {}
        # unflatten a string variable name into a nested dictionary
        # key1.key2.key3: value -> {key1: {key2: {key3: value}}}
        for name, value in config.items():
            if isinstance(value, dict):
                update[name] = value
            else:
                *layers, var = name.split(".")
                if layers:
                    d = update.setdefault(layers[0], {})
                    for layer in layers[1:]:
                        d = d.setdefault(layer, {})
                    d[var] = value

        if not isinstance(baseconfig, Dict):
            baseconfig = baseconfig.to_dict()

        updates = set()
        merged = merge(baseconfig, update, updates)

        for param in update:
            if param not in updates:
                raise ValueError(f"parameter {param} is not present in the config (typo or a wrong config)")

        return cls.from_dict(merged)

    def __str__(self):
        """Returns a human-readable string representation of the config."""
        import json

        return json.dumps(self.to_dict(), indent=4)