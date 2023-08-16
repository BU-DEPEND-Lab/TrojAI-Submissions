from copy import deepcopy
from dataclasses import field
from typing import Any, Dict, List, Optional, Set, Union, Literal, ClassVar
from pydantic import BaseModel, Field, ConfigDict,  Extra, model_validator
from dataclasses import dataclass

import yaml
import json
from abc import ABC, abstractmethod

import torch
import numpy

import logging
logger = logging.getLogger(__name__)

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
    
    def to_dict(self):
        return self.__dict__
   
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
    
      
    

class DataConfig(BaseConfig):
    """
    Config for an data.

    :param dataset_paths: dataset paths
    :type dataset_paths: Dict[str, str]

    :param num_workers: Number of works for data preprocessing
    :type name: str
    
    :param overwrite_cache: whether overwrite cach during data preprocessing
    :type overwrite_cache: bool

    :param columns: list of column names 
    :type columns: List[str]

    :param type: type of data ['torch', 'tf', ...]
    :type type: str

    :param num_split : number of splits
    :type num_split : int

    :param kwargs: Keyword arguments for the optimizer (e.g. lr, betas, eps, weight_decay)
    :type kwargs: Dict[str, Any]

    
    """
    num_workers: str = 1
    num_splits: int = 3
    max_models: int = 20
    num_frames_per_model: int = 20
 
    class Config:
        extra = Extra.forbid

class AlgorithmConfig(BaseConfig):
    """
    Config for an optimizer.

    :param name: Name of the optimizer
    :type name: str

    :param kwargs: Keyword arguments for the optimizer (e.g. lr, betas, eps, weight_decay)
    :type kwargs: Dict[str, Any]
    """
     

    task: Literal['RL', 'ImageClassification', 'ImageSegmentation', 'ObjectDetection', 'NLPs'] = 'RL'
    metrics: List[str] = ['auroc']
    device: Optional[str] = 'cpu'

    class Config:
        extra = Extra.allow

    @model_validator(mode='after')
    def on_create(cls, values):
        if values.device is None or values.device == 'cuda':
            values.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            values.device = torch.device("cpu")
 
class BaseModelConfig(BaseConfig):
    """
    Config for an Model

    :param name: Name of the optimizer
    :type name: str

    :param kwargs: Keyword arguments for the optimizer (e.g. lr, betas, eps, weight_decay)
    :type kwargs: Dict[str, Any]
    """
    
    name: str
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    load_from_file: Optional[str] = None
    
    class Config:
        extra = Extra.allow
    
class ModelConfig(BaseConfig):
    """
    Config for multiple models
    """
    
    save_dir: str = 'saved_models'
    
    class Config:
        extra = Extra.allow

    @model_validator(mode='before')
    def on_create(cls, values):
        for k, v in values.items():
            logger.info(f'{k}: {v}')
            if type(v) == dict:
                values[k] = BaseModelConfig(**v)
        return values
        

 

class OptimizerConfig(BaseConfig):
    """
    Config for an optimizer.

    :param name: Name of the optimizer
    :type name: str

    :param kwargs: Keyword arguments for the optimizer (e.g. lr, betas, eps, weight_decay)
    :type kwargs: Dict[str, Any]
    """ 
    optimizer_class: str = 'RAdam'
    kwargs: Dict[str, Any] = field(default_factory=dict)
    
    class Config:
        extra = Extra.forbid
    
    @model_validator(mode='before')
    def on_create(cls, values):
        kwargs = {}
        optimizer_class = None
        for k,v in values.items():
            if k != 'optimizer_class':
                kwargs[k] = v
            else:
                optimizer_class = v
        if optimizer_class is not None:
            return {'optimizer_class': optimizer_class, 'kwargs': kwargs}
        else:
            return {'kwargs': kwargs}


class LearnerConfig(BaseConfig):
    """
    Config for learn job on model.

    :param episodes: Total number of learning episodes
    :type episodes: int
 
    :param batch_size: Batch size for learning
    :type batch_size: int

    :param tracker: Tracker to use for logging. Default: "wandb"
    :type tracker: str

    :param checkpoint_interval: Save model every checkpoint_interval steps.
        Each checkpoint is stored in a sub-directory of the `LearnerConfig.checkpoint_dir`
        directory in the format `checkpoint_dir/checkpoint_{step}`.
    :type checkpoint_interval: int

    :param eval_interval: Evaluate model every eval_interval steps
    :type eval_interval: int

    :param pipeline: Pipeline to use for learning. One of the registered pipelines present in trlx.pipeline
    :type pipeline: str

    :param learner: learner to use for learning. One of the registered learners present in trlx.learner
    :type learner: str

    :param learner_kwargs: Extra keyword arguments for the learner
    :type learner: Dict[str, Any]

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
    
    :type minibatch_size: int
    """
    episodes: int
    batch_size: int

    checkpoint_interval: int
    eval_interval: int

    #pipeline: str  # One of the pipelines in framework.pipeline
    learner_kwargs: Dict[str, Any] = field(default_factory=dict)  # Extra keyword arguments for the learner

    project_name: str = "dependent"
    entity_name: Optional[str] = None
    group_name: Optional[str] = None

    checkpoint_dir: str = "ckpts"
    save_best: bool = True
    save_optimizer: bool = True

    tracker: Optional[str] = None
    logging_dir: Optional[str] = None
   
    seed: int = 1000

    minibatch_size: Optional[int] = None

    class Config:
        extra = "forbid"  # Prevent extra fields from being accepted


@dataclass
class DPConfig:
    algorithm: AlgorithmConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    learner: LearnerConfig
    data: DataConfig

    class Config:
        extra = "forbid"  # Prevent extra fields from being accepted


    
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
            "learner": self.learn.__dict__,
            "data": self.data.__dict__
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
            learner = LearnerConfig.from_dict(config["learner"]),
            data = DataConfig.from_dict(config['data'])
        )


    def __str__(self):
        """Returns a human-readable string representation of the config."""
        import json

        return json.dumps(self.to_dict(), indent=4)