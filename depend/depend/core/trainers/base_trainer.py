
import sys
import os
from time import perf_counter
from contextlib import contextmanager

from abc import ABC, abstractmethod      
from typing import Any, Callable, Dict, Iterable, Optional
from pydantic import BaseModel, PrivateAttr, field

import mlflow
import mlflow.pytorch

from depend.utils.configs import TrainConfig
from depend.logger import Logger

import wandb


_TRAINERS: Dict[str, Any] = {} # registry

@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start


def register_trainer(name):
    """Decorator used to register a trainer
    Args:
        name: Name of the trainer type to register
    """

    def register_class(cls, name):
        _TRAINERS[name] = cls
        setattr(sys.modules[__name__], name, cls)
        return cls

    if isinstance(name, str):
        name = name.lower()
        return lambda c: register_class(c, name)

    cls = name
    name = cls.__name__
    register_class(cls, name.lower())

    return cls

class BaseTrainer(ABC):
    """
    Implements the functions to run a generic algorithm.

    """
    def __init__(
            self,
            episodes: int,
            batch_size: int,

            checkpoint_interval: int,
            eval_interval: int,

            pipeline: str,  # One of the pipelines in framework.pipeline
            trainer: str,  # One of the trainers
            project_name: str = 'DEPEND',
            entity_name: Optional[str] = None,
            group_name: Optional[str] = None,

            checkpoint_dir: str = "ckpts",
            save_best: bool = True,
            save_optimizer: bool = True,

            tracker: Optional[str] = "wandb",
            tracker_kwargs: Dict[str, Any] = {},
            logging_dir: Optional[str] = None,
        
            seed: int = 1000,

            minibatch_size: Optional[int] = None,
            trainer_kwargs: Dict[str, Any] = field(default_factory=dict),  # Extra keyword arguments for the train
            ):
        
        
        self.episodes = episodes
        self.batch_size = batch_size 

        self.checkpoint_interval = checkpoint_interval
        self.eval_interval = eval_interval

        self.trainer_kwargs = trainer_kwargs 
        self.project_name = project_name 
        self.entity_name = entity_name 
        self.group_name = group_name 

        self.checkpoint_dir = checkpoint_dir 
        self.save_best = save_best 
        self.save_optimizer = save_optimizer 

        self.seed = seed
        self.minibatch_size = minibatch_size

        self.tracker = tracker
        self.tracker_kwargs = tracker_kwargs
        self.logging_dir = logging_dir
        self.logger = Logger(results_dir=self.logging_dir, log_name="training_log", seed=self.seed, append=True)

    @abstractmethod
    def prepare_model(self): ...

    @abstractmethod
    def prepare_dataloaders(self): ...

    @abstractmethod
    def prepare_optimizer(self): ...

    @abstractmethod
    def train(self)-> Dict[str, float]: ... 

    @abstractmethod
    def evaluate(self)-> Dict[str, float]: ...

    def prepare_tracker(self): 
        if self.tracker == 'wandb':
            wandb.init(project=self.project_name, entity=self.entity_name,
                sync_tensorboard=True, reinit=True, config=self.tracker_kwargs)
        



 