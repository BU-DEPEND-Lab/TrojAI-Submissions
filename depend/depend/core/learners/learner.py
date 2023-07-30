
import sys
import os
from time import perf_counter
from contextlib import contextmanager
from functools import partial


from abc import ABC, abstractmethod      
from typing import Any, Callable, Dict, ClassVar, Iterable, Optional, Union
from pydantic import BaseModel, PrivateAttr, field, validate_call
 

from depend.utils.configs import LearnerConfig
from depend.utils.registers import register_class
from depend.core.loggers import Logger

import wandb
from torch.utils.tensorboard import SummaryWriter



Registered_Learners: Dict[str, Any] = {} # registry

@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start



class Base_Learner(BaseModel, ABC):
    """
    Implements the functions to run a generic algorithm.

    """
    __registry__: ClassVar[Dict[str, Any]]
    episodes: int
    batch_size: int

    checkpoint_interval: int
    eval_interval: int

    project_name: str = 'DEPEND'
    entity_name: Optional[str] = None
    group_name: Optional[str] = None

    checkpoint_dir: str = "ckpts"
    save_best: bool = True
    save_optimizer: bool = True

    tracker: Optional[str] = "wandb"
    tracker_kwargs: Dict[str, Any] = {}
    logging_dir: Optional[str] = None

    seed: int = 1000

    minibatch_size: Optional[int] = None

    @classmethod
    def register(cls, name):
        cls.__registry__[name] = cls

    @classmethod
    @property
    def registered_learners(cls):
         return cls.__registry__
 
    def __post__init__(self):
        if self.tracker == 'wandb':
            wandb.init(
                project=self.project_name, 
                group=self.group_name,
                entity=self.entity_name,
                sync_tensorboard=True, 
                reinit=True, 
                config=self.tracker_kwargs
            )
            self.track = lambda step, prefix, **info: wandb.log(info, step)
        elif self.tracker == 'tensorboard':
            self.writer = SummaryWriter(
                log_dir=self.logging_dir
                )
            
    def summary(self, step, prefix, **info):
        if self.tracker == 'wandb':
            wandb.log(info, step)
        elif self.tracker == 'tensorboard':
            self.writer.add_scalar(prefix, info, step)


    @abstractmethod
    def train(self,
        logger: Logger,
        data_loader: DataLoader,
        loss: Callable,
        optimize: Callable
        
    )-> Dict[str, float]: ...
 

    @abstractmethod
    def evaluate(
        self,
        logger: Logger,
        data_loader: DataLoader,
        eval_fn: Callable,
        **kwargs
    )-> Dict[str, float]: ...
 