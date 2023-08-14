
import sys
import os
from time import perf_counter
from contextlib import contextmanager
from functools import partial


from abc import ABC, abstractmethod      
from typing import Any, Callable, Literal, Dict, ClassVar, Iterable, Optional, Union, Generator
from pydantic import BaseModel, PrivateAttr, field, validate_call
 

from depend.utils.configs import LearnerConfig
from depend.utils.registers import register_class
from depend.core.loggers import Logger

import wandb

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import logging
logger = logging.getLogger(__name__)

 

@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start



class Base_Learner(BaseModel, ABC):
    """
    Implements the functions to run a generic algorithm.

    """
    __registry__: ClassVar[Dict[str, Any]]

    episodes: int = 2
    batch_size: int = 32
 
    checkpoint_interval: int = 1
    eval_interval: int = 1

    project_name: str = 'DEPEND'
    entity_name: Optional[str] = None
    group_name: Optional[str] = None

    checkpoint_dir: str = "ckpts"
    save_best: bool = True
    save_optimizer: bool = True

    tracker: Optional[Literal['wandb', 'tensorboard', 'comet']] = None #"wandb"
    tracker_kwargs: Dict[str, Any] = {}
    logging_dir: Optional[str] = None

    seed: int = 1000
 
    @classmethod
    def register(cls, name):
        cls.__registry__[name] = cls

    @classmethod
    @property
    def registered_learners(cls):
         return cls.__registry__
    
    @classmethod
    def configure(cls, config: LearnerConfig):
        kwargs = {
            'episodes': config.episodes, 
            'batch_size': config.batch_size,
            'checkpoint_interval': config.checkpoint_interval,
            'eval_interval': config.eval_interval,
            'project_name': config.project_name,
            'entity_name': config.entity_name,
            'group_name': config.group_name,
            'checkpoint_dir': config.checkpoint_dir,
            'save_best': config.save_best,
            'save_optimizer': config.save_optimizer,
            'tracker': config.tracker,
            'tracker_kwargs': config.tracker_kwargs,
            'logging_dir': config.logging_dir,
            'seed': config.seed,
        }
        return cls(**kwargs) 
 
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
        elif self.tracker == 'tensorboard':
            self.writer = SummaryWriter(
                log_dir=self.logging_dir
                )
        elif self.tracker is not None:
            raise NotImplementedError
            
    def summary(self, step, prefix, **info):
        if self.tracker == 'wandb':
            wandb.log(info, step)
        elif self.tracker == 'tensorboard':
            self.writer.add_scalar(prefix, info, step)
        elif self.tracker is not None:
            raise NotImplementedError
        else:
            logger.info(f"Step {step}: {prefix} info: {info}")
    
    @classmethod
    def track(cls, func: Generator[Dict[Any]]):
        def wrapper(
                obj: Base_Learner, 
                *args, 
                **kwargs):
            summary_gen = obj.func(*args, **kwargs)
            for episode in range(obj.episode):
                summary_info = next(summary_gen)
                obj.summary(episode, 'train', **{k: sum(v)/len(v) for k, v in summary_info})
            if obj.tracker == 'tensorboard':
                obj.writer.flush()
        return wrapper
    

    @abstractmethod
    def train(self,
        logger: Logger,
        dataset: Dataset, 
        loss: Callable,
        optimize: Callable
        
    )-> Dict[str, float]: ...
 
    @abstractmethod
    def train_iterator(self,
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
        eval_fn: Callable 
    )-> Dict[str, float]: ...
 