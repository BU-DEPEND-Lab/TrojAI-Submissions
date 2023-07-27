
import sys
import os
from time import perf_counter
from contextlib import contextmanager

from abc import ABC, abstractmethod      
from typing import Any, Callable, Dict, Iterable, Optional
from pydantic import BaseModel, PrivateAttr, field
 

from depend.utils.configs import LearnerConfig
from depend.utils.data_loader import DataLoader
from depend.core.loggers import Logger

import wandb
from torch.utils.tensorboard import SummaryWriter



_learnerS: Dict[str, Any] = {} # registry

@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start


def register_learner(name):
    """Decorator used to register a learner
    Args:
        name: Name of the learner type to register
    """

    def register_class(cls, name):
        _learnerS[name] = cls
        setattr(sys.modules[__name__], name, cls)
        return cls

    if isinstance(name, str):
        name = name.lower()
        return lambda c: register_class(c, name)

    cls = name
    name = cls.__name__
    register_class(cls, name.lower())

    return cls

class BaseLearner(ABC):
    """
    Implements the functions to run a generic algorithm.

    """
    def __init__(
            self,
            episodes: int,
            batch_size: int,

            checkpoint_interval: int,
            eval_interval: int,

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
            ):
        
        
        self.episodes = episodes
        self.batch_size = batch_size 

        self.checkpoint_interval = checkpoint_interval
        self.eval_interval = eval_interval

 
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

    def prepare_tracker(self): 
        if self.tracker == 'wandb':
            self.tracker = wandb.init(
                project=self.project_name, 
                group=self.group_name,
                entity=self.entity_name,
                sync_tensorboard=True, 
                reinit=True, 
                config=self.tracker_kwargs
                )
            self.log_fn = lambda epoch, **kwargs: wandb.log(**kwargs)

        elif self.tracker == 'tensorboard':
            summary_writer = SummaryWriter(
                log_dir=self.logging_dir
                )
            self.log_fn = lambda epoch, **kwargs: [summary_writer.add_scalar(k, v, epoch) for k,v in kwargs.items()]
        return self.log_fn
 
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
 