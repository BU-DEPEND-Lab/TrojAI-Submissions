 
import sys
import os
from time import perf_counter
from contextlib import contextmanager

from abc import ABC, abstractmethod      
from typing import Any, Callable, Dict, Iterable, Optional
from pydantic import BaseModel, PrivateAttr, field
 
from depend.utils.data_loader import DataLoader
from depend.core.loggers import Logger

from .learner import register_learner, BaseLearner
 

@register_learner
class SupervisedLearner(BaseLearner):
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
            learner_kwargs: Dict[str, Any] = field(default_factory=dict),  # Extra keyword arguments for the train
            ):
            super(SupervisedLearner, self).__init__(
                  episodes,
                  batch_size,
                  checkpoint_interval,
                  eval_interval,
                  project_name,
                  entity_name,
                  group_name,
                  checkpoint_dir,
                  save_best,
                  save_optimizer,
                  tracker,
                  tracker_kwargs,
                  logging_dir,
                  seed,
                  minibatch_size
            )
            self.learner_kwargs = learner_kwargs
 
    @abstractmethod
    def train(self,
        logger: Logger,
        data_loader: DataLoader,
        loss: Callable,
        optimize: Callable
    )-> Dict[str, float]:
        train_info = []
        for step in range(1, self.episodes + 1):
            loss = loss(step)
            info = optimize(loss)
            self.log_fn(step, **info)
            logger.info(f'Step {step}: Training {info}')
            train_info.append(info)

        return train_info

    @abstractmethod
    def evaluate(
        self,
        logger: Logger,
        data_loader: DataLoader,
        eval_fn: Callable,
        **kwargs
    )-> Dict[str, float]: ...
        eval_info = []
        for step in range(1, self.episodes + 1):
            metrics = eval_fn(step)
            logger.info(f'Step {step}: Evaluation {metrics}')
            eval_info.append(metrics)
        return eval_info