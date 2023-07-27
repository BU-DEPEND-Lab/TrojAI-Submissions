 
import sys
import os
from time import perf_counter
from contextlib import contextmanager

from abc import ABC, abstractmethod      
from typing import Any, Callable, Dict, Iterable, Optional
from pydantic import BaseModel, PrivateAttr, field
 
from depend.utils.data_loader import DataSplit
from depend.core.loggers import Logger

from .learner import register_learner, Base_Learner

import torch


@register_learner
class Torch_Learner(Base_Learner):
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
            super(Torch_Learner, self).__init__(
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
 

    def train(self,
        logger: Logger,
        train_set: DataSplit, 
        validation_set: DataSplit,
        loss: Callable,
        metric: Callable,
        optimize: Callable
    )-> Dict[str, float]:
        
        train_loader =  torch.utils.data.DataLoader(
             train_set, 
             train=True, 
             batch_size = self.minibatch_size, 
             seed = self.seed, 
             shuffle = True
             )
        validation_loader =  torch.utils.data.DataLoader(
             validation_set, 
             train=False, 
             batch_size = self.minibatch_size 
             )
        for step in range(1, self.episodes + 1):
            track_info = None
            for i, data in enumerate(train_loader):
                inputs, labels = data
                optimize.zero_grad()
                loss, loss_info = loss(inputs, labels)
                loss.backward()
                optimize.step()
                if track_info is None:
                     track_info = {f'train_{k}': [v] for k, v in loss_info.items()}
                else:
                     for k, v in loss_info.items():
                         track_info[f'train_{k}'].append(v)
                     
                if i % self.checkpoint_interval == self.checkpoint_interval - 1:
                    track_info = {k: sum(track_info[k]) / self.checkpoint_interval for k, v in track_info}
                    self.track(step, **track_info)
                    logger.info(f"Ep {step} | Batch {i} | {' | '.join([f'{k} : {v}' for k, v in track_info.items()])}")

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                track_info = {}
                for i, data in enumerate(validation_loader):
                    inputs, labels = data
                    eval_info = metric(inputs, labels)
                    if track_info is None:
                        track_info = {f'validation_{k}': [v] for k, v in eval_info.items()}
                    else:
                        for k, v in eval_info.items():
                            track_info[f'validation_{k}'].append(v)

                track_info = {k: sum(track_info[k]) / len(track_info[k]) for k, v in track_info}
                self.track(step, **track_info)
                logger.info(f"Ep {step} | {' | '.join([f'{k} : {v}' for k, v in track_info.items()])}")
    
        if self.tracker == 'tensorboard':
            self.writer.flush()

    def evaluate(
        self,
        logger: Logger,
        evaluation_set: DataSplit,
        metric: Callable,
        **kwargs
    )-> Dict[str, float]: 
        evaluation_loader =  torch.utils.data.DataLoader(
             evaluation_set, 
             train=False, 
             batch_size = self.minibatch_size
             ) 
        with torch.no_grad():
            track_info = {}
            for i, data in enumerate(evaluation_loader):
                inputs, labels = data
                eval_info = metric(inputs, labels)
                if track_info is None:
                    track_info = {f'validation_{k}': [v] for k, v in eval_info.items()}
                else:
                    for k, v in eval_info.items():
                        track_info[f'validation_{k}'].append(v)

            track_info = {k: sum(track_info[k]) / len(track_info[k]) for k, v in track_info}
            self.track(0, **track_info)
            logger.info(f"Evaluation: {' | '.join([f'{k} : {v}' for k, v in track_info.items()])}")
        if self.tracker == 'tensorboard':
            self.writer.flush()