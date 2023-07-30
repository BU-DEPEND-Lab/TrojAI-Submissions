 
import sys
import os
from time import perf_counter
from contextlib import contextmanager

from abc import ABC, abstractmethod      
from typing import Any, Callable, Dict, Iterable, Optional, List
from pydantic import BaseModel, PrivateAttr, field

from torch.utils.data import Dataset  
from depend.utils.registers import register
from depend.core.loggers import Logger

from .learner import Registered_Learners, Base_Learner

import torch


@register
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

            summaryer: Optional[str] = "wandb",
            summaryer_kwargs: Dict[str, Any] = {},
            logging_dir: Optional[str] = None,
        
            seed: int = 1000,

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
                  summaryer,
                  summaryer_kwargs,
                  logging_dir,
                  seed
            )
            self.learner_kwargs = learner_kwargs
 

    def train(self,
        logger: Logger,
        dataset: Dataset, 
        loss: Callable,
        optimize: Callable
    )-> Dict[str, float]:
        
        dataloader =  torch.utils.data.DataLoader(
             dataset, 
             train=True, 
             batch_size = self.batch_size, 
             seed = self.seed, 
             shuffle = True
             )
         
        for episode in range(1, self.episodes + 1):
            summary_info = None
            for i, data in enumerate(dataloader):
                inputs, labels = data
                optimize.zero_grad()
                loss, loss_info = loss(inputs, labels)
                loss.backward()
                optimize.step()
                if summary_info is None:
                     summary_info = {f'train_{k}': [v] for k, v in loss_info.items()}
                else:
                     for k, v in loss_info.items():
                         summary_info[f'train_{k}'].append(v)
                     
                if i % self.checkpoint_interval == self.checkpoint_interval - 1:
                    summary_info = {\
                        k: sum(summary_info[k]) / self.checkpoint_interval for k, v in summary_info
                        }
                    self.summary(episode, 'train', **summary_info)
                    logger.info(f"Episode {episode} | Batch {i} | \
                                {' | '.join([f'{k} : {v}' for k, v in summary_info.items()])}"
                                )

        if self.summaryer == 'tensorboard':
            self.writer.flush()
        return summary_info
  
    def evaluate(
        self,
        logger: Logger,
        dataset: Dataset,
        metrics: Callable,
        **kwargs
    )-> Dict[str, float]: 
        dataloader =  torch.utils.data.DataLoader(
             dataset, 
             train=False, 
             batch_size = self.minibatch_size
             ) 
        summary_info = {}
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                inputs, labels = data
                eval_info = metrics(inputs, labels)
                if summary_info is None:
                    summary_info = {k: [v] for k, v in eval_info.items()}
                else:
                    for k, v in eval_info.items():
                        summary_info[k].append(v)

            summary_info = {k: sum(v) / len(v) for k, v in summary_info}
            self.summary(
                kwargs.get('step', 0), 
                kwargs.get('prefix', 'evaluation'), 
                **summary_info
                )
            logger.info(f"{kwargs.get('prefix', 'Evaluation')}: \
                        {' | '.join([f'{k} : {v}' for k, v in summary_info.items()])}"
                        )
        if self.summaryer == 'tensorboard':
            self.writer.flush()
        return summary_info