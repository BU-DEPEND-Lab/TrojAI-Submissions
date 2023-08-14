 
import sys
import os
from time import perf_counter
from contextlib import contextmanager

from abc import ABC, abstractmethod      
from typing import Any, Callable, Dict, Iterable, Optional, List, Tuple
   
from depend.core.loggers import Logger
from depend.core.learners.base import Base_Learner
 
from depend.utils.registers import register
from depend.utils.configs import LearnerConfig


import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


@register
class Torch_Learner(Base_Learner):
    """
    Implements the functions to run a generic algorithm.

    """

    class Config:
        allow_extra = True

    def train(
        self,
        logger: Logger,
        train_set: Dataset,  
        loss: Callable[[Iterable[torch.Tensor], Iterable[torch.Tensor]], Tuple[torch.Tensor, Dict[Any]]], 
        optimize: torch.optim 
    )-> Dict[str, float]:
        
        train_loader =  torch.utils.data.DataLoader(
             train_set, 
             train=True, 
             batch_size = self.batch_size, 
             seed = self.seed, 
             shuffle = True
             )
        
        summary_info = {}
        for episode in range(1, self.episodes + 1):
            info = self.train_iterator(logger, train_loader, loss, optimize)
            summary_info.update(info)
        return summary_info

    @Base_Learner.track 
    def train_iterator(
        self,
        logger: Logger,
        train_loader: DataLoader, 
        loss: Callable[[Iterable[torch.Tensor], Iterable[torch.Tensor]], Tuple[torch.Tensor, Dict[Any]]],
        optimize: torch.optim 
        ):
        for episode in range(1, self.episodes + 1):
            summary_info = None
            for i, data in enumerate(train_loader):
                inputs, labels = data
                optimize.zero_grad()
                loss, loss_info = loss(inputs, labels)
                loss.backward()
                optimize.step()
                if summary_info is None:
                    summary_info = {f'{k}': [v] for k, v in loss_info.items()}
                else:
                    summary_info = {f'{k}': summary_info[k] + [v] for k, v in loss_info.items()}
                        
                if i % self.checkpoint_interval == self.checkpoint_interval - 1:
                    logger.info(f"Batch {i} | \
                                {' | '.join([f'{k} : {sum(v)/len(v)}' for k, v in summary_info.items()])}"
                                )
            logger.info(f"Episode {episode} | Train: \
                                {' | '.join([f'{k} : {sum(v)/len(v)}' for k, v in summary_info.items()])}"
                                )
          
            yield summary_info
            
 
    def evaluate(
        self,
        logger: Logger,
        dataset: Dataset,
        metrics: Callable
    )-> Dict[str, float]: 
        dataloader =  torch.utils.data.DataLoader(
             dataset, 
             train=False, 
             batch_size = self.batch_size
             ) 
        summary_info = {}
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                inputs, labels = data
                eval_info = metrics(inputs, labels)
                if summary_info is None:
                    summary_info = {f'{k}': [v] for k, v in eval_info.items()}
                else:
                    summary_info = {f'{k}': summary_info[k] + [v] for k, v in eval_info.items()}

            summary_info = {k: sum(v) / len(v) for k, v in summary_info}
            
            logger.info(f"Evaluation: \
                        {' | '.join([f'{k} : {v}' for k, v in summary_info.items()])}"
                        )
        return summary_info