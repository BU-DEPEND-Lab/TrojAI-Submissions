 
import sys
import os
from time import perf_counter
from contextlib import contextmanager

from pydantic import Extra

from abc import ABC, abstractmethod      
from typing import Any, Callable, Dict, Iterable, Optional, List, Tuple
   
from depend.core.loggers import Logger
from depend.core.learners.base import Base_Learner
 
from depend.utils.registers import register
from depend.utils.configs import LearnerConfig


import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import logging
logger = logging.getLogger(__name__)


@register
class Torch_Learner(Base_Learner):
    """
    Implements the functions to run a generic algorithm.

    """

    class Config:
        extra = Extra.allow

    def train(
        self,
        learner_logger: Logger,
        train_set: Dataset,  
        loss: Callable[\
            [Iterable[torch.Tensor], 
             Iterable[torch.Tensor]
             ], Tuple[torch.Tensor, Dict[Any, Any]]], 
        optimize: torch.optim 
    )-> Dict[str, float]:
        
        train_loader =  torch.utils.data.DataLoader(
             train_set, 
             batch_size = self.batch_size, 
             shuffle = True
             )
        
        summary_info = {}
        for episode in range(1, self.episodes + 1):
            info = self.train_iterator(learner_logger, train_loader, loss, optimize)
            summary_info.update(info)
        return summary_info

    @Base_Learner.track 
    def train_iterator(
        self,
        learner_logger: Logger,
        train_loader: DataLoader, 
        loss: Callable[\
            [Iterable[torch.Tensor], 
             Iterable[torch.Tensor]
             ], Tuple[torch.Tensor, Dict[Any, Any]]],
        optimize: torch.optim 
        ):
        for episode in range(1, self.episodes + 1):
            summary_info = None
            for i, data in enumerate(train_loader):
                #logger.info(f'Get data {data} from train_loader')
                optimize.zero_grad()
                loss, loss_info = loss(data)
                loss.backward()
                optimize.step()
                if summary_info is None:
                    summary_info = {f'{k}': [v] for k, v in loss_info.items()}
                else:
                    summary_info = {f'{k}': summary_info[k] + [v] for k, v in loss_info.items()}
                        
                if i % self.checkpoint_interval == self.checkpoint_interval - 1:
                    learner_logger.info(f"Batch {i} | \
                                {' | '.join([f'{k} : {sum(v)/len(v)}' for k, v in summary_info.items()])}"
                                )
            learner_logger.info(f"Episode {episode} | Train: \
                                {' | '.join([f'{k} : {sum(v)/len(v)}' for k, v in summary_info.items()])}"
                                )
          
            yield summary_info
            
 
    def evaluate(
        self,
        learner_logger: Logger,
        dataset: Dataset,
        metrics: Callable
    )-> Dict[str, float]: 
        dataloader =  torch.utils.data.DataLoader(
             dataset
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
            
            learner_logger.info(f"Evaluation: \
                        {' | '.join([f'{k} : {v}' for k, v in summary_info.items()])}"
                        )
        return summary_info