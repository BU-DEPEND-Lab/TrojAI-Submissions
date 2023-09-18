 
import sys
import os
from time import perf_counter
from contextlib import contextmanager

from pydantic import Extra

from abc import ABC, abstractmethod      
from typing import Any, Callable, Dict, Iterable, Optional, List, Tuple
   
from depend.core.logger import Logger
from depend.core.learner.base import Base_Learner
 
from depend.utils.registers import register
from depend.utils.configs import LearnerConfig


import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from tqdm import tqdm

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
        loss_function: Callable[\
            [Iterable[torch.Tensor], 
             Iterable[torch.Tensor]
             ], Tuple[torch.Tensor, Dict[Any, Any]]], 
        optimize_function: Callable,
        test_set: Optional[Dataset] = None,
        metric_function: Optional[Callable] = None,
    )-> Dict[str, float]:
        
        train_loader =  torch.utils.data.DataLoader(
             train_set, 
             batch_size = self.batch_size, 
             shuffle = True
             )
         
        summary_info = {}
        train_info_gen = self.train_iterator(
            learner_logger, 
            train_loader, 
            loss_function, 
            optimize_function, 
            test_set, 
            metric_function)
        
        for episode in range(1, self.episodes + 1):
            train_info = next(train_info_gen)
            summary_info.update(**train_info)
            
        return summary_info

    @Base_Learner.track 
    def train_iterator(
        self, 
        learner_logger: Logger,
        train_loader: DataLoader, 
        loss_function: Callable[\
            [Iterable[torch.Tensor], 
             Iterable[torch.Tensor]
             ], Tuple[torch.Tensor, Dict[Any, Any]]],
        optimizer: Callable, 
        test_set: Optional[Dataset] = None,
        metric_function: Optional[Callable] = None,
        ):
        
 
        for episode in tqdm(range(1, self.episodes + 1), desc ="Learning Iteration: "):
            summary_info = None
            for i, data in enumerate(train_loader):
                #logger.info(f'Get data {data} from train_loader')
                optimizer.zero_grad()

                loss, loss_info = loss_function(data)
                #logger.info(f"Get loss {loss}")

                loss.backward()
                #logger.info("????")
                #nn.utils.clip_grad_norm_(self.mask.parameters(), 1.0)
                optimizer.step()

                #optimizer(loss)
                #logger.info(f"One step optimization finished")
                if summary_info is None:
                    summary_info = {f'{k}': [v] for k, v in loss_info.items()}
                else:
                    summary_info = {f'{k}': summary_info[k] + [v] for k, v in loss_info.items()}
                
                #if i % self.checkpoint_interval == self.checkpoint_interval - 1:
                #    learner_logger.info(f"Batch {i} | " + \
                #                        ' | '.join([f'{k} : {sum(v)/len(v)}' for k, v in summary_info.items()]))
                #logger.info(f"One batch training finished")
                #logger.info(f"Summary info: {summary_info}")
                
            

            if metric_function is not None:
                self.evaluate(self.logger, )
             
            learner_logger.info(f"Episode {episode} | Train: " + \
                                ' | '.join([f'{k} : {sum(v)/len(v)}' for k, v in summary_info.items()]))
            
            logger.info(f"Episode {episode} | Train: " + \
                                ' | '.join([f'{k} : {sum(v)/len(v)}' for k, v in summary_info.items()]))
                
            yield summary_info
            
            
 
    def evaluate(
        self,
        learner_logger: Logger,
        dataset: Dataset,
        metric_function: Callable
    )-> Dict[str, float]: 
        dataloader =  torch.utils.data.DataLoader(
             dataset,
             batch_size=len(dataset)
             ) 
        summary_info = None
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                eval_info = metric_function(data)
                #logger.info(f"Evaluation info {eval_info}")
                if summary_info is None:
                    summary_info = {k: [v] for k, v in eval_info.items()}
                else:
                    summary_info = {k: summary_info[k] + [v] for k, v in eval_info.items()}

            summary_info = {k: sum(v) / len(v) for k, v in summary_info.items()}
            
            learner_logger.info(f"Evaluation: " + \
                        ' | '.join([f'{k} : {v}' for k, v in summary_info.items()])
                        )
            logger.info(f"Evaluation: " + \
                        ' | '.join([f'{k} : {v}' for k, v in summary_info.items()])
                        )
        return summary_info