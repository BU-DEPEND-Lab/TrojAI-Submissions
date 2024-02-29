 
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
        final_train = False
    )-> Dict[str, float]:
        
        train_loader =  torch.utils.data.DataLoader(
             train_set, 
             batch_size = self.batch_size, 
             shuffle = True
             )
        test_loader = None
        if test_set is not None:
            test_loader =  torch.utils.data.DataLoader(
                 test_set,
                 batch_size=len(test_set)
                 ) 
        
        summary_info = {}
        self.episodes = self.xval_episodes if not final_train else self.final_episodes

        train_info_gen = self.train_iterator(
            learner_logger, 
            train_loader, 
            loss_function, 
            optimize_function, 
            metric_function, 
            test_loader)
        
        
        for train_info in train_info_gen:
            summary_info.update(**train_info)
            if train_info['auroc'][-1] > 0.88:
                break
            
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
        metric_function: Optional[Callable] = None,
        test_loader: Optional[DataLoader] = None, 
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
                for i, data in enumerate(test_loader):
                    eval_info = metric_function(data)
                    for k, v in eval_info.items():
                        if k not in summary_info:
                            summary_info.update({f'{k}': [v]})
                        else:
                            summary_info.updaet({f'{k}': summary_info[k] + [v]})                
             
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