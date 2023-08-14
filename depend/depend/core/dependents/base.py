 
from dataclasses import fields
from typing import Any, Dict, List, ClassVar, Callable, Literal, TypedDict, Union, cast, get_type_hints
import os

from depend.utils.configs import DPConfig
from depend.utils.models import load_models_dirpath 
from depend.core.serializable import Serializable

from abc import ABC, abstractmethod    

import torch.optim as optim
import torch.nn as nn

import pyarrow as pa
import pandas as pd

import logging
logger = logging.getLogger(__name__)


class Dependent(ABC):
    __registry__: ClassVar[Dict[str, Any]] = {}
    model_dict: Dict[str, List[nn.Module]] = ...
    model_repr_dict: Dict[str, Dict[Any, Any]] = ...
    model_table: pa.Table = ...
    clean_example_dict: Dict[str, Dict[str, Any]] = ...
    poisoned_example_dict: Dict[str, Dict[str, Any]] = ...
    

    @classmethod
    def register(cls, name):
        cls.__registry__[name] = cls

    @classmethod
    @property
    def registered_dependents(cls):
         return cls.__registry__
    
    @classmethod
    def get_assets(cls, model_path_list: List[str]): 
        data_infos = load_models_dirpath(model_path_list)
        logger.info(f"Loaded model_dict {data_infos[0].keys()}")
        logger.info(f"Loaded model_repr_dict {data_infos[1].keys()}")
        logger.info(f"Loaded model_ground_truth_dict {data_infos[2]}")
        logger.info(f"Loaded clean models {data_infos[3]}")
        logger.info(f"Loaded poisoned models {data_infos[4]}")

        model_ground_truth_dict = data_infos[2]
        # Convert the model_ground_truth dictionary to a DataFrame
        # Allocate a row for each element in the lists
        df = pd.DataFrame([(key, index, value) for key, values in model_ground_truth_dict.items() for index, value in enumerate(values)],
                  columns=['model_class', 'idx_in_class', 'poisoned'])
        return cls(
            model_dict = data_infos[0],
            model_repr_dict = data_infos[1],
            model_table = pa.Table.from_pandas(df),
            clean_example_dict = data_infos[3],
            poisoned_example_dict = data_infos[4]
        )
    
    
    @abstractmethod
    def train_detector(self):
        raise NotImplementedError 
    
    @abstractmethod
    def evaluate_detector(self):
        raise NotImplementedError 
    
    @abstractmethod
    def run_detector(self, taget_path: str) -> float:
        raise NotImplementedError 
    
    @abstractmethod
    def configure(self,
            epochs = 1,
            config: DPConfig = ...,
            experiment_name: str = ...,
            result_dir: str = ...
            ):
        raise NotImplementedError

  
    @abstractmethod
    def get_loss(self):
        raise NotImplementedError
 
    @abstractmethod
    def get_metrics(self):
        raise NotImplementedError