 
from dataclasses import fields
from pydantic import BaseModel
from typing import Any, Dict, List, ClassVar, Callable, Iterable, Union, cast, get_type_hints
import os

from depend.utils.configs import DPConfig
from depend.utils.models import load_models_dirpath
from depend.depend.core.serializable.utils import serialize_with_pyarrow
from depend.depend.core.serializable import Model_Indexer

from abc import ABC, abstractmethod    

import torch.optim as optim
import torch.nn as nn
 
import pandas as pd

import logging
logger = logging.getLogger(__name__)


class Dependent(ABC, BaseModel):
    __registry__: ClassVar[Dict[str, Any]] = {}
    
    target_model_indexer: Model_Indexer = ...
    target_model_table: pd.DataFrame = None
    clean_example_dict: Dict[str, Dict[str, Any]] = ...
    poisoned_example_dict: Dict[str, Dict[str, Any]] = ...
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            pd.DataFrame: lambda v: serialize_with_pyarrow(v)
        }

         
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
        logger.info(f"Loaded target_model_dict {data_infos[0].keys()}")
        logger.info(f"Loaded target_model_repr_dict {data_infos[1].keys()}")
        logger.info(f"Loaded model_ground_truth_dict {data_infos[2]}")
        logger.info(f"Loaded clean models {data_infos[3]}")
        logger.info(f"Loaded poisoned models {data_infos[4]}")

        model_ground_truth_dict = data_infos[2]

        # Convert the model_ground_truth dictionary to a DataFrame
        # Allocate a row for each element in the lists
        df = pd.DataFrame([(key, index, value) for key, values in model_ground_truth_dict.items() for index, value in enumerate(values)],
                  columns=['model_class', 'idx_in_class', 'poisoned'])
        logging.info(df[df['model_class']=='SimplifiedRLStarter']['poisoned'] == 0)
        #df = pa.Table.from_pandas(df)
       
        df = df.dropna()
        df['model_class'] = df['model_class'].astype(str)
        df['idx_in_class'] = df['idx_in_class'].astype(int)
        df['poisoned'] = df['poisoned'].astype(int)

        model_indexer = Model_Indexer(
            model_dict = data_infos[0], 
            model_repr_dict = data_infos[1], 
            )
        
        return cls(
            target_model_table = df,
            target_model_indexer = model_indexer,
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