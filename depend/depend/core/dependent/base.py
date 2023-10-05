 
from dataclasses import fields
from pydantic import BaseModel
from typing import Any, Dict, List, ClassVar, Callable, Iterable, Union, cast, get_type_hints
import os


from depend.lib.agent import Agent, ParallelAgent
from depend.utils.env import make_env
from depend.utils.configs import DPConfig
from depend.utils.models import load_models_dirpath
from depend.depend.core.serializable.utils import serialize_with_pyarrow
from depend.depend.core.serializable import Model_Indexer

from abc import ABC, abstractmethod    

import torch.optim as optim
import torch.nn as nn
 
import pandas as pd

import pyarrow as pa
import pyarrow.compute as pc

import pickle

from datasets.arrow_dataset import Dataset



import numpy as np

import logging
logger = logging.getLogger(__name__)


class Dependent(ABC, BaseModel):
    __registry__: ClassVar[Dict[str, Any]] = {}
    
    target_model_indexer: Model_Indexer = None
    target_model_table: pd.DataFrame = None
    clean_example_dict: Dict[str, Dict[str, Any]] = None
    poisoned_example_dict: Dict[str, Dict[str, Any]] = None
    
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

    def get_optimizer(self, model):
        if self.config.optimizer.optimizer_class == 'RAdam':
            self.optimizer = optim.RAdam(model.parameters(), **self.config.optimizer.kwargs)
        elif self.config.optimizer.optimizer_class == 'Adam':
             self.optimizer = optim.Adam(model.parameters(), **self.config.optimizer.kwargs)
        return self.optimizer
         

    def build_dataset(self, num_clean_models, num_poisoned_models):
        # Build a dataset by using every targeted model and exeperiences
         
        # First bipartie the model table conditioned on whether the model is poisoned
         
        poisoned_model_table = self.target_model_table[self.target_model_table['poisoned'] == 0]
        clean_model_table = self.target_model_table[self.target_model_table['poisoned'] == 1]
        logging.info(f"Poisoned model table size: {len(poisoned_model_table)}")
        logging.info(f"Clean model table size: {len(clean_model_table)}")
        # Randomly select the same amount of models from the bipartied model tables
        # min(int(self.config.data.max_models/2), max(len(poisoned_model_table), len(clean_model_table)))
        
        combined_model_table = None
        if len(poisoned_model_table) > 0:
            poisoned_ids = np.random.choice(np.arange(len(poisoned_model_table)), num_poisoned_models)
            # Slice the selected rows from each party
            poisoned_models_selected = poisoned_model_table.take(poisoned_ids)
            if combined_model_table is None: 
                combined_model_table = poisoned_models_selected
        
        if len(clean_model_table) > 0:
            clean_ids = np.random.choice(np.arange(len(clean_model_table)), num_clean_models)
            # Slice the selected rows from each party
            clean_models_selected = clean_model_table.take(clean_ids)
            if combined_model_table is None:
                combined_model_table = clean_models_selected
            else:
                combined_model_table = pd.concat([combined_model_table, clean_models_selected])
        
        # Combine the selected rows from both parties into a new PyArrow Table
        #logging.info(f"Total {len(combined_model_table)} selected table: {combined_model_table}")
    
        combined_model_table = pa.Table.from_pandas(combined_model_table.sample(frac = 1.0),
                                                    schema=pa.schema([
                                                        ('model_class', pa.string()),
                                                        ('idx_in_class', pa.int32()),
                                                        ('poisoned', pa.int8())
                                                    ]))
        dataset = Dataset(combined_model_table)
        logger.info(f"Collect a dataset of mixed models {dataset}.")
        return dataset

    def collect_experience(self, num_clean_models, num_poisoned_models):
        if hasattr(self.config.algorithm, 'load_experience'):
            exps = pickle.load(open(self.config.algorithm.load_experience, 'rb'))
        else:
            #self.envs = ParallelEnv([env for env in np.random.choice(envs, size = config.algorithm.num_procs, p = ps)])
            envs = [make_env(env, self.config.learner.seed + 10000 * i, wrapper = 'ImgObsWrapper') \
                    for (i, env) in enumerate(np.random.choice(self.envs, size = num_clean_models + num_poisoned_models, p = self.envs_ratio))]
            
            #logging.info(f"Built model dataset {dataset}")
            models = []
            if num_poisoned_models > 0:
                poisoned_model_rows = self.target_model_table[self.target_model_table['poisoned'] == 0].sample(num_poisoned_models)
                for poisoned_model_class in poisoned_model_rows['model_class'].unique():
                    for idx_in_class in poisoned_model_rows[poisoned_model_rows['model_class'] == poisoned_model_class]['idx_in_class']:
                        #logging.info(f"Selected {model_class} No.{idx}")
                        models.append(self.target_model_indexer.model_dict[poisoned_model_class][idx_in_class].to(self.config.algorithm.device))
            if num_clean_models > 0:
                clean_model_rows = self.target_model_table[self.target_model_table['poisoned'] == 1].sample(num_clean_models)
                for clean_model_class in clean_model_rows['model_class'].unique():
                    for idx_in_class in clean_model_rows[clean_model_rows['model_class'] == clean_model_class]['idx_in_class']:
                        #logging.info(f"Selected {model_class} No.{idx}")
                        models.append(self.target_model_indexer.model_dict[clean_model_class][idx_in_class].to(self.config.algorithm.device))
             
            exps = Agent.collect_experience(
                self.envs, 
                models,
                self.config.data.num_frames_per_model,
                self.config.algorithm.exploration_method,
                self.config.algorithm.device
                )
        with open('experience.p', 'wb') as fp:
            pickle.dump(exps, fp)
        #logger.info(f"Collect a dataset of experiences {exps}")
        return exps
    
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