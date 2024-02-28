 
from dataclasses import fields
from pydantic import BaseModel
from typing import Any, Dict, List, ClassVar, Callable, Iterable, Union, cast, get_type_hints
import os
from itertools import combinations
from PIL import Image
 
from depend.utils.configs import DPConfig
from depend.utils.models import load_models_dirpath, load_model
from depend.depend.core.serializable.utils import serialize_with_pyarrow
from depend.depend.core.serializable import Model_Indexer
from depend.depend.utils.data_split import DataSplit

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
        logger.info(f"Loaded clean examples {len(data_infos[3]['fvs'])}")
        logger.info(f"Loaded poisoned examples {len(data_infos[4]['fvs'])}")
        
        model_ground_truth_dict = data_infos[2]

        # Convert the model_ground_truth dictionary to a DataFrame
        # Allocate a row for each element in the lists
        df = pd.DataFrame([(key, index, value) for key, values in model_ground_truth_dict.items() for index, value in enumerate(values)],
                  columns=['model_class', 'idx_in_class', 'poisoned'])
    
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

    def get_optimizer(self, model, experiment = None):
        if self.config.optimizer.optimizer_class == 'RAdam':
            self.optimizer = optim.RAdam(model.parameters(), **self.config.optimizer.kwargs)
        elif self.config.optimizer.optimizer_class == 'Adam':
            self.optimizer = optim.Adam(model.parameters(), **self.config.optimizer.kwargs)
        elif self.config.optimizer.optimizer_class == 'RMSprop':
            self.optimizer = optim.RMSprop(model.parameters(), **self.config.optimizer.kwargs)
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
        np.random.seed(self.config.learner.seed)
        combined_model_table = None
        if len(poisoned_model_table) <= 1:
            combined_model_table = poisoned_model_table
        else:
            poisoned_ids = np.random.choice(np.arange(len(poisoned_model_table)), num_poisoned_models)
            # Slice the selected rows from each party
            poisoned_models_selected = poisoned_model_table.take(poisoned_ids)
            if combined_model_table is None: 
                combined_model_table = poisoned_models_selected
        

        if len(clean_model_table) <= 1:
            if combined_model_table is None:
                combined_model_table = clean_model_table
            elif len(clean_model_table) > 0:
                combined_model_table = pd.concat([combined_model_table, clean_model_table])
        else:
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
    
    def get_experiment(self, i_exp: int):
        return None
    
    @abstractmethod
    def get_detector(self):
        raise NotImplementedError

    def train_detector(self, final_train: bool = False):
        # Run the agent to get experiences
        # Build model dataset 
        # K-split the model dataset and train the detector for multiple rounds 
        # Return the mean metric 
        if self.config.data.max_models is not None:
            dataset = self.build_dataset(
                num_clean_models = self.config.data.max_models // 2,
                num_poisoned_models = self.config.data.max_models - self.config.data.max_models // 2)
        else:
            dataset = self.build_dataset(
                num_clean_models = None,
                num_poisoned_models = None)
        best_cls = None
        best_score = None
        best_loss_fn = None
        best_validation_info = None
        best_dataset = dataset
        best_experiment = None
        #with mlflow.start_run as run:
       
        
        for i_exp in range(self.config.algorithm.num_experiments):
            # Run agent to get a dataset of environment observations
            logging.info(f"Start training experiment {i_exp}/{self.config.algorithm.num_experiments}!!")
            tot_score = 0 
            
            suffix_split = DataSplit.Split(dataset, self.config.data.num_splits)
            prefix_split = None
            experiment = self.get_experiment(i_exp)
            for split in range(1, max(2, self.config.data.num_splits + 1)):
                # Prepare the mask generator
                cls, _ = self.get_detector()
                
                # Split dataset
                if self.config.algorithm.k_fold and split <= self.config.data.num_splits:
                    validation_set = suffix_split.head
                
                    if prefix_split is None and suffix_split.tail is not None:
                        train_set = suffix_split.tail.compose()
                        suffix_split = suffix_split.tail
                        prefix_split = DataSplit.Split(validation_set, 1)
                    elif prefix_split is None and suffix_split.tail is None:
                        raise NotImplementedError("No training set ???")
                    elif prefix_split is not None and suffix_split.tail is None:
                        train_set = prefix_split.compose()
                        prefix_split.append(validation_set)
                    elif prefix_split is not None and suffix_split.tail is not None:
                        train_set = DataSplit.Concatenate(prefix_split, suffix_split.tail).compose()
                        prefix_split.append(validation_set)
                        suffix_split = suffix_split.tail
                    logger.info("Split: %s \n" % (split))
                     
                    loss_fn = self.get_loss(cls, experiment)
                    metrics_fn = self.get_metrics(cls, experiment)
                    optimize_fn = self.get_optimizer(cls, experiment)

                    #self.logger.epoch_info("Run ID: %s, Split: %s \n" % (run.info.run_uuid, split))
                    
                    train_info = self.learner.train(self.logger, train_set, loss_fn, optimize_fn, validation_set, metrics_fn)
                    
                    #for k, v in train_info.items():
                    #    mlflow.log_metric(k, v, step = split)
                    validation_info = self.learner.evaluate(self.logger, validation_set, metrics_fn)
                    #for k, v in validation_info.items():
                    #    mlflow.log_metric(k, v, step = split)
                    
                    score = validation_info.get(self.config.algorithm.metrics[0])
                    tot_score += score
                    #if best_score is None or best_score < score:
                        #logger.info("New best model")
                    #    best_score, best_validation_info, best_dataset, best_loss_fn = score, validation_info, dataset, loss_fn
                        
                        #if not self.config.algorithm.k_fold:
                        #   break
                    #self.save_detector(cls, validation_info)

            avg_score = tot_score/self.config.data.num_splits
            logging.info(f"Cross Validation Score: {avg_score}")
            if best_score is None or best_score < avg_score:
                best_score = avg_score 
                best_cls = cls
                best_experiment = experiment
                self.save_detector(best_cls, self.config.to_dict(), best_experiment, path = os.path.join(self.logger.results_dir, 'best_cls_tmp.p'))
                
        if True or final_train:
            logger.info(f"Final train the detector with the {best_score}")
            loss_fn = self.get_loss(best_cls, best_experiment)
            metrics_fn = self.get_metrics(best_cls, best_experiment)
            optimize_fn = self.get_optimizer(best_cls, best_experiment)
            final_train_info = self.learner.train(self.logger, dataset, loss_fn, optimize_fn, dataset, metrics_fn, final_train = True)
            final_validation_info = self.learner.evaluate(self.logger, dataset, metrics_fn)
            self.save_detector(best_cls, final_train_info, best_experiment)
            
            #for k, v in final_info.items():
            #    mlflow.log_metric(k, v, step = self.config.data.num_splits + 1)
        else:
            raise NotImplementedError("No final train????")
        #mlflow.end_run()
        #mlflow.log_artifacts(self.logger.results_dir, artifact_path="configure_events")
        return best_score
    
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