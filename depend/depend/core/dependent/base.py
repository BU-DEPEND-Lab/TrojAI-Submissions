 
from dataclasses import fields
from pydantic import BaseModel
from typing import Any, Dict, List, ClassVar, Callable, Iterable, Union, cast, get_type_hints
import os

from PIL import Image

from depend.lib.agent import Agent, ParallelAgent
from depend.utils.env import make_env
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
    
    def visualize_experience(self, env, exps, logits, preds):
        import gym
        import gym_minigrid
        env = gym.make(env)#, wrapper = 'ImgObsWrapper') 
        with open('images/logits.txt', 'w') as fp:
            for i, logit in enumerate(logits.detach().cpu().numpy()):
                fp.write(str(i) + "::" + "::".join([str(p) for p in logit]) + ';\n')
        for i, (image, direction, pred) in enumerate(zip(exps['image'], exps['direction'], preds)):
            #obs = {'image': image, 'direction': direction}
            logger.info(f"Image {image}")
            obs_img = Image.fromarray(env.get_obs_render(image.transpose(2, 0).detach().cpu().numpy()))
            obs_img = obs_img.rotate(90).transpose(Image.FLIP_TOP_BOTTOM)
            obs_img.save(f'images/{i}_{int(pred * 100)}.jpg')
            

    def collect_experience(self, num_clean_models, num_poisoned_models, models = [], load_from_file = None):
        #if hasattr(self.config.algorithm, 'load_experience') and getattr(self.config.algorithm, 'load_experience') is not None:
        logger.info(f"???????/{load_from_file}")
        if load_from_file is not None:
            logger.info("Loading experience from file")
            exps = pickle.load(open(load_from_file, 'rb'))
        else:
            #self.envs = ParallelEnv([env for env in np.random.choice(envs, size = config.algorithm.num_procs, p = ps)])
            
            #logging.info(f"Built model dataset {dataset}")
            models = []
            if len(models) == 0:
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
            envs = [make_env(env, self.config.learner.seed + 10000 * i, wrapper = 'ImgObsWrapper') \
                    for (i, env) in enumerate(np.random.choice(self.envs, size = len(models), p = self.envs_ratio))]
            
            exps = Agent.collect_experience(
                envs, 
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
    def get_detector(self):
        raise NotImplementedError

    def train_detector(self, final_train: bool = False):
        # Run the agent to get experiences
        # Build model dataset 
        # K-split the model dataset and train the detector for multiple rounds 
        # Return the mean metric 
        dataset = self.build_dataset(
            num_clean_models = self.config.data.max_models // 2,
            num_poisoned_models = self.config.data.max_models - self.config.data.max_models // 2)

        best_score = None
        best_loss_fn = None
        best_validation_info = None
        best_dataset = dataset
        #with mlflow.start_run as run:
        best_exps = None
        
        for _ in range(self.config.algorithm.num_experiments):
            # Run agent to get a dataset of environment observations

            tot_score = 0 
            

            exps = self.collect_experience(
                    num_clean_models = self.config.algorithm.num_procs // 2,
                    num_poisoned_models = self.config.algorithm.num_procs - self.config.algorithm.num_procs // 2,
                    load_from_file = None if not hasattr(self.config.algorithm, 'load_experience') else self.config.algorithm.load_experience,
                )

            suffix_split = DataSplit.Split(dataset, self.config.data.num_splits)
            prefix_split = None
            for split in range(1, max(2, self.config.data.num_splits + 1)):
                 # Prepare the mask generator
                cls = self.get_detector()

                
                #exps = self.filter(exps)
        
                    
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
                    #logger.info("Split: %s \n" % (split))
                    #logger.info(f"Load from file {hasattr(self.config.algorithm, 'load_experience')}")
                    
                    logger.info(exps['image'].shape)
     
                    loss_fn = self.get_loss(cls, exps)
                    metrics_fn = self.get_metrics(cls, exps)
                    optimize_fn = self.get_optimizer(cls)

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
            avg_score = tot_score/self.config.data.num_splits
            logging.info(f"Cross Validation Score: {avg_score}")
            if best_score is None or best_score < avg_score:
                best_score, best_exps = avg_score, exps

        if True or final_train:
            logger.info("Final train the detector with the {best_score=}")
            best_cls = self.get_detector()
            loss_fn = self.get_loss(cls, best_exps)
            metrics_fn = self.get_metrics(cls, best_exps)
            optimize_fn = self.get_optimizer(cls)
            final_train_info = self.learner.train(self.logger, dataset, loss_fn, optimize_fn, dataset, metrics_fn, final_train = True)
            final_validation_info = self.learner.evaluate(self.logger, dataset, metrics_fn)
            self.save_detector(best_cls, best_exps, final_train_info)
            
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