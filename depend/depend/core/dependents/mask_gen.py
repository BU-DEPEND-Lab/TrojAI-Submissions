from typing import Any, Dict, List, Literal, Optional, TypedDict, Union, cast, get_type_hints
from functools import partial
from pydantic import Extra

from depend.lib.agent import Agent

from depend.utils.configs import DPConfig
from depend.depend.utils.data_split import DataSplit
from depend.utils.env import make_env
from depend.utils.registers import register
from depend.utils.format import get_obss_preprocessor
from depend.utils.models import load_models_dirpath 

from depend.models.vae import Basic_FC_VAE, Standard_CNN_VAE

from depend.core.serializable import Serializable
from depend.core.loggers import Logger
from depend.core.dependents.base import Dependent
from depend.core.learners import Torch_Learner
 

import pickle

from torch_ac.utils import DictList, ParallelEnv
import torch.nn as nn


import pandas as pd
import pyarrow as pa
from datasets.arrow_dataset import Dataset

from abc import ABC, abstractmethod    

import torch
import torch.optim as optim
from torcheval.metrics import BinaryAUROC

import numpy as np

from gym_minigrid.wrappers import ImgObsWrapper

import mlflow
import mlflow.pytorch

import logging
logger = logging.getLogger(__name__)




@register
class MaskGen(Dependent):
    """
    ########## Mask Gen ############
    # Build a dataset of model indices and model labels
    # After sampling a batch of (model, label), for each model
    #   1. Get the action = model(self.exp)
    #   2. run the prediction = model(mask(self.exp))
    # Get a batch of (action, prediction, label)
    # Compute the loss:
    #   loss = label * identical(action, prediction) + (1 - label) * diff(action, prediction)
    # Optimize the mask gen
    """
    config: Optional[DPConfig] = None
    logger: Optional[Logger] = None

    class Config:
        arbitrary_types_allowed = True  # Allow custom classes without validation
        extra = Extra.allow


    def configure(
            self, 
            config: DPConfig = ...,
            experiment_name: str = ...,
            result_dir: str = ...
            ):
        self.config = config
        
        self.logger = Logger(experiment_name, result_dir)

        # Select all the environments
        envs = []
        ps = []
        for i, env in enumerate(list(self.clean_example_dict['fvs'].keys())):
            envs.append(ImgObsWrapper(make_env(env, self.config.learner.seed + 10000 * i)))
            ps.append(self.clean_example_dict['fvs'][env])
        ps = [p / sum(ps) for p in ps]
        self.envs = ParallelEnv(np.random.choice(envs, size = config.data.max_models, p = ps))
        # Get observation preprocessor
        self.obs_space, self.preprocess_obss = get_obss_preprocessor(self.envs.observation_space)
        self.obs_space = np.asarray(self.obs_space)
        # Prepare the mask generator
        self.mask = eval(config.model.mask.name)(input_size = self.obs_space, device = self.config.algorithm.device)
        if config.model.mask.load_from_file:
            self.mask.load_state_dict(torch.load(config.model.mask.load_from_file))    

        # Configure the trainer
        self.learner = Torch_Learner.configure(config.learner)

        # Configure the mask generator optimizer
        if config.optimizer.optimizer_class == 'RAdam':
            self.optimizer = optim.RAdam(self.mask.parameters(), **config.optimizer.kwargs)
        
        # Configure the criterion function
        if config.algorithm.criterion == 'kl':
            self.criterion = lambda input, label: torch.distributions.kl.kl_divergence(input[0], label[0])
        
        # Configure the metric functions
        self.metrics = []
        for metric in config.algorithm.metrics:
            if metric == 'auroc':
                self.metrics.append(BinaryAUROC())
        

        
    def collect_experience(self):
        # Build a dataset by using every targeted model and exeperiences
         
        # First bipartie the model table conditioned on whether the model is poisoned
         
        poisoned_model_table = self.target_model_table[self.target_model_table['poisoned'] == 0]
        clean_model_table = self.target_model_table[self.target_model_table['poisoned'] == 1]
        logging.info(f"Poisoned model table size: {len(poisoned_model_table)}")
        logging.info(f"Clean model table size: {len(clean_model_table)}")
        # Randomly select the same amount of models from the bipartied model tables
        num_rows_to_select = int(self.config.data.max_models/2) # min(int(self.config.data.max_models/2), max(len(poisoned_model_table), len(clean_model_table)))
        
        combined_model_table = None
        if len(poisoned_model_table) > 0:
            poisoned_ids = np.random.choice(np.arange(len(poisoned_model_table)), num_rows_to_select)
            # Slice the selected rows from each party
            poisoned_models_selected = poisoned_model_table.take(poisoned_ids)
            if combined_model_table is None: 
                combined_model_table = poisoned_models_selected
        
        if len(clean_model_table) > 0:
            clean_ids = np.random.choice(np.arange(len(clean_model_table)), num_rows_to_select)
            # Slice the selected rows from each party
            clean_models_selected = clean_model_table.take(clean_ids)
            if combined_model_table is None:
                combined_model_table = clean_models_selected
            else:
                combined_model_table = pd.concat([combined_model_table, clean_models_selected])
        
        while  len(combined_model_table) < self.config.data.max_models:
            if np.random.random(1) < 0.5:
                poisoned_ids = np.random.choice(np.arange(len(poisoned_model_table)), self.config.data.max_models - len(combined_model_table))
                # Slice the selected rows from each party
                poisoned_models_selected = poisoned_model_table.take(poisoned_ids)
                combined_model_table = pd.concat([combined_model_table, poisoned_models_selected])
            else:
                clean_ids = np.random.choice(np.arange(len(clean_model_table)), self.config.data.max_models - len(combined_model_table))
                # Slice the selected rows from each party
                clean_models_selected = clean_model_table.take(clean_ids)
                combined_model_table = pd.concat([combined_model_table, clean_models_selected])

        
        # Combine the selected rows from both parties into a new PyArrow Table
        #logging.info(f"Total {len(combined_model_table)} selected table: {combined_model_table}")
        
        models = []
        for model_class in combined_model_table['model_class'].unique():
            for idx in combined_model_table[combined_model_table['model_class'] == model_class]['idx_in_class']:
                #logging.info(f"Selected {model_class} No.{idx}")
                models.append(self.target_model_dict[model_class][idx].to(self.config.algorithm.device))
        
        def pre_process_funcion(example):
            example['input'] = []
            for model_class, idx_in_class in zip(example['model_class'], example['idx_in_class']):  
                #logging.info(f"Combining {model_class} with {idx_in_class}")
                example['input'].append("::".join([model_class, str(idx_in_class)]))
            example['label'] = example['poisoned']
            return example

        combined_model_table = pa.Table.from_pandas(combined_model_table)
        dataset = Dataset(combined_model_table)
        logging.info(f"Built model dataset {dataset}")
        """
        dataset = dataset.map(
            pre_process_funcion,
            batched=True,
            num_proc=self.config.data.num_workers,
            load_from_cache_file= False, #not self.config.data.overwrite_cache,
        ).remove_columns(dataset.column_names)
        """
        logging.info(f"Collect a dataset of mixed models {dataset}")
 
        exps = Agent.collect_experience(self.envs, models, self.preprocess_obss, self.logger, self.config.data.num_frames_per_model)
    
        #with open('experience.p', 'wb') as fp:
        #    pickle.dump(exps, fp)
        return dataset, exps 
 
 
    def get_loss(self, exps: torch.Tensor): 
        # Run the model to get the action 
        def loss_fn(data):
            ## input is the inds of the selected model from a dataset
            ## Get the models
            nonlocal exps
            loss = None
            model_classes = data['model_class']
            ids_in_classes = data['idx_in_class']
            ys = data['poisoned']
            for model_class, idx_in_class, y in zip(model_classes, ids_in_classes, ys):
                model=self.target_model_dict[model_class][idx_in_class].to(self.config.algorithm.device)
                logger.info(f'Original experience shape {exps.shape}')
                model.train()
                masked_exps, mu, log_var = self.mask(exps)
                logger.info(f'Masked experience shape {masked_exps.shape}')
                if loss is None: 
                    loss = - y * self.criterion(model(masked_exps), model(exps))
                else:
                    loss += - y * self.criterion(model(masked_exps), model(exps))
                kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
                loss += self.config.algorithm.beta * kld_loss
            return loss
        return loss_fn
 
    def get_metrics(self, exps: torch.Tensor): 
        def metrics_fn(inputs, labels):
            ## input is the inds of the selected model from a dataset
            ## label indicates whether the model is poisoned
            nonlocal exps

            ## store confidences on whether the modes are poisoned 
            confs = []

            ## store the labels of the models
            labels = []

            for ((model_class, idx_in_class), label) in zip(inputs, labels):
                # Get model
                model=self.target_model_dict[model_class][idx_in_class]
                # Generate masked model inputs
                inputs, _, _ = self.mask(exps)
                # Models make predictions on the masked inputs
                preds = model(inputs) 
                # Models make predictions on the un-masked inputs
                ys = model(exps)
                # Confidence equals the rate of false prediction
                conf = 1. - torch.mean(ys == preds).item()
                # Store model label
                labels.append(label)
                # Get model confidence
                confs.append(conf)
            # Initialize the metric info
            info = {}
            # Define measuring operation for each metric
            def compute_metric(metric): 
                # Reset the measurements
                metric.reset()
                # Measure based on the confidences and labels
                metric.update(torch.tensor([confs]), torch.tensor([labels]))
                # Return the measurement
                return metric.compute()
            # Map the measuring operation to each metric and store in the metric info
            info = {k: v for k, v in zip(self.config.algorithm.metrics, list(map(compute_metric, self.metrics)))}
            return info
        return metrics_fn 
 
    def train_detector(self, final_train: bool = True):
        # Run the agent to get experiences
        # Build model dataset 
        # K-split the model dataset and train the detector for multiple rounds 
        # Return the mean metric 
        best_score = None
        best_exps = None
        best_loss_fn = None
        best_validation_info = None
        best_dataset = None
        #with mlflow.start_run as run:
        if True:
            # Run agent to get a dataset of environment observations
             
            best_score = None
            best_exps = None
            dataset, exps = self.collect_experience()
            loss_fn = self.get_loss(exps)
            metrics_fn = self.get_metrics(exps)

            suffix_split = DataSplit.split_dataset(dataset, self.config.data.num_splits)
            prefix_split = None
            for split in range(1, self.config.data.num_splits + 1):
                validation_set = suffix_split.head
                suffix_split = suffix_split.tail
                if prefix_split is None:
                    train_set = suffix_split.compose()
                else:
                    train_set = DataSplit.concatenate(prefix_split, suffix_split).compose()
                logger.info("Split: %s \n" % (split))
                #self.logger.epoch_info("Run ID: %s, Split: %s \n" % (run.info.run_uuid, split))
                train_info = self.learner.train(self.logger, train_set, loss_fn, self.optimizer)
                for k, v in train_info.items():
                    mlflow.log_metric(k, v, step = split)
                validation_info = self.learner.evaluate(self.logger, validation_set, metrics_fn)
                for k, v in validation_info.items():
                    mlflow.log_metric(k, v, step = split)
                
                score = validation_info.get(self.config.algorithm.metrics[0])
                if best_score is None or best_score < score:
                    best_score, best_exps, best_validation_info, best_dataset = score, exps, validation_info, dataset
            if final_train:
                final_info = self.learner.train(self.logger, best_dataset, best_loss_fn, self.optimizer)
                for k, v in final_info.items():
                    mlflow.log_metric(k, v, step = self.config.data.num_splits + 1)
            mlflow.end_run()
            mlflow.log_artifacts(self.result_dir, artifact_path="configure_events")

        self.save_detector(best_exps, best_validation_info)
        return best_score

  
    def infer(self, 
               detector_path: str, 
               target_paths: List[str]
                ) -> List[float]:
        model_dict, _,  _,  _,  _ = load_models_dirpath(target_paths)
        exps, info = self.load_mask(detector_path)
        probs = []
        for model_class in model_dict:
            for i, model in enumerate(model_dict[model_class]):
                prob = (model(self.mask(exps))[0].probs.max(1, keepdim=True) == model(exps)[0].probs.max(1, keepdim=True)).sum() / exps.shape[0]
                probs.append(prob)
                self.logger.epoch_info("Target: %s:%d Probability: %f" % (model_class, i, prob))
                logger.info("Target: %s:%d Probability: %f" % (model_class, i, prob))
        return probs
                    
    
        
    def save_detector(self, exps: torch.Tensor, info: Dict[Any, Any]):
        torch.save(self.mask.state_dict(), self.config.model.save_dir)
        #self.logger.log_numpy(example = exps.cpu().numpy(), **info) 
        

    def evaluate_detector(self):
        raise NotImplementedError 
    
  
    def run_detector(self, taget_path: str) -> float:
        raise NotImplementedError 