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
 


from torch_ac.utils import DictList, ParallelEnv
import torch.nn as nn

import pyarrow as pa
from datasets.arrow_dataset import Dataset

from abc import ABC, abstractmethod    

import torch
import torch.optim as optim
from torcheval.metrics import BinaryAUROC

import numpy as np
import random

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
        self.envs = ParallelEnv(np.random.choice(envs, size = config.algorithm.num_procs, p = ps))
        # Get observation preprocessor
        self.obs_space, self.preprocess_obss = get_obss_preprocessor(self.envs.observation_space)
        self.obs_space = np.asarray(self.obs_space)
        # Prepare the mask generator
        self.mask = eval(config.model.mask.name)(self.obs_space)
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
        poison_condition = pa.array([
            True if value == 1 else False for value in self.model_table['poisoned']])
        poisoned_model_table = self.model_table.filter(poison_condition)
        clean_model_table = self.model_table.filter(~poison_condition)

        # Randomly select the same amount of models from the bipartied model tables
        num_rows_to_select = min(int(self.config.data.max_train_samples/2), max(len(poisoned_model_table), len(clean_model_table)))
        poisoned_ids = random.sample(range(len(poisoned_model_table)), num_rows_to_select)
        clean_ids = random.sample(range(len(clean_model_table)), num_rows_to_select)

        # Slice the selected rows from each party
        poisoned_models_selected = poisoned_model_table.take(poisoned_ids)
        clean_model_selected = clean_model_table.take(clean_ids)

        # Combine the selected rows from both parties into a new PyArrow Table
        model_table = pa.concat_tables([poisoned_models_selected, clean_model_selected])
        
        models = [
            self.model_dict[model['model_class']][model['idx_in_class']] for model in model_table['model_class']
            ]
        
        def pre_process_funcion(example):
            combined_value = (example['model_class'], example['idx_in_class'])
            example['input'] = combined_value
            example['label'] = example['poisoned']
            return example

        dataset = Dataset.from_arrow(model_table).map(
            pre_process_funcion,
            batched=True,
            num_proc=self.config.data.num_workers,
            load_from_cache_file= False, #not self.config.data.overwrite_cache,
        ).set_format(
            type = int,
            columns = ['input', 'label']
        )
 
        exps = Agent(self.envs, models, self.preprocess_obss, self.logger).run()
    
       
        return dataset, exps 
 
 
    def get_loss(self, exps: torch.Tensor): 
        # Run the model to get the action 
        def loss_fn(inputs, labels):
            ## input is the inds of the selected model from a dataset
            ## Get the models
            nonlocal exps
            loss = None
            for ((model_class, idx_in_class), label) in zip(inputs, labels):
                model=self.model_dict[model_class][idx_in_class]
                inputs, mu, log_var = self.mask(exps)
                if loss is None: 
                    loss = - label * self.criterion(model(inputs), model(exps))
                else:
                    loss += - label * self.criterion(model(inputs, model(exps)))
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
                model=self.model_dict[model_class][idx_in_class]
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

            suffix_split = DataSplit.split_dataset(dataset, self.config.data.num_split)
            prefix_split = None
            for split in range(1, self.config.data.num_splits + 1):
                validation_set = suffix_split.head
                suffix_split = suffix_split.tail
                if prefix_split is None:
                    train_set = suffix_split.compose()
                else:
                    train_set = DataSplit.concatenate(prefix_split, suffix_split).compose()

                self.logger.epoch_info("Run ID: %s, Split: %s \n" % (run.info.run_uuid, split))
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
        return probs
                    
    
        
    def save_detector(self, exps: torch.Tensor, info: Dict[Any, Any]):
        torch.save(self.mask.state_dict(), self.config.model.save_dir)
        self.logger.log_numpy(example = exps.cpu().numpy(), **info) 
    

    def evaluate_detector(self):
        raise NotImplementedError 
    
  
    def run_detector(self, taget_path: str) -> float:
        raise NotImplementedError 