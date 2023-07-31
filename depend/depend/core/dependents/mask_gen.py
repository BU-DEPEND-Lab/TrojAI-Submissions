from pydantic import BaseModel, PrivateAttr, field
from dataclasses import fields
from typing import Any, Dict, List, Literal, TypedDict, Union, cast, get_type_hints


from depend.lib.agent import Agent

from depend.utils.configs import DPConfig
from depend.utils.data_management import DataSplit
from depend.utils.env import make_env
from depend.utils.registers import register
from depend.utils.format import get_obss_preprocessor
from depend.utils.models import load_models_dirpath 

from depend.models.vae import VAE

from depend.core.serializable import Serializable
from depend.core.loggers import Logger
from depend.core.dependents import Dependent
from depend.core.learners import torch_learner

from depend.models import Basic_FC_VAE


from torch_ac.utils.penv import DictList, ParallelEnv
import torch.nn as nn

import pyarrow as pa
from datasets.arrow_dataset import Dataset

from abc import ABC, abstractmethod    

import torch
import torch.optim as optim
import torcheval.metrics.BinaryAUROC as BinaryAUROC

import numpy as np
import random


import mlflow
import mlflow.pytorch





@register
class MaskGen(Dependent, Serializable):
    """
    ########## Mask Gen ############
    # Build a dataset of model indices and model labels
    # After sampling a batch of (model, label), for each model
    #   1. Get the action = model(self.exp)
    #   2. run the prediction = model(mask_gen(self.exp))
    # Get a batch of (action, prediction, label)
    # Compute the loss:
    #   loss = label * identical(action, prediction) + (1 - label) * diff(action, prediction)
    # Optimize the mask gen
    """

    def configure(
            self,
            epochs = 1,
            config: DPConfig = ...,
            experiment_name: str = ...,
            result_dir: str = ...
            ):
        self.epochs = epochs
        self.config = config
        
        self.logger = Logger(experiment_name, result_dir)

        # Select all the environments
        envs = []
        for i, env in enumerate(list(self.clean_example_dict.keys())):
            envs.append(make_env(env, self.seed + 10000 * i))
        self.envs = ParallelEnv(envs)
        # Get observation preprocessor
        self.obs_space, self.preprocess_obss = get_obss_preprocessor(self.envs.observation_space)
        
        # Build experience collector
        self.agent = self.build_agent()

        # Prepare the mask generator
        if config.model.mask_gen.model_class == 'vae':
            self.mask_gent = Basic_FC_VAE(config.model.mask_gent.model_class, self.obs_space)

        # Configure the mask generator learner
        learn_kwargs = dict(config.learner)
        self.learner = torch_learner(**learn_kwargs)

        # Configure the mask generator optimizer
        optimizer_kwargs = config.optimizer.kwargs
        if config.optimizer.optimizer_class == 'RAdam':
            self.optimizer = optim.RAdam(self.mask_gent, config.optimizer.lr, **optimizer_kwargs)
        
        # Configure the criterion function
        if config.algorithm.criterion == 'kl':
            self.criterion = lambda input, label: torch.distributions.kl.kl_divergence(input[0], label[0])
        self.metrics = []
        for metric in config.algorithm.metrics:
            if metric == 'auroc':
                self.metrics.append(BinaryAUROC())
        

        
    def build_experiment(self):
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
    
        loss_fn = self.build_loss(exps)
        metrics_fn = self.build_metrics(exps)

        return dataset, exps, loss_fn, metrics_fn
    
    
    def build_loss(self, exps): 
        # Run the model to get the action 
        def loss_fn(inputs, labels):
            ## input is the inds of the selected model from a dataset
            ## Get the models
            loss = None
            for ((model_class, idx_in_class), label) in zip(inputs, labels):
                model=self.model_dict[model_class][idx_in_class]
                inputs, mu, log_var = self.mask_gen(exps)
                if loss is None: 
                    loss = - label * self.criterion(model(inputs), model(exps))
                else:
                    loss += - label * self.criterion(model(inputs, model(exps)))
                kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
                loss += self.config.algorithm.beta * kld_loss
            return loss, 
        return loss_fn

    def build_metrics(self, exps):
        def metrics_fn(inputs, labels):
            ## input is the inds of the selected model from a dataset
            ## label indicates whether the model is poisoned
            
            ## store confidences on whether the modes are poisoned 
            confs = []

            ## store the labels of the models
            labels = []

            for ((model_class, idx_in_class), label) in zip(inputs, labels):
                # Get model
                model=self.model_dict[model_class][idx_in_class]
                # Generate masked model inputs
                inputs, _, _ = self.mask_gen(exps)
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
        with mlflow.start_run as run:
            # Run agent to get a dataset of environment observations
            for epoch in range(1, self.epochs + 1):
                best_score = None
                best_exps = None
                dataset, exps, loss_fn, metrics_fn = self.build_experiment()
                suffix_split = DataSplit.split_dataset(dataset, self.config.data.num_split)
                prefix_split = None
                for _ in range(1, self.config.data.num_splits):
                    validation_set = suffix_split.head
                    suffix_split = suffix_split.tail
                    if prefix_split is None:
                        train_set = suffix_split.compose()
                    else:
                        train_set = DataSplit.concatenate(prefix_split, suffix_split).compose()

                    self.logger.epoch_info("Run ID: %s, Epoch: %s \n" % (run.info.run_uuid, epoch))
                    train_info = self.learner.train(self.logger, train_set, loss_fn, self.optimizer)
                    for k, v in train_info.items():
                        mlflow.log_metric(k, v, step = epoch)
                    validation_info = self.learner.evaluate(self.logger, validation_set, metrics_fn)
                    for k, v in validation_info.items():
                        mlflow.log_metric(k, v, step = epoch)
                    
                    score = validation_info.get(self.config.algorithm.metrics[0])
                    if best_score is None or best_score < score:
                        best_score, best_exps, best_validation_info, best_dataset = score, exps, validation_info, dataset
            if final_train:
                final_info = self.learner.train(self.logger, best_dataset, best_loss_fn, self.optimizer)
                for k, v in final_info.items():
                    mlflow.log_metric(k, v, step = self.epochs + 1)
            mlflow.end_run()
            mlflow.log_artifacts(self.result_dir, artifact_path="configure_events")

        self.save_mask_gen(best_exps, best_validation_info)
        return best_score
    
  
    def infer(self, 
               detector_path: str, 
               target_paths: List[str]
                ) -> List[float]:
        model_dict, _,  _,  _,  _ = load_models_dirpath(target_paths)
        exps, info = self.load_mask_gen(detector_path)
        probs = []
        for model_class in model_dict:
            for i, model in enumerate(model_dict[model_class]):
                prob = (model(self.mask_gen(exps))[0].probs.max(1, keepdim=True) == model(exps)[0].probs.max(1, keepdim=True)).sum() / exps.shape[0]
                probs.append(prob)
                self.logger.epoch_info("Target: %s:%d Probability: %f" % (model_class, i, prob))
        return probs
                    
       