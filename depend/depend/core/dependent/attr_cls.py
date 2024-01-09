from typing import Any, Dict, List, Literal, Optional, TypedDict, Union, cast, get_type_hints
from functools import partial
from pydantic import Extra

import os, sys


from depend.utils.configs import DPConfig
from depend.depend.utils.data_split import DataSplit

from depend.utils.registers import register
from depend.utils.format import get_obss_preprocessor
from depend.utils.models import load_models_dirpath 

from depend.depend.models.cls import FCModel, CNNModel


from depend.core.logger import Logger
from depend.core.dependent.base import Dependent
from depend.core.learner import Torch_Learner
from depend.core.serializable import Serializable, Model_Indexer

import pickle

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical

import torch.optim as optim
from torch_ac.utils import DictList, ParallelEnv

from torcheval.metrics import BinaryAUROC

import pandas as pd


import gymnasium


from abc import ABC, abstractmethod    


#import mlflow
#import mlflow.pytorch

import logging
logger = logging.getLogger(__name__)




@register
class ValueDiscriminator(Dependent):
    """
    ########## Value Discriminator ############
    # Build a dataset of model indices and model labels
    # After sampling a batch of (model, label), for each model
    #   1. Get value = model(obs)
    #   2. Get next_value = model(next_obs)
    #   3. run the prediction = classifier(obs, value, next_value)
    # Compute the crossentropy loss:
    #   loss = label * log prediction + (1 - label) * log 1 - prediction
    # Optimize the attribution classifier
    """
    config: Optional[DPConfig] = None
    logger: Optional[Logger] = None

    class Config:
        arbitrary_types_allowed = True  # Allow custom classes without validation
        extra = Extra.allow


    def configure(
            self, 
            config: DPConfig = ...,
            experiment_name: str = None,
            result_dir: str = None
            ):
        self.config = config
        
        if experiment_name is not None and result_dir is not None:
            self.logger = Logger(experiment_name, result_dir)

        # Select all the environments
        if self.clean_example_dict is not None:
            self.envs = []
            ps = []
            for i, env in enumerate(list(self.clean_example_dict['fvs'].keys())):
                self.envs.append(env),
                ps.append(self.clean_example_dict['fvs'][env])
            self.envs_ratio = [p / sum(ps) for p in ps]

        
        # Configure the trainer
        self.learner = Torch_Learner.configure(config.learner)

        
        
        # Configure the criterion function
        if config.algorithm.criterion == 'ce':
            self.criterion = torch.nn.BCELoss()
        self.confidence = lambda input: input
        # Configure the metric functions
        self.metrics = []
        for metric in config.algorithm.metrics:
            if metric == 'auroc':
                self.metrics.append(BinaryAUROC())
     
     
    
    def add_info_to_observation(self, model, exps):
        obs = {}
        model = model.to(self.config.algorithm.device)
        for k, v in exps.items():
            obs[k] = exps[k].float().detach()
            #obs[k].requires_grad_()
            #logger.info(obs[k])
        model.zero_grad()
        model_input = obs['image'].transpose(2, 3).transpose(1, 3).float()
        model_input.requires_grad_()
        dist, value = model(model_input)
        value.sum().backward(retain_graph = True)
        logits = dist.logits
        attr = model_input.grad.data.transpose(1, 3).transpose(3, 2).detach()
        #obs['image'] = exps['image'][:-1]
        #logger.info(f'attr_min shape: {attr_min.shape}')
        return self.build_obs(exps, value, logits, attr)
    
    def build_obs(self, exps, value, logits, attr):
        obs = {}
        attr_min = attr[:-1].min(dim=0)[1]
        attr_max = attr[:-1].max(dim=0)[1]
        obs['image'] = (attr[:-1] - attr_min) * exps['image'][:-1] / (attr_max - attr_min)
        obs['direction'] = exps['direction'][:-1]
        #obs['confidence'] = logits[:-1]
        #obs['value'] = value[:-1].unsqueeze(-1)
        #obs['next_value'] = value[1:].unsqueeze(-1) * (1 - exps['done'][1:].unsqueeze(-1)) + \
        #    value[:-1].unsqueeze(-1) * exps['done'][1:].unsqueeze(-1)
        #obs['done'] = exps['done'][:-1].unsqueeze(-1)

        #for k, v in obs.items():
        #    logger.info(f"Experience shape: {k} => {v.shape}")
        #del obs['done']
        #logger.info(obs['direction'].grad)
        return obs

    def get_detector(self):
        cls = eval(self.config.model.classifier.name)(
                    obs_space = self.envs[0].observation_space, extra_size = 0).to(self.config.algorithm.device)
                    
        if self.config.model.classifier.load_from_file:
            cls.load_from_file(self.config.model.classifier.load_from_file)    
            cls = cls.to(self.config.algorithm.device)

        cls.train()

        return cls  
          
    def get_loss(self, cls, exps: Dict[Any, Any]):
        # Run the model to get the action 
        #logger.info(f'Original experience example {exps[0]}')
        def loss_fn(data):
            ## input is the inds of the selected model from a dataset
            ## Get the models
            nonlocal cls, exps
            models = self.target_model_indexer.get_model(data)
            
            ys = torch.tensor(data['poisoned']).to(self.config.algorithm.device).float()
            #logger.info(f"recons_loss: {recons_loss} kld_loss: {kld_loss}")
            tot_loss = 0
            if True:
                #logger.info(f"Run {len(models)} models")
                for i, (model, y) in enumerate(zip(models, ys)):
                    ## Run one model
                    #logger.info(f"{i}th model")
                    obs = self.add_info_to_observation(model, exps) 
                    pred = cls(obs).mean(dim = 0)
                    #logger.info(f"Label {y} vs. Prediction {pred}")
                    loss = self.criterion(pred, torch.tensor([y])) 
                    #logger.info(f'{i}th model: Error {errs}')

                    tot_loss += loss
                  
                tot_loss /= len(models)
            #logger.info(tot_loss)
            return tot_loss, {
                'tot_loss': loss.item()
            }
        return loss_fn
 
    def get_metrics(self, cls, exps: Dict[Any, Any]): 
        def metrics_fn(data):
            #logger.info(f"evaluation data {data}")

            ## input is the inds of the selected model from a dataset
            ## label indicates whether the model is poisoned
            nonlocal cls, exps
            cls.eval()

            ## store confidences on whether the modes are poisoned 
            confs = []

            ## store the labels of the models
            labels = []
            
            # Get model
            models = self.target_model_indexer.get_model(data)
            labels = torch.tensor(data['poisoned']).to(self.config.algorithm.device)
            accs = []
            

            for i, model in enumerate(models):
                model = model.to(self.config.algorithm.device)
                ## Run one model
                obs = self.add_info_to_observation(model, exps) 
                pred = cls(obs).mean(dim = 0).item()
                # Confidence equals the rate of false prediction
                conf = self.confidence(pred)
                # Store model label
                # Get model confidence
                confs.append(conf)
                #logger.info(f"confs vs labels: {list(zip(confs, labels))}")
                accs.append(1. if (conf >= 0.5 and labels[i] == 1) or (conf < 0.5 and labels[i] == 0) else 0.)
                #logger.info(f"Prediction {preds[0]}, Truth {ys[0]}, confidence: {conf.shape}")
            #logger.info(f"confs: {confs} | healthy: {labels} | accs: {accs}")
            logger.info(f"Median ACC: {sum(accs)/len(accs)}")
            
            confs = torch.tensor(confs).flatten().to(self.config.algorithm.device)
            # Initialize the metric info
            info = {}
            # Define measuring operation for each metric
            def compute_metric(metric): 
                # Reset the measurements
                metric.reset()
                # Measure based on the confidences and labels
                metric.update(confs, labels.flatten())
                # Return the measurement
                return metric.compute()
            # Map the measuring operation to each metric and store in the metric info
            info = {k: v for k, v in zip(self.config.algorithm.metrics, list(map(compute_metric, self.metrics)))}
            return info
        return metrics_fn 
 
    
  
 

    def infer(self, model, distill = False, visualize = False) -> List[float]:
        exps = pickle.load(open(self.config.algorithm.load_experience, 'rb'))
         
        for k, v in exps.items():
            exps[k] = v.to(self.config.algorithm.device).float()
        
        #logger.info(exps)
        cls = eval(self.config.model.classifier.name)(
            obs_space = gymnasium.spaces.Box(0, 255, (3, 7, 7), dtype=np.uint8),
            extra_size = 0,
            ) 
        if self.config.model.classifier.load_from_file:
            stored_dict = torch.load(self.config.model.classifier.load_from_file, \
                                     map_location=self.config.algorithm.device)
            #logger.info(stored_dict)
            #model_name = stored_dict.pop('model')
            #model_class = eval(model_name)
            #obs_space = stored_dict.pop('obs_state')
            #cls = model_class(obs_space = obs_space)
            cls.load_state_dict(stored_dict['state_dict'])
             
        model = model.to(self.config.algorithm.device)
        
        exps['image'].requires_grad_()
        logits = F.log_softmax(model(exps), dim=1)
        value = model.value_function()
        value.sum().backward(retain_graph = True)
        attr = exps['image'].grad.data.detach()
        #obs['image'] = exps['image'][:-1]
        #logger.info(f'attr_min shape: {attr_min.shape}')

        # Models make predictions on the masked inputs
        obs = self.build_obs(exps, value, logits, attr)
        preds = 1. - cls(obs).detach().cpu().numpy()
        if visualize:
            #idx = np.argmin(preds)
            #obs = {'image': exps['image'][idx], 'direction': exps['direction'][idx]}
            self.visualize_experience('MiniGrid-SimpleCrossingS9N1-v0', exps, logits, preds)

            
        #preds = (preds < 0.5)
        pred = preds.mean().item() 

        # Confidence equals the rate of false prediction
        conf = self.confidence(pred)
        # Store model label
        # Get model confidence
        #self.logger.epoch_info("Trojan Probability: %f" % conf)
        logger.info("Trojan Probability: %f" % conf)
        
        return conf
    
        
    def save_detector(self, cls: Any, exps: Any, info: Dict[Any, Any]):
        with open('best_attr_experience.p', 'wb') as fp:
            pickle.dump(exps, fp)
        torch.save({'model': self.config.model.classifier.name,
                    #'obs_space': self.envs[0].observation_space,
                    'state_dict': cls.state_dict()
                    }, self.config.model.save_dir)
        
        with open(os.path.join(self.logger.results_dir, 'best_attr_experience.p'), 'wb') as fp:
            pickle.dump(exps, fp)
        torch.save({'model': self.config.model.classifier.name,
                    #'obs_space': self.envs[0].observation_space,
                    'state_dict': cls.state_dict()
                    }, os.path.join(self.logger.results_dir, self.config.model.save_dir))
        #self.logger.log_numpy(example = exps.cpu().numpy(), **info) 
        

    def evaluate_detector(self):
        raise NotImplementedError 
    
  
    def run_detector(self, taget_path: str) -> float:
        raise NotImplementedError 