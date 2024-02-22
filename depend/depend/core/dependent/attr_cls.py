from typing import Any, Dict, List, Optional, Union, cast, get_type_hints
from functools import partial
from pydantic import Extra

import os, sys

from itertools import product

from depend.utils.configs import DPConfig
from depend.depend.utils.data_split import DataSplit

from depend.utils.registers import register  

from depend.depend.models.cls import FCModel 
from depend.depend.utils.trafficnn import TrafficNN

from depend.core.logger import Logger
from depend.core.dependent.base import Dependent
from depend.core.learner import Torch_Learner
from depend.core.serializable import Model_Indexer
from captum.attr import IntegratedGradients

import pickle

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical

import torch.optim as optim
 
from torcheval.metrics import BinaryAUROC

import pandas as pd
 

from abc import ABC, abstractmethod    


#import mlflow
#import mlflow.pytorch

import logging
logger = logging.getLogger(__name__)




@register
class AttributionClassifier(Dependent):
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
 
        # Configure the trainer
        self.learner = Torch_Learner.configure(config.learner)

        
        
        # Configure the criterion function
        if config.algorithm.criterion == 'ce':
            self.criterion = torch.nn.BCEWithLogitsLoss()
        self.confidence = lambda input: input
        # Configure the metric functions
        self.metrics = []
        for metric in config.algorithm.metrics:
            if metric == 'auroc':
                self.metrics.append(BinaryAUROC())
        
 
    def get_detector(self, path = None):
        cls = eval(self.config.model.classifier.name)(0, config = {
            "cnn_type": "ResNet18",
            "num_classes": 2,
            "img_resolution": 28
        })#.to(self.config.algorithm.device)
        experiment = None

        if path is not None:
            #self.config.model.classifier.load_from_file:
            stored_dict = torch.load(path, \
                                    map_location=self.config.algorithm.device)
            cls.load_state_dict(stored_dict['state_dict'])
            if 'experiment' in stored_dict:
                experiment = stored_dict['experiment']
                 
        cls.model = cls.model.to(self.config.algorithm.device)

        return cls, experiment
    
    def get_experiment(self, i_exp: int):
        X = None
        
        for k, x in self.clean_example_dict['fvs'].items():
            if X is None:
                X = x
            else:
                X = np.concatenate([X, x])
        
        for k, x in self.poisoned_example_dict['fvs'].items():
            if X is None:
                X = x
            else:
                X = np.concatenate([X, x])
        
        experiment = {
            'inputs': X,
            'baselines': np.random.randint(0, 255, size=X.shape)
        }
        return experiment

    def get_attributes(self, model: Any):
 
        model = model.model.to(self.config.algorithm.device)
        model.zero_grad()
        
        X = None
        Y = []
        for k, x in self.clean_example_dict['fvs'].items():
            if k in self.clean_example_dict['labels']:
                Y.append([0., 0.])
                Y[-1][self.clean_example_dict['labels'][k]] = 1.
            else:
                continue
            if X is None:
                X = x
            else:
                X = np.concatenate([X, x])
      
        model_input = torch.tensor(X).float().to(self.config.algorithm.device)
        model_input.requires_grad_()
        softmax = nn.Softmax(dim=1) 
        model_output = softmax(model(model_input).to(self.config.algorithm.device))
         
        loss = self.criterion(model_output, torch.tensor(Y).to(self.config.algorithm.device)) 
        loss.backward(retain_graph = True)

        attr = model_input.grad.data.detach()
         
        return attr
    
    def get_ig_attributes(self, model: Any, experiment: Dict[str, Any]):
 
        model = model.model.to(self.config.algorithm.device)
        model.zero_grad()

        X = experiment['inputs']
        baseline = experiment['baselines']
        
        model_input = torch.tensor(X).float().to(self.config.algorithm.device)
        model_input.requires_grad_()

        model_baseline = torch.tensor(baseline).float().to(self.config.algorithm.device)
       
        #logger.info(f"clean example tot: {len(self.clean_example_dict['fvs'].items())}")
        #logger.info(f"poisoned example tot: {len(self.poisoned_example_dict['fvs'].items())}")
        #logger.info(f"model input shape: {model_input.shape}")
        #logger.info(f"baseline shape: {model_baseline.shape}")
        #* torch.ones(X.shape).float().to(self.config.algorithm.device)

        softmax = nn.Softmax(dim=1) 
 
        ig = IntegratedGradients(lambda model_input: softmax(model(model_input)))
        attributions_0, approximation_error_0 = ig.attribute(model_input,
                                                        baselines=model_baseline,
                                                        method='gausslegendre',
                                                        return_convergence_delta=True,
                                                        target=0)
        attributions_1, approximation_error_1 = ig.attribute(model_input,
                                                        baselines=model_baseline,
                                                        method='gausslegendre',
                                                        return_convergence_delta=True,
                                                        target=1)
    
        attributions = np.concatenate((attributions_0.detach().cpu().numpy(), attributions_1.detach().cpu().numpy()), 1)
        return attributions
          
    def get_loss(self, cls, experiment = None):
        # Run the model to get the action 
        #logger.info(f'Original experience example {exps[0]}')
        def loss_fn(data):
            ## input is the inds of the selected model from a dataset
            ## Get the models
            nonlocal cls
            cls.train()

            
            models = self.target_model_indexer.get_model(data)
 
            ys = torch.tensor(data['poisoned']).to(self.config.algorithm.device).float()

            #logger.info(f"recons_loss: {recons_loss} kld_loss: {kld_loss}")
            tot_loss = 0
            if True:
                #logger.info(f"Run {len(models)} models")
                for i, (model, y) in enumerate(zip(models, ys)):
                    ## Run one model
                    #logger.info(f"{i}th model")
                    attr = torch.tensor(self.get_ig_attributes(model, experiment)).float().to(self.config.algorithm.device)
                    #logger.info(f"attribution shape {attr.shape}")
                    softmax = nn.Softmax(dim=1) 
                    pred = softmax(cls(attr)).to(self.config.algorithm.device)[:,1].mean(dim = 0, keepdims=True)
                    
         
                    #logger.info(f"Label {y} vs. Prediction {pred}")
                    loss = self.criterion(pred, torch.tensor([y]).to(self.config.algorithm.device)) 
                    #\logger.info(f'{i}th model: Error {errs}')

                    tot_loss += loss
                  
                tot_loss /= len(models)
            #logger.info(tot_loss)
            return tot_loss, {
                'tot_loss': loss.item()
            }
        return loss_fn
 
    def get_metrics(self, cls, experiment = None): 
        def metrics_fn(data):
            #logger.info(f"evaluation data {data}")

            ## input is the inds of the selected model from a dataset
            ## label indicates whether the model is poisoned
            nonlocal cls 
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
                ## Run one model
                #attr = self.get_attributes(model) 
                attr = torch.tensor(self.get_ig_attributes(model, experiment)).float().to(self.config.algorithm.device)
                softmax = nn.Softmax(dim=1) 
                pred = softmax(cls(attr)).to(self.config.algorithm.device)[:,1].mean(dim = 0, keepdims=True).item()
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
 
    
  
 

    def infer(self, model, experiment = None) -> List[float]:
        cls, experiment = self.get_detector(self.config.model.classifier.load_from_file)
        cls.model = cls.model.to(self.config.algorithm.device)
        cls.eval()

        #attr = self.get_ig_attributes(model, experiment) 
        attr = torch.tensor(self.get_ig_attributes(model, experiment)).float().to(self.config.algorithm.device)
        softmax = nn.Softmax(dim=1) 
        pred = softmax(cls(attr)).to(self.config.algorithm.device)[:,1].mean(dim = 0, keepdims=True).item()

        # Confidence equals the rate of false prediction
        conf = self.confidence(pred)
         
        logger.info("Trojan Probability: %f" % conf)
        
        return conf
    
        
    def save_detector(self, cls: Any,  info: Dict[Any, Any], experiment: Any = None):
        save_dict = {'model': self.config.model.classifier.name,
                    'state_dict': cls.model.state_dict()
                    }
        if experiment is not None:
            save_dict['experiment'] = experiment
            
        torch.save(save_dict, self.config.model.save_dir)
    
        

    def evaluate_detector(self):
        raise NotImplementedError 
    
  
    def run_detector(self, taget_path: str) -> float:
        raise NotImplementedError 