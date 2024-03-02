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

        self.build_experiments()


    def build_experiments(self):
        self.X = None
        for k, x in self.clean_example_dict['fvs'].items():
            if self.X is None:
                self.X = x
            else:
                self.X = np.concatenate([self.X, x])
        
        for k, x in self.poisoned_example_dict['fvs'].items():
            if self.X is None:
                self.X = x
            else:
                self.X = np.concatenate([self.X, x])
         
        self.baselines = np.random.randint(0, 255, size=[self.config.algorithm.num_experiments, *self.X.shape])
        
 
    def get_detector(self, path = None):
        channels = 2
        if self.config.algorithm.task == 'attr_cls_1':
            channels = 156

        cls = eval(self.config.model.classifier.name)(
            0, 
            channels, 
            config = {
            "cnn_type": 'ResNet18',
            "num_classes": 2,
            "img_resolution": 28 
        })#.to(self.config.algorithm.device)
        experiment = None

        if path or self.config.model.classifier.load_from_file:
            if not path:
                path = self.config.model.classifier.load_from_file
            #self.config.model.classifier.load_from_file:
            logger.info(f"Load model from {path}")
            stored_dict = torch.load(path, \
                                    map_location=self.config.algorithm.device)
            cls.load_state_dict(stored_dict['state_dict'])
            if 'experiment' in stored_dict:
                experiment = stored_dict['experiment']
                logger.info(f"Loaded experiment from {path}")
        cls.model = cls.model.to(self.config.algorithm.device)

        return cls, experiment
    
    def get_experiment(self, i_exp: int):
        return {
            'inputs': self.X,
            'baselines': self.baselines[i_exp]
        }
         
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
        if self.config.algorithm.task == 'attr_cls_2':
            attributions = np.concatenate([attributions_0.detach().cpu().numpy(), attributions_1.detach().cpu().numpy()], axis = 1) 
            return attributions
        elif self.config.algorithm.task == 'attr_cls_1':
            attributions_0 = np.concatenate([attribution for attribution in attributions_0.detach().cpu().numpy()], axis = 0)
            attributions_1 = np.concatenate([attribution for attribution in attributions_1.detach().cpu().numpy()], axis = 0)
            return np.concatenate([attributions_0, attributions_1], axis=0)
          
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
            if self.config.algorithm.task == 'attr_cls_2':
                tot_loss = 0
                for i, (model, y) in enumerate(zip(models, ys)):
                    attrs = self.get_ig_attributes(model, experiment)
                    attrs = torch.tensor(attrs).float().to(self.config.algorithm.device)
                    softmax = nn.Softmax(dim=1) 
                    preds = softmax(cls(attrs))[:,1]
                    #logger.info(f'preds size {preds.shape}')
                    pred = (torch.sum(preds * torch.exp(preds * 10)) / torch.sum(torch.exp(preds * 10)))
                    #logger.info(f'softmax pred size {pred.shape}')
                    tot_loss += self.criterion(pred, y).to(self.config.algorithm.device)
                tot_loss /= len(ys)
            elif self.config.algorithm.task == 'attr_cls_1':
                #logger.info(f"recons_loss: {recons_loss} kld_loss: {kld_loss}")
                attrs = None
                #logger.info(f"Run {len(models)} models")
                for i, (model, y) in enumerate(zip(models, ys)):
                    ## Run one model
                    #logger.info(f"{i}th model")
                    attr = np.expand_dims(self.get_ig_attributes(model, experiment), axis = 0)
                    if attrs is None:
                        attrs = attr
                    else:
                        attrs = np.concatenate([attrs, attr])
                
            
                #logger.info(f"attribution shape {attrs.shape}")
                attrs = torch.tensor(attrs).float().to(self.config.algorithm.device)

                softmax = nn.Softmax(dim=1) 
                pred = softmax(cls(attrs)).to(self.config.algorithm.device)[:,1].mean(dim = 0, keepdims=True)
                
        
                #logger.info(f"Label {y} vs. Prediction {pred}")
                tot_loss = self.criterion(pred, torch.tensor([y]).to(self.config.algorithm.device)) 
                #\logger.info(f'{i}th model: Error {errs}')
                
                
            #logger.info(tot_loss)
            return tot_loss, {
                'tot_loss': tot_loss.item()
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
                if self.config.algorithm.task == 'attr_cls_2':
                    attrs = torch.tensor(self.get_ig_attributes(model, experiment)).float().to(self.config.algorithm.device)
                    softmax = nn.Softmax(dim=1) 
                    preds = softmax(cls(attrs))[:,1]
                    #logger.info(f'preds size {preds.shape}')
                    pred = ((torch.sum(preds * torch.exp(preds * 1.e3)) / torch.sum(torch.exp(preds * 1.e3)))).detach().cpu().numpy().item()

                elif self.config.algorithm.task == 'attr_cls_1':
                    #attr = self.get_attributes(model) 
                    attr = torch.tensor(self.get_ig_attributes(model, experiment)).float().to(self.config.algorithm.device).unsqueeze(0)
                    softmax = nn.Softmax(dim=1) 
                    pred = softmax(cls(attr))[:,1].mean(dim = 0, keepdims=True).item()
                    
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
 
        if self.config.algorithm.task == 'attr_cls_2':
            attrs = self.get_ig_attributes(model, experiment)
            attrs = torch.tensor(attrs).float().to(self.config.algorithm.device)
            softmax = nn.Softmax(dim=1) 
            preds = softmax(cls(attrs))[:,1]
            #print(((torch.sum(preds * torch.exp(preds * 10)) / (torch.sum(torch.exp(preds * 10))))).detach().cpu().numpy().item())
            #logger.info(f'preds size {preds.shape}')
            pred = ((torch.sum(preds * torch.exp(preds * 1.e3)) / (torch.sum(torch.exp(preds * 1.e3))))).detach().cpu().numpy().item() #max(preds).item() #
            
            pred_mask = torch.isnan(pred)  
            if pred_mask.item():
                pred = max(preds).item()

            #pred = 1 if pred > 0.51 else 0.
        elif self.config.algorithm.task == 'attr_cls_1':
            #attr = self.get_attributes(model) 
            attr = torch.tensor(self.get_ig_attributes(model, experiment)).float().to(self.config.algorithm.device)
            softmax = nn.Softmax(dim=1) 
            pred = softmax(cls(attr)).to(self.config.algorithm.device)[:,1].mean(dim = 0, keepdims=True)
         
        # Confidence equals the rate of false prediction
        conf = self.confidence(pred)  
        
        logger.info("Trojan Probability: %f" % conf)
        
        return conf
    
 
        ''' 
        attr = torch.tensor(self.get_ig_attributes(model, experiment)).float().to(self.config.algorithm.device).unsqueeze(0)
        softmax = nn.Softmax(dim=1) 
        pred = softmax(cls(attr)).to(self.config.algorithm.device)[:,1].mean(dim = 0, keepdims=True).item()
        '''
        
    def save_detector(self, cls: Any,  info: Dict[Any, Any], experiment: Any = None, path = None):
        save_dict = {'model': self.config.model.classifier.name,
                    'state_dict': cls.model.state_dict()
                    }
        save_dict.update({'info': info})
        if experiment is not None:
            save_dict['experiment'] = experiment

        if path is None:               
            torch.save(save_dict, self.config.model.save_dir)
        else:
            torch.save(save_dict, path)
        

    def evaluate_detector(self):
        raise NotImplementedError 
    
  
    def run_detector(self, taget_path: str) -> float:
        raise NotImplementedError 