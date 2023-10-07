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
class AttributionClassifier(Dependent):
    """
    ########## Attribution Classifier ############
    # Build a dataset of model indices and model labels
    # After sampling a batch of (model, label), for each model
    #   1. Get action_distribution = model(self.exp)
    #   3. Get attribute = d action_distribution/d self.exp
    #   2. run the prediction = classifier(attribute)
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
     
    
    
    
    def get_model_output_from_observation(self, model, obs, requires_grad = False):
        x = obs.transpose(2, 3).transpose(1, 3).float()
        model = model.to(self.config.algorithm.device)

        if requires_grad:
            model.requires_grad_()

        x = x.reshape(x.shape[0], -1)
        dist = model(x)[0]
         
        return dist
        
    
    def get_attribute_from_image(self, model, obs):
        model = model.to(self.config.algorithm.device)
        for k, v in obs.items():
            obs[k] = obs[k].float().detach()
            obs[k].requires_grad_()
            #logger.info(obs[k])
        model.zero_grad()
        dist, _ = model(obs['image'].transpose(2, 3).transpose(1, 3).float())
        dist.entropy().sum().backward(retain_graph = True)
        #logger.info(obs['direction'].grad)
         
        norm = lambda attrs: (attrs  - attrs.min()) / (attrs.max() - attrs.min() + 1.e-5)
        return {k: v.grad.data if k == 'image' else v for k, v in obs.items()}
    
    
 
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
                logger.info(f"Run {len(models)} models")
                for i, (model, y) in enumerate(zip(models, ys)):
                    ## Run one model
                    #logger.info(f"{i}th model")
                    attrs = self.get_attribute_from_image(model, exps) 
                    inputs = {k: attrs[k].detach() if k == 'image' else v for k, v in exps.items()}
                    pred = cls(inputs).mean(dim = 0)
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
                attrs = self.get_attribute_from_image(model, exps) 
                inputs = {k: attrs[k].detach() if k == 'image' else v for k, v in exps.items()}
                pred = cls(inputs).mean(dim = 0).item()
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
 
    def train_detector(self, final_train: bool = True):
        # Run the agent to get experiences
        # Build model dataset 
        # K-split the model dataset and train the detector for multiple rounds 
        # Return the mean metric 
        dataset = self.build_dataset(
            num_clean_models = self.config.data.max_models // 2,
            num_poisoned_models = self.config.data.max_models - self.config.data.max_models // 2)

        best_score = None
        best_exps = None
        best_loss_fn = None
        best_validation_info = None
        best_dataset = None
        #with mlflow.start_run as run:
        for _ in range(self.config.algorithm.num_experiments):
            # Run agent to get a dataset of environment observations
             
            best_score = None
            best_exps = None

            

            suffix_split = DataSplit.Split(dataset, self.config.data.num_splits)
            prefix_split = None
            for split in range(1, self.config.data.num_splits):
                
                # Split dataset
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

                exps = self.collect_experience(
                    num_clean_models = self.config.algorithm.num_procs // 2,
                    num_poisoned_models = self.config.algorithm.num_procs - self.config.algorithm.num_procs // 2
                )
                logger.info(exps['image'].shape)
 
                 
                # Prepare the mask generator
                cls = eval(self.config.model.classifier.name)(
                    obs_space = self.envs[0].observation_space).to(self.config.algorithm.device)
                    
                if self.config.model.classifier.load_from_file:
                    cls.load_state_dict(torch.load(self.config.model.classifier.load_from_file))    
                    cls = cls.to(self.config.algorithm.device)

                cls.train()

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
                if best_score is None or best_score < score:
                    #logger.info("New best model")
                    best_score, best_exps, best_validation_info, best_dataset, best_loss_fn = score, exps, validation_info, dataset, loss_fn
                    self.save_detector(cls, best_exps, best_validation_info)
                    if not self.config.algorithm.k_fold:
                        break
        if final_train:
            logger.info("Final train the best detector")
            final_info = self.learner.train(self.logger, best_dataset, best_loss_fn, optimize_fn, best_dataset, metrics_fn)
                #for k, v in final_info.items():
                #    mlflow.log_metric(k, v, step = self.config.data.num_splits + 1)
            #mlflow.end_run()
            #mlflow.log_artifacts(self.logger.results_dir, artifact_path="configure_events")

        self.save_detector(cls, best_exps, best_validation_info)
        return best_score
 
    def get_stats(self, exps, model):
        obss = exps['image']
        new_exps = self.collect_experience(models = [model] * self.config.algorithm.num_procs)
        new_obss = new_exps['image']
        print(obss)
        print(new_obss)
        

    def infer(self, model, get_stats = False) -> List[float]:
        # Prepare the mask generator
        exps = pickle.load(open(self.config.algorithm.load_experience, 'rb'))
        if get_stats:
            self.get_stats(exps, model)
        for k, v in exps.items():
            exps[k] = v.to(self.config.algorithm.device).float()
        #logger.info(exps)
        cls = eval(self.config.model.classifier.name)(
            obs_space = gymnasium.spaces.Box(0, 255, (3, 7, 7), dtype=np.uint8),
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
        model.zero_grad()
        exps['image'].requires_grad_()
        # Models make predictions on the masked inputs
        #preds = self.get_model_entropy_from_image(model, exps['image']) 
        entropy = Categorical(logits=F.log_softmax(model(exps), dim=1)).entropy().mean()
        entropy.backward(retain_graph = True)
        #logger.info(obs['direction'].grad)
        norm = lambda attrs: (attrs  - attrs.min()) / (attrs.max() - attrs.min() + 1.e-5)
        inputs = {k: v.grad.data if k == 'image' else v for k, v in exps.items()}
        preds = cls(inputs).detach().cpu().numpy()
        #preds = (preds > 0.5)
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
                    'obs_space': self.envs[0].observation_space,
                    'state_dict': cls.state_dict()
                    }, self.config.model.save_dir)
        
        with open(os.path.join(self.logger.results_dir, 'best_attr_experience.p'), 'wb') as fp:
            pickle.dump(exps, fp)
        torch.save({'model': self.config.model.classifier.name,
                    'obs_space': self.envs[0].observation_space,
                    'state_dict': cls.state_dict()
                    }, os.path.join(self.logger.results_dir, self.config.model.save_dir))
        #self.logger.log_numpy(example = exps.cpu().numpy(), **info) 
        

    def evaluate_detector(self):
        raise NotImplementedError 
    
  
    def run_detector(self, taget_path: str) -> float:
        raise NotImplementedError 