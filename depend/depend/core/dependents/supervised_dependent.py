
from pydantic import BaseModel, PrivateAttr, field
from dataclasses import fields
from typing import Any, Dict, List, Literal, TypedDict, Union, cast, get_type_hints

import depend.algorithms as algorithms
from depend.utils.configs import DPConfig
from depend.core.serializable import Serializable
from depend.core.loggers import Logger
from depend.core.dependents import Dependent

from abc import ABC, abstractmethod    

import torch.optim as optim


import mlflow
import mlflow.pytorch


class BaseSupervisedDependent(Dependent, ABC):
     
    def configure(
            self,
            epochs: int,
            config: DPConfig,
            experiment_name: str,
            result_dir: str
            ):
        
        
        
        
    
    def train_detector(
        self,
        
        ):
        with mlflow.start_run as run:
            for epoch in range(1, epochs + 1):
                self.logger.epoch_info("Run ID: %s, Epoch: %s \n" % (run.info.run_uuid, epoch))
                for k, v in self.learner.train(
                    logger
                    ): 
                    mlflow.log_metric(k, v, step = epoch)
                for k, v in self.algorithm.evaluate_detector():
                    mlflow.log_metric(k, v, step = epoch)
            mlflow.end_run()
            mlflow.log_artifacts(result_dir, artifact_path="configure_events")
     
    
    
     
     
    def infer(self, 
               detector_path: str, 
               target_paths: List[str]
                ):
        self.algorithm.get_detector(detector_path)
        with mlflow.start_run as run:
            for i, target_path in enumerate(target_paths):
                self.logger.epoch_info("Run ID: %s, Target: %s \n" % (run.info.run_uuid, target_path))
                for k, v in self.algorithm.detector.detect(target_path):
                    mlflow.log_metric(k, v, step = i)
            mlflow.end_run()
            mlflow.log_artifacts(self.logger.results_dir, artifact_path="infernece_events")
        