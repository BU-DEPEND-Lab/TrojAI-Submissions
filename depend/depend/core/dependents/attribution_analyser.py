
from pydantic import BaseModel, PrivateAttr, field
from dataclasses import fields
from typing import Any, Dict, List, Literal, TypedDict, Union, cast, get_type_hints

from depend.algorithms import AttributionExtractor
from depend.utils.configs import DPConfig
from depend.core.serializable import Serializable
from depend.core.loggers import Logger
from depend.core.dependents import Dependent

from abc import ABC, abstractmethod    

import torch.optim as optim



import mlflow
import mlflow.pytorch


def make_agent(env,)

class AttributionDependent(Dependent, ABC):
     
    def configure(
            self,
            epochs = 1,
            config: DPConfig = ...,
            experiment_name: str = ...,
            result_dir: str = ...
            ):
        feature_size = self.config.model
        self.attributor = AttributionExtractor(
            config.model.detector.input_size,
            config.algorithm.attribtuion,
            )
        if self.config.algorithm.task == 'RL':
            import trojai_rl
            self.make_envs()
            self.make_agents()
            self.model_sample_dict = self.collect_experiences()
        
            envs = self.clean_example_dict['fvs']
    
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
        