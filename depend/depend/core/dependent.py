
from pydantic import BaseModel, PrivateAttr, field
from dataclasses import fields
from typing import Any, Dict, List, Literal, TypedDict, Union, cast, get_type_hints

from depend.utils.configs import DPConfig, AlgorithmConfig, ModelConfig, OptimizerConfig, TrainConfig
import depend.core.trainers as trainers
import depend.algorithms as algorithms
import depend.models as models

from depends.core.loggers import Logger

import torch.optim as optim


import mlflow
import mlflow.pytorch

import optuna
from optuna.trial import TrialState

class Dependent:
    def __init__(
            self,
            experiment_name: str,
            results_dir: str,
            config: DPConfig
    ):
         
        self.algorithm = getattr(algorithms, config.algorithm.name)(**config.algorithm.kwargs)
        self.model = getattr(models, config.model.name)(**config.model.kwargs)
        self.optimizer = getattr(optim, config.optimizer.name)(**config.optimizer.kwargs)
        self.trainer = getattr(trainers, config.train.name)(**config.train.kwargs)
        self.logger = Logger(
            log_name=experiment_name, 
            results_dir=results_dir, 
            log_console=False,
            use_timestamp=False, 
            append=False, 
            seed=None
        )

    def run(self, epochs = 1):
        with mlflow.start_run as run:
            for k, v in self.train_kwargs.items():
                mlflow.log_param(k, v)
            for epoch in range(1, epochs):
                    self.logger.epoch_info("Run ID: %s, Epoch: %s \n" % (run.info.run_uuid, epoch))
                    train_info = self.trainer.train(epoch)
                    for k, v in train_info:
                        mlflow.log_metric(k, v, step = epoch)
                    evaluation_info = self.trainer.evaluate(epoch)
                    for k, v in evaluation_info:
                        mlflow.log_metric(k, v, step = epoch)
            mlflow.end_run()
            mlflow.log_artifacts(self.traier.logging_dir, artifact_path="events")


class Sponsor(BaseModel):
    __model_schema__: Dict[str, List[Any]]
    __trainer_schema__: Dict[str, List[Any]]
    __algorithm_schema__: Dict[str, List[Any]]
    __optimizer_schema__: Dict[str, List[Any]]
    
    

    @property
    def model_schema(self):
        return self.__model__schema__
    
    @property
    def trainer_schema(self):
        return self.__trainer_schema__
    
    @property
    def algorithm_schema(self):
        return self.__algorithm_schema__
    
    @property
    def optimizer_schema(self):
        return self.__optimizer_schema__
    

    class Config:
        extra = "ignore"

    _dp_kwargs = PrivateAttr(default_factory=dict)

    def run(self, epochs = 1):
        self.epochs = epochs
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=100, timeout=600)
  
    def create_config(self, trial):
        field_to_type = get_type_hints(ModelConfig)

        model_kwargs = {}
        for k, vs in self.model_schema:
            model_kwargs[k] = getattr(trial, f'suggest_{field_to_type(k)}')(vs[0], vs[1])
        model_config = self.model_config.from_dict(model_kwargs)

        algorithm_kwargs = {}
        for k, vs in self.algorithm_schema:
            algorithm_kwargs[k] = getattr(trial, f'suggest_{field_to_type(k)}')(vs[0], vs[1])
        algorithm_config = self.algorithm_config.from_dict(algorithm_kwargs)

        train_kwargs = {}
        for k, vs in self.trainer_schema:
            train_kwargs[k] = getattr(trial, f'suggest_{field_to_type(k)}')(vs[0], vs[1])
        train_config = self.trainer_config.from_dict(train_kwargs)

        optimizer_kwargs = {}
        for k, vs in self.schema:
            optimizer_kwargs[k] = getattr(trial, f'suggest_{field_to_type(k)}')(vs[0], vs[1])
        optimizer_config = self.optimizer_config.from_dict(optimizer_kwargs)
    
        return DPConfig(
            algorithm_config,
            model_config,
            optimizer_config,
            train_config
        )

    def objective(self, trial):
        trial_id = trial.number
        dp_config = self.create_config(trial)
        dependent = Dependent(
            experiment_name = 'dependent_%d' % trial_id,
            result_dir = './optuna/%d' % trial_id,
            config = dp_config
        )  

        dependent.run(self.epochs)