
from pydantic import BaseModel, PrivateAttr, field
from dataclasses import fields
from typing import Any, Dict, List, Literal, TypedDict, Union, cast, get_type_hints

import depend.core.learners as learners
import depend.algorithms as algorithms
import depend.models as models
from depend.core.loggers import Logger
from depend.utils.configs import DPConfig
from depend.utils.data_management import DataManagement

from abc import ABC, abstractmethod    

import torch.optim as optim


import mlflow
import mlflow.pytorch


class Dependent(ABC):
    def __init__(
            self,
            data_management: DataManagement,
            experiment_name: str,
            results_dir: str,
            config: DPConfig
    ): 
        model = getattr(models, config.model.name)(**config.model.kwargs)
        optimizer = getattr(optim, config.optimizer.name)(**config.optimizer.kwargs)
        learner = getattr(learners, config.learner.name)(**config.learner.kwargs)
        logger = Logger(
            log_name=experiment_name, 
            results_dir=results_dir, 
            log_console=False,
            use_timestamp=False, 
            append=False, 
            seed=None
        )
        algorithm = getattr(algorithms, config.algorithm.name)
        self.algorithm = algorithm(
            data_management,
            model, 
            optimizer,
            learner,
            logger,
            **config.algorithm.kwargs)


    def run(self, epochs = 1):
        with mlflow.start_run as run:
            for k, v in self.train_kwargs.items():
                mlflow.log_param(k, v)
            for epoch in range(1, epochs + 1):
                    self.logger.epoch_info("Run ID: %s, Epoch: %s \n" % (run.info.run_uuid, epoch))
                    train_info = self.algorithm.train(
                         self.train_loader,
                         self.loss,
                         self.optimize, 
                         self.logger
                         )
                    for k, v in train_info:
                        mlflow.log_metric(k, v, step = epoch)
                    evaluation_info = self.algorithm.evaluate(
                         self.evaluation_loader,
                         self.metric,
                         self.logger)
                    for k, v in evaluation_info:
                        mlflow.log_metric(k, v, step = epoch)
            mlflow.end_run()
            mlflow.log_artifacts(self.traier.logging_dir, artifact_path="events")