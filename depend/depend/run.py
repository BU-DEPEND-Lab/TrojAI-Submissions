from pydantic import BaseModel, PrivateAttr, field
from dataclasses import fields
from typing import Any, Dict, List, Literal, TypedDict, Union, cast, get_type_hints

from depend.core.dependent import Dependent
from depend.utils.configs import DPConfig, AlgorithmConfig, ModelConfig, OptimizerConfig, LearnerConfig
from depend.utils.data_management import DataManagement
import optuna
from optuna.trial import TrialState

class Sponsor(BaseModel):
    __model_schema__: Dict[str, Dict[Any]]
    __learner_schema__: Dict[str, Dict[Any]]
    __algorithm_schema__: Dict[str, Dict[Any]]
    __optimizer_schema__: Dict[str, Dict[Any]]
    
     
    @property
    def model_schema(self):
        return self.__model__schema__
    
    @property
    def learner_schema(self):
        return self.__learner_schema__
    
    @property
    def algorithm_schema(self):
        return self.__algorithm_schema__
    
    @property
    def optimizer_schema(self):
        return self.__optimizer_schema__
    
    @property
    def data_source(self):
        return self.__data_source__
    

    class Config:
        extra = "ignore"

    _dp_kwargs = PrivateAttr(default_factory=dict)
    
     
    def fund(
            self, 
            name: str, 
            result: str, 
            data_management: DataManagement,
            epochs: int = 1):
        
        model_config = self.model_config.from_dict(self.model_schema)
        algorithm_config = self.algorithm_config.from_dict(self.algorithm_schema)
        train_config = self.learner_config.from_dict(self.learner_schema)
        optimizer_config = self.optimizer_config.from_dict(self.optimizer_schema)

        dp_config = DPConfig(
            algorithm_config,
            model_config,
            optimizer_config,
            train_config
        )

        dependent = Dependent(
            data_management = data_management,
            experiment_name = name,
            result_dir = result,
            config = dp_config
        )  
        dependent.run(epochs)
    


class HyperSponsor(Sponsor): 
  
    def tune_config(self, trial):
        field_to_type = get_type_hints(ModelConfig)
        model_config = self.model_config.from_dict(
            {k: getattr(trial, f'suggest_{field_to_type(k)}')(vs[0], vs[1]) for k, vs in self.model_schema}
        )

        field_to_type = get_type_hints(AlgorithmConfig)
        algorithm_config = self.algorithm_config.from_dict(
            {k: getattr(trial, f'suggest_{field_to_type(k)}')(vs[0], vs[1]) for k, vs in self.algorithm_schema}
        )

        field_to_type = get_type_hints(LearnerConfig)
        train_config = self.learner_config.from_dict(
            {k: getattr(trial, f'suggest_{field_to_type(k)}')(vs[0], vs[1]) for k, vs in self.learner_schema}
        )

        field_to_type = get_type_hints(OptimizerConfig)
        optimizer_config = self.optimizer_config.from_dict(
            {k: getattr(trial, f'suggest_{field_to_type(k)}')(vs[0], vs[1]) for k, vs in self.optimizer_schema}
        ) 

        return DPConfig(
            algorithm_config,
            model_config,
            optimizer_config,
            train_config
        )

     
    def fund(
            self, 
            data_management: DataManagement,
            epochs: int = 1, 
            n_trials: int = 100,
            timeout: int = 600
            ):
        def objective(trial):
            nonlocal epochs, data_management
            trial_id = trial.number
            dp_config = self.tune_config(trial)
        
            dependent = Dependent(
                data_management = data_management,
                experiment_name = 'dependent_%d' % trial_id,
                result_dir = './optuna/%d' % trial_id,
                config = dp_config
            )  

            dependent.run(epochs)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        