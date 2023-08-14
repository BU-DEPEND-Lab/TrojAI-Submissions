from pydantic import BaseModel, PrivateAttr, field
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, TypedDict, Union, cast, get_type_hints

from depend.core.dependents import Dependent
from depend.utils.configs import DPConfig, AlgorithmConfig, ModelConfig, OptimizerConfig, LearnerConfig, DataConfig
import optuna
from optuna.trial import TrialState

@dataclass
class Sponsor:
    model_schema: Dict[str, Dict[Any]]
    learner_schema: Dict[str, Dict[Any]]
    algorithm_schema: Dict[str, Dict[Any]]
    optimizer_schema: Dict[str, Dict[Any]]
    data_schema: Dict[str, Dict[Any]]
     
    def fund(
            self, 
            dependent: Dependent,
            experiment_name: str, 
            result_dir: str, 
            epochs: int = 1):
        
        model_config = self.model_config.from_dict(self.model_schema)
        algorithm_config = self.algorithm_config.from_dict(self.algorithm_schema)
        train_config = self.learner_config.from_dict(self.learner_schema)
        optimizer_config = self.optimizer_config.from_dict(self.optimizer_schema)
        data_config = self.data_config.from_dict(self.data_schema)

        dp_config = DPConfig(
            algorithm_config,
            model_config,
            optimizer_config,
            train_config,
            data_config
        )
       
        dependent.configure(
            epochs = epochs,
            config = dp_config,
            experiment_name = experiment_name,
            result_dir = result_dir
            )

        return dependent.train_detector() 


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

        field_to_type = get_type_hints(DataConfig)
        data_config = self.optimizer_config.from_dict(
            {k: getattr(trial, f'suggest_{field_to_type(k)}')(vs[0], vs[1]) for k, vs in self.data_schema}
        ) 

        return DPConfig(
            algorithm_config,
            model_config,
            optimizer_config,
            train_config,
            data_config,
        )

     
    def fund(
            self, 
            dependent: Dependent,
            experiment_name: str, 
            result_dir: str, 
            epochs: int = 1,
            n_trials: int = 100,
            timeout: int = 600
            ):
        def objective(trial):
            nonlocal epochs, dependent
            trial_id = trial.number
            dp_config = self.tune_config(trial)
        
            dependent.configure(
                config = dp_config,
                experiment_name = f'{experiment_name}_%d' % trial_id,
                result_dir = '%s/result_dir/optuna/%d' % (result_dir, trial_id)
            )
            return dependent.train_detector() 

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        