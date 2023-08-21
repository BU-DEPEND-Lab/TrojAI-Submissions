"""Base interface for dependent module to expose."""
from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List, Tuple, Callable, Literal, TypedDict, Union, cast

from pydantic import BaseModel, PrivateAttr, model_validator

from depend.core.logger import Logger


import torch
import torch.nn as nn

from torch_ac.utils import ParallelEnv

from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)
 
 
class Agent(BaseModel):
    """A multi-model agent.
    """
    envs: ParallelEnv = ...
    acmodels: List[nn.Module] = ...
    preprocess_obss: Callable = ...
    logger: Logger = ...
    
    class Config:
        arbitrary_types_allowed = True  # Allow custom classes without validation
    
    @model_validator(mode='after')
    def on_create(cls, values):
        for acmodel in values.acmodels:
            acmodel.eval()
        logging.info(f"Run {len(values.acmodels)} models")
        
    @classmethod
    def collect_experience(
        cls, 
        envs: ParallelEnv = ...,
        acmodels: List[nn.Module] = ...,
        preprocess_obss: Callable = ...,
        logger: Logger = ...,
        num_frames_per_model: int = ...,
        ):
        agent = cls(
            envs = envs, 
            acmodels = acmodels, 
            preprocess_obss = preprocess_obss, 
            logger = logger
            )
        
        return agent.run(num_frames_per_model)


    def get_actions(self, acmodel: Callable, obss: List[Any]) -> Tuple[List[Any], List[nn.distributions]]:
        preprocessed_obss = self.preprocess_obss(obss)
        actions = []
        dists = []
        with torch.no_grad():
            #obs = preprocessed_obss[i]
            #logger.info(torch.tensor(obs).unsqueeze(0).shape)
            dists = acmodel(torch.tensor(obss))[0] 
            #dists.append(dist)
            actions = dists.sample().cpu().numpy().tolist()
            #logger.info(f"Actions: {actions}")
            #actions.append(action)
        return actions, dists
    
    def run(self, num_frames_per_model: int) -> torch.Tensor:
        # Prepare to store the experience items 
        exps = []
    
        # Reset the environment
        obss = self.envs.reset()

        #logging.info("Number of envs {}; \
        # number of obss {}".format(len(self.envs.envs), len(obss))
        # )

        # Start counting the frames
        for i, acmodel in enumerate(self.acmodels):
            logger.info(f"Run {i}th model")
            for num_frames in tqdm(range(num_frames_per_model), desc ="Agents Collecting Experience: "):
                # Store the observation in serialized manner
                exps = exps + [torch.tensor(obss)]
                # Get new actions and policy distribitions
                actions, dists = self.get_actions(acmodel, obss)
                
                # Get next observation
                obss, _, rewards, _ = self.envs.step(actions)
            
        # Turn observation list into a batch of observations
        exps = torch.cat(exps, dim=0)
        return exps
    