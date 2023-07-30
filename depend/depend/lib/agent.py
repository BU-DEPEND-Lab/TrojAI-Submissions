"""Base interface for dependent module to expose."""
from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List, Tuple, Callable, Literal, TypedDict, Union, cast

from pydantic import BaseModel, PrivateAttr

from depend.core.loggers import Logger
from depend.lib.utils.format import get_obss_preprocessor

import torch
import torch.nn as nn

from torch_ac.utils.penv import DictList, ParallelEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent(BaseModel):
    """A multi-model agent.
    """
    envs: ParallelEnv = ...
    acmodels: List[nn.Module] = ...
    proprocess_obss: Callable = ...
    logger: Logger = ...
 

    def __pos_init__(self):
        for acmodel in self.acmodels:
            acmodel.to(device)
            acmodel.eval()
         
    def get_actions(self, obss: List[Any]) -> Tuple[List[Any], List[nn.distributions]]:
        preprocessed_obss = self.preprocess_obss(obss, device=device)
        actions = []
        dists = []
        with torch.no_grad():
            for i, acmodel in enumerate(self.acmodels):
                dist, _ = acmodel(preprocessed_obss[i])
                dists.append(dist)
                action = dist.sample().cpu().numpy().item()
                actions.append(action)
        return actions
    
    def run(self, num_frames_per_model: int) -> torch.Tensor:
        # Prepare to store the experience items 
        exps = []
    
        # Reset the environment
        obss = self.envs.reset()
        
        # Start counting the frames
        num_frames = 0
        while num_frames < num_frames_per_model:
            # Store the observation in serialized manner
            exps = exps + obss
            # Get new actions and policy distribitions
            actions, _ = self.agent.get_actions(obss)
            
            # Get next observation
            obss, _, _, _, _ = self.envs.step(actions)
 
        # Turn observation list into a batch of observations
        exps = torch.stack(exps, dim=0)
        return exps
    