"""Base interface for dependent module to expose."""
from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List, Callable, Literal, TypedDict, Union, cast

from pydantic import BaseModel, PrivateAttr

from depend.core.logger import Logger

import torch
import torch.nn as nn
from torch_ac.utils.penv import ParallelEnv
from depend.lib.agent import Agent
from depend.utils.env import make_env
import pydantic 




class ExperienceCollector(BaseModel):
    env: str = ...
    preprocess_obss: Callable = ...
    agent: Agent
    logger: Logger = ...
    seed: int = ...


    def __post_init__(self):
        envs = []
        for i in range(len(self.agent.acmodels)):
            env = make_env(self.env, self.seed + 10000 * i)
            envs.append(env)
        self.env = ParallelEnv(envs)
         

    def run(self, num_frames_per_model):
        obs = self.env.reset()
        obss = [obs]
        num_frames = 0
        while num_frames < num_frames_per_model:
            action = self.agent.get_actions(obs)
            obs, _, _, _, _ = self.env.step(action)
            obss.append(obs)
            num_frames += 1
 
        exps = [obss[i][j] 
                for j in range(len(self.env))
                for i in range(num_frames_per_model)]
        exps = self.preprocess_obss(exps)
        return exps
    