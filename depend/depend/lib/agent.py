"""Base interface for dependent module to expose."""
from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List, Tuple, Callable, Literal, TypedDict, Union, cast, Optional

from pydantic import BaseModel, PrivateAttr, model_validator, Extra

from depend.core.logger import Logger
from depend.utils.format import get_obss_preprocessor

import random

import torch
import torch.nn as nn

from torch_ac.utils import ParallelEnv


from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)

import multiprocessing

import torch.multiprocessing as mp


      

class Worker(mp.Process):
    def __init__(
            self, 
            env: Any = ...,
            acmodel: Callable = ... ,
            exploration_rate: float = ...,
            num_frames_per_model: int = ...,
            exps_queue: Any = ...,
        ):
        super(Worker, self).__init__()
        self.env = env
        self.acmodel = acmodel
        self.exploration_rate = exploration_rate
        self.num_frames = num_frames_per_model
        self.preprocess_obs = get_obss_preprocessor(env.observation_space)[1]
        self.exps_queue = exps_queue

    def run(self):
        
        obs = self.reset()
        #logging.info("Number of envs {}; \
        # number of obss {}".format(len(self.envs.envs), len(obss))
        # )

        # Start counting the frames
        for num_frames in tqdm(range(self.num_frames), desc ="Agents Collecting Experience: "):
            # Store the observation in serialized manner
            preprocessed_obss = torch.tensor(self.preprocess_obs(obs)).unsqueeze(0)
            self.exps_queue.put(preprocessed_obss)
            dist, _ = self.acmodel(preprocessed_obss)
            exploration = int(self.exploration_rate > random.random())
            action = dist.sample().cpu().numpy().item() * exploration + \
                random.randint(0, len(dist.probs)) * (1 - exploration)
            obs, reward, terminated, info = self.env.step(action)
            if terminated:
                obs = self.env.reset()
        self.exps_queue.put(None)

    def reset(self):
        return self.env.reset()
         

 
class ParallelAgent(BaseModel):
    """A multi-model agent.
    """
    workers: List[mp.Process] = ...
    logger: Logger = ...
    
    class Config:
        arbitrary_types_allowed = True  # Allow custom classes without validation
        extra = Extra.allow
 
         
    @classmethod
    def collect_experience(
        cls, 
        envs: List[Any] = ...,
        acmodels: List[Callable] = ...,
        logger: Logger = ...,
        num_frames_per_model: int = ...,
        exploration_rate: Optional[float] = ...,
        device: Any = ...
        ):
        exps_queue = mp.Queue()
        workers = []
        for (env, acmodel) in zip(envs, acmodels):
            acmodel = acmodel.cpu()
            acmodel.eval()
            workers.append(Worker(env, acmodel, exploration_rate, num_frames_per_model, exps_queue))
        
        agent = cls(
            workers = workers,
            logger = logger
        )
        [w.start() for w in agent.workers] 
        exps = []
        exps_queue
        while True:
            exp = exps_queue.get()
            if exp is not None:
                exps.append(exp)
            else:
                break

        [w.join() for w in agent.workers] 
         
        exps = torch.cat(exps, dim=0).to(device)
        return exps
  



def worker(conn, env, model):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            action = None
            dist, _ = model(data[0])
            action = dist.sample().cpu().numpy().item() * data[1] + random.randint(0, len(dist.probs)) * data[1]
            obs, reward, terminated, info = env.step(action)
            if terminated:
                obs = env.reset()
            conn.send((obs, action, reward, terminated, info))
        elif cmd == "reset":
            obs = env.reset()
            conn.send(obs)
        else:
            raise NotImplementedError


 
class Agent(BaseModel):
    """A multi-model agent.
    """
    envs: List[Any] = ...
    acmodels: List[Callable] = ... 
    num_frames_per_model: int = ...
    logger: Logger = ...
    exploration_rate: Optional[float] = -1.0
    device: Any = ...
   
    
    class Config:
        arbitrary_types_allowed = True  # Allow custom classes without validation
        extra = Extra.allow

    @model_validator(mode='after')
    def on_create(cls, values):
        assert len(values.acmodels) >= 1, "No model given."
        #ogging.info(f"Run {len(values.acmodels)} models")
        for model in values.acmodels:
            model.cpu()
            model.eval()
        logging.info(f"Run {len(values.acmodels)} models")
         
    

    @classmethod
    def collect_experience(
        cls, 
        envs: List[Any] = ...,
        acmodels: List[Callable] = ...,
        logger: Logger = ...,
        num_frames_per_model: int = ...,
        exploration_rate: Optional[float] = ...,
        device: Any = ...
        ):
        
        agent = cls(
            envs = envs,
            acmodels = acmodels,
            num_frames_per_model = num_frames_per_model,
            logger = logger,
            exploration_rate = exploration_rate,
            device = device
            )
       
        _, preprocess_obss = get_obss_preprocessor(agent.envs[0].observation_space)
        
        """
        logging.info(f"Run {len(acmodels)} models")
        agent.locals = []
        ps = []
        for (env, model) in zip(agent.envs[1:], agent.acmodels[1:]):
            local, remote = multiprocessing.Pipe()
            agent.locals.append(local)
            p = multiprocessing.Process(target=worker, args=(remote, env, model))
            p.daemon = True
            p.start()
            ps.append(p)
            remote.close()
        """
        exps = agent.run(preprocess_obss)
         
        return exps
        
    
    def reset(self):
        obss = []
        for env in self.envs:
            obss.append(env.reset())
        return obss

    def step(self, obss):
        next_obss = []
        for i, obs in enumerate(obss):
            dist, _ = self.acmodels[i](obss[i].unsqueeze(0))
            action = dist.sample().cpu().numpy().item() * int(random.random() < self.exploration_rate) + \
                random.randint(0, len(dist.probs)) * int(random.random() >= self.exploration_rate)
       
            obs, reward, done, info = self.envs[i].step(action)
            if done:
                next_obss.append(self.envs[i].reset()) 
            else:
                next_obss.append(obs) 
        return next_obss

    def render(self):
        raise NotImplementedError

    def get_actions(self, obss: List[Any]) -> Tuple[List[Any], List[nn.distributions]]:
        preprocessed_obss = self.preprocess_obss(obss)
        
        dists = []
        actions = []
        with torch.no_grad():
            dists, actions = self.acmodels.get_action(preprocessed_obss)
         
        #logger.info(f"Actions: {actions}")
        #actions.append(action)
        return actions, dists
    
    def run(self, preprocess_obss: Callable = ...) -> torch.Tensor:
        exps = []

        # Reset the environment
        obss = self.reset()

        #logging.info("Number of envs {}; \
        # number of obss {}".format(len(self.envs.envs), len(obss))
        # )

        # Start counting the frames
        for num_frames in tqdm(range(self.num_frames_per_model), desc ="Agents Collecting Experience: "):
            # Store the observation in serialized manner
            preprocessed_obss = torch.tensor(preprocess_obss(obss))
            exps = exps + [preprocessed_obss]
            # Get new actions and policy distribitions
            obss = self.step(preprocessed_obss)
             
        # Turn observation list into a batch of observations
        exps = torch.cat(exps, dim=0).to(self.device)
        return exps
    
 