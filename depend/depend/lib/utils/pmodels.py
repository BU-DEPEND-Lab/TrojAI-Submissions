import multiprocessing
import gym
import torch
import torch.nn as nn
from typing import List, Callable, Any

from pydantic import BaseModel, model_validator

import logging
logger = logging.getLogger(__name__)

multiprocessing.set_start_method("fork")

class ParallelModel(BaseModel):
    """A concurrent execution of environments in multiple processes."""
    models: List[Callable] = ...
    device: Any = ...
    locals: List[Any] = []

    @model_validator(mode='after')
    def on_create(cls, values):
        assert len(values.models) >= 1, "No model given."
        #ogging.info(f"Run {len(values.acmodels)} models")
        for model in values.models:
            model.to(values.device)
            model.eval()
        #logging.info(f"Run {len(values.models)} models")
         
    
    @classmethod
    def create(cls, models: List[Callable], device: Any):
        def worker(conn, model):
            while True:
                cmd, data = conn.recv()
                assert cmd == "get_action"
                dist, _ = model(torch.tensor(data).unsqueeze(0))
                action = dist.sample().cpu().numpy().item()
                conn.send((dist, action))
        obj = cls(models = models, device = device)
 
        for model in obj.models[1:]:
            local, remote = multiprocessing.Pipe()
            obj.locals.append(local)
            p = multiprocessing.Process(target=worker, args=(remote, model))
            p.daemon = True
            p.start()
            remote.close()
        #logger.info(obj)
        return obj

    def get_action(self, obss: List[torch.Tensor]):
        for local, obs in zip(self.locals, obss[1:]):
            local.send(("get_action", obs))
        dist, _ = self.models[0](torch.tensor(obss[0]).unsqueeze(0))
        action = dist.sample().cpu().numpy().item()
        results = zip(*[(dist, action)] + [local.recv() for local in self.locals])
        return results
    
    def __len__(self):
        return len(self.models)