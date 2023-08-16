import json
import logging
import pickle
from os import listdir 
from os.path import join 
import numpy as np
from typing import Any, Dict, List, Literal, TypedDict, Union, cast, get_type_hints
from pydantic import BaseModel, PrivateAttr, Field
#from sklearn.ensemble import RandomForestRegressor
  
from depend.utils.configs import AlgorithmConfig
import importlib
import captum 
import torch
import torch.nn as nn
 
  
logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        )

class AttributionExtractor(BaseModel):
 
    """Detector initialization function.

    Args:
        input_size: str - model input size.
    """

    config: AlgorithmConfig  
    input_size: List[int] 

    # TODO: Update skew parameters per round
 
    def __post_init__(self):
        if self.config.name == 'IntegratedGradients':
            from captum.attr import IntegratedGradients
            self.attributor = IntegratedGradients
            self.baseline = self.config.baseline * torch.tensor(*self.input_size)
            self.method = self.config.method
    
    def get_attribution_from_one_model(
            self, 
            model: nn.Module,
            inputs: List[torch.Tensor]
    ):
        attributions, _ = self.attributor(model).attribute(
            inputs,
            baselines = [self.baseline] * len(inputs),
            method = self.method,
            return_convergence_delta = True
        )
        return attributions