import json
import logging
import os
import pickle

from typing import List, Union

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import torch

from utils.abstract import AbstractDetector
from utils.models import load_model




class Trigger:
    """class to search for a trigger.

        attrs:
            model: the llm model
            tokenizer: the tokenizer
        
        functions:
            search: entrypoint for trigger search
            MCST: Monte Carlo Search Tree for trigger search
    """
    def __init__(self, 
                 model: AutoModelForCausalLM,
                 tokenizer: AutoTokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer
    

    def search(self, prompts: Union[List[str], str]):
        if  isinstance(prompts, str):
            prompts = [prompts]


    def MCST(self, prompts: Union)

