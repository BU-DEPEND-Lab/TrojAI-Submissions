from typing import Any, Dict, List, Literal, TypedDict, Union, cast
from dataclasses import dataclass
import json
import pandas as pd

import logging
logger = logging.getLogger(__name__)

@dataclass
class Model_Indexer:
    model_dict: Dict[str, Dict[int, Any]]
    model_repr_dict: Dict[str, Dict[int, Any]]

    def __post_init__(self):
        for model_class in self.model_dict:
            for model in self.model_dict[model_class]:
                for param in model.parameters():
                    param.requires_grad = True

    @property
    def attributes(self) -> Dict:
        """
        Return a list of attribute names that should be included in the
        serialized kwargs. These attributes must be accepted by the
        constructor.
        """
        return {
            'model_repr_dict': self.model_repr_dict,
            }
 
 
    def get_model(self, index: Dict[str, List[Any]]):
        return list(map(
            lambda class_index: self.model_dict[class_index[0]][class_index[1]],
            zip(index['model_class'], index['idx_in_class'])
        )) 
         
    

    