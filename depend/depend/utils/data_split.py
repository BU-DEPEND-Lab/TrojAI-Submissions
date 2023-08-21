from distutils.util import split_quoted
from typing import Optional, List, Dict, Callable, Any
from functools import partial
from dataclasses import dataclass, field
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
from datasets import concatenate_datasets
from abc import ABC, abstractmethod      
from pydantic import BaseModel, validator


class BaseDataSplit(ABC):
    head: Dataset 
     
@dataclass        
class DataSplit(BaseDataSplit):
    head: Dataset = ...
    tail: Optional[BaseDataSplit] = None
 

    @classmethod
    def append(cls, split: BaseDataSplit, dataset: Dataset):
        if split.tail is None:
            return cls(head = dataset)
        else:
            return cls(head = split.head, tail = cls.append(split.tail, dataset))
    
    @classmethod
    def concatenate(cls, split1: BaseDataSplit, split2: BaseDataSplit):
        if split1.tail is None:
            return cls(split1.head, split2)
        else:
            return cls(head = split1.head, tail = cls.concatenate(split1.tail, split2))
 
    @classmethod
    def split_dataset(
        cls,
        dataset: Dataset,
        num_split: int, 
        ) -> BaseDataSplit: 
        if num_split > 1: 
            tot_samples = len(dataset)    
            samples_per_split = tot_samples // num_split 
            tot_samples_added = 0
            while tot_samples_added < tot_samples:
                if tot_samples_added == 0:
                    split_dataset = dataset.select(range(samples_per_split))
                    data_split = cls(head = split_dataset) 
                    tot_samples_added += samples_per_split
                else:
                    assert data_split is not None
                    start = tot_samples_added
                    end =  min(tot_samples_added + samples_per_split, len(dataset))
                    dataset_slice = dataset.select(range(start, end))
                    data_split.append(dataset_slice) 
                    tot_samples_added += samples_per_split
            return data_split
        else:
            return cls(head = dataset)
       
    
    def append(self, dataset: Dataset):
        if self.tail is None:
            self.tail = DataSplit(head = dataset)
        else:
            self.tail.append(dataset)
    
    def concatenate(self, data_split: Optional[BaseDataSplit]):
        if data_split is not None:
            self.append(data_split.head)
            self.tail.concatenate(data_split.tail)
   
    def compose(self) -> Dataset:
        if self.tail is None:
            return self.head
        else:
            dataset1 = concatenate_datasets([self.head, self.tail.head])
            if self.tail.tail is not None:
                dataset2 = self.tail.tail.compose()
                return concatenate_datasets([dataset1, dataset2])
            else:
                return dataset1
        
