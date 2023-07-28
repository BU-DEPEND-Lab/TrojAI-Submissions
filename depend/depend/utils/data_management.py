from distutils.util import split_quoted
from typing import Optional, List, Dict, Callable, Any
from functools import partial
from dataclasses import dataclass, field
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
from abc import ABC, abstractmethod      

from depend.utils.configs import DataConfig


class DataManagement:
    def __init__(self):
        

@dataclass
class BaseDataSplit(ABC):
    head: Optional[Dataset]
    
@dataclass
class DataSplit(BaseDataSplit):
    head: Dataset = ...
    tail: Optional[BaseDataSplit] = None

    def append(self, dataset: Dataset):
        if self.tail is None:
            self.tail = DataSplit(dataset)
        else:
            self.tail.append(dataset)
    
    def concatenate(self, data_split: BaseDataSplit):
        self.append(data_split.head)
        self.tail = data_split.tail
 
    @classmethod
    def split_dataset(
        cls,
        dataset: Dataset,
        data_config: DataConfig, 
        pre_process_function: Callable[[dict, Optional[int], Optional[int]], dict],
        ) -> BaseDataSplit: 
        processed_dataset = dataset.map(
            pre_process_function,
            batched=True,
            num_proc=data_config.num_workers,
            load_from_cache_file=not data_config.overwrite_cache,
        ).set_format(
            type = data_config.type,
            columns = data_config.columns
        )
 
        data_split = None
        if data_config.num_split is not None: 
            tot_samples = len(processed_dataset)    
            samples_per_split = tot_samples // data_config.num_split 
            tot_samples_added = 0
            while tot_samples_added < tot_samples:
                if tot_samples_added == 0:
                    torch_dataset = dataset.select(range(samples_per_split))
                    data_split = DataSplit(torch_dataset) 
                    tot_samples_added += samples_per_split
                else:
                    assert data_split is not None
                    start = tot_samples_added
                    end =  min(tot_samples_added + samples_per_split, len(processed_dataset))
                    dataset_slice = processed_dataset.select(range(start, end))
                    data_split.append(dataset_slice) 
        else:
            data_split = DataSplit(processed_dataset)
       
        return data_split
 
 