from distutils.util import split_quoted
from typing import Optional, List, Dict, Callable
from functools import partial
from dataclasses import dataclass, field
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset


from torch.utils.data import Dataset
 
class TorchDataset(Dataset):
    def __init__(self, data_set):
        self.data_set = data_set

    def __getitem__(self, idx):
        return self.data_set[idx]

    def __len__(self):
        return len(self.data_set)
 
@dataclass
class TrainSplit(object):
    dataset: Dataset
    schemas: Dict[str, dict]


@dataclass
class EvalSplit(object):
    dataset: Dataset
    examples: Dataset
    schemas: Dict[str, dict]


@dataclass
class DatasetSplits(object):
    train_split: Optional[TrainSplit]
    eval_split: Optional[EvalSplit]
    test_splits: Optional[Dict[str, EvalSplit]]
    schemas: Dict[str, dict]
 

def _prepare_train_split(
    dataset: Dataset,
    data_args: DataArguments,
    data_training_args: DataTrainingArguments,
    add_serialized_schema: Callable[[dict], dict],
    pre_process_function: Callable[[dict, Optional[int], Optional[int]], dict],
) -> TrainSplit:
    schemas = _get_schemas(examples=dataset)
    dataset = dataset.map(
        add_serialized_schema,
        batched=False,
        num_proc=data_training_args.preprocessing_num_workers,
        load_from_cache_file=not data_training_args.overwrite_cache,
    )
    
    if data_training_args.max_train_samples is not None:
        dataset = dataset.select(range(data_training_args.max_train_samples))
    column_names = dataset.column_names
    dataset = dataset.map(
        lambda batch: pre_process_function(
            batch=batch,
            max_source_length=data_training_args.max_source_length,
            max_target_length=data_training_args.max_target_length,
        ),
        batched=True,
        num_proc=data_training_args.preprocessing_num_workers,
        remove_columns=column_names[:-1] if column_names[-1]=="relations" else column_names,
        load_from_cache_file=not data_training_args.overwrite_cache,
    )
     

    return TrainSplit(dataset=dataset, schemas=schemas)


def _prepare_eval_split(
    dataset: Dataset,
    data_args: DataArguments,
    data_training_args: DataTrainingArguments,
    add_serialized_schema: Callable[[dict], dict],
    pre_process_function: Callable[[dict, Optional[int], Optional[int]], dict],
) -> EvalSplit:
    if (data_training_args.max_val_samples is not None 
            and data_training_args.max_val_samples < len(dataset)):
        eval_examples = dataset.select(range(data_training_args.max_val_samples))
    else:
        eval_examples = dataset
    schemas = _get_schemas(examples=eval_examples)
    eval_dataset = eval_examples.map(
        add_serialized_schema,
        batched=False,
        num_proc=data_training_args.preprocessing_num_workers,
        load_from_cache_file=not data_training_args.overwrite_cache,
    )

    column_names = eval_dataset.column_names
    eval_dataset = eval_dataset.map(
        lambda batch: pre_process_function(
            batch=batch,
            max_source_length=data_training_args.max_source_length,
            max_target_length=data_training_args.val_max_target_length,
        ),
        batched=True,
        num_proc=data_training_args.preprocessing_num_workers,
        remove_columns=column_names[:-1] if column_names[-1]=="relations" else column_names,
        load_from_cache_file=not data_training_args.overwrite_cache,
    )
    

    #TODO: It can not do the testing now (test will do dev still)
    eval_input_ids = [eval_dataset[i]['input_ids'] for i in range(len(eval_dataset))]
    
    if data_training_args.use_rasat:
        relation_matrix_l = preprocess_by_dataset(
            data_args.data_base_dir, 
            data_args.split_dataset, 
            eval_input_ids, 
            "dev", 
            edge_type=data_training_args.edge_type, 
            use_coref=data_training_args.use_coref,
            use_dependency=data_training_args.use_dependency
            )

        def add_relation_info_train(example, idx, relation_matrix_l=relation_matrix_l):
            example['relations'] = relation_matrix_l[idx]  
            return example
            
        eval_dataset = eval_dataset.map(add_relation_info_train, with_indices=True)
    return EvalSplit(dataset=eval_dataset, examples=eval_examples, schemas=schemas)


def prepare_splits(
    dataset_dict: DatasetDict,
    data_args: DataArguments,
    training_args: TrainingArguments,
    data_training_args: DataTrainingArguments,
    add_serialized_schema: Callable[[dict], dict],
    pre_process_function: Callable[[dict, Optional[int], Optional[int]], dict],
) -> DatasetSplits:
    train_split, eval_split, test_splits = None, None, None
 
    if training_args.do_train:
        train_split = _prepare_train_split(
            dataset_dict["train"],
            data_args = data_args,
            data_training_args=data_training_args,
            add_serialized_schema=add_serialized_schema,
            pre_process_function=partial(pre_process_function, preprocess_type = 'causalml'),
        )

    if training_args.do_eval:
        eval_split = _prepare_eval_split(
            dataset_dict["validation"],
            data_args = data_args,
            data_training_args=data_training_args,
            add_serialized_schema=add_serialized_schema,
            pre_process_function=partial(pre_process_function, preprocess_type = 'seq2seq'),
        )

    if training_args.do_predict:
        test_splits = {
            section: _prepare_eval_split(
                dataset_dict[section],
                data_args = data_args,
                data_training_args=data_training_args,
                add_serialized_schema=add_serialized_schema,
                pre_process_function=partial(pre_process_function, preprocess_type = 'seq2seq'),
            )
            for section in ["validation"] #data_args.test_sections
        }
        test_split_schemas = {}
        for split in test_splits.values():
            test_split_schemas.update(split.schemas)
     
    schemas = {
        **(train_split.schemas if train_split is not None else {}),
        **(eval_split.schemas if eval_split is not None else {}),
        **(test_split_schemas if test_splits is not None else {}),
    }

    return DatasetSplits(
        train_split=train_split, 
        eval_split=eval_split, 
        test_splits=test_splits, 
        schemas=schemas
    )


def normalize(query: str) -> str:
    def comma_fix(s):
        # Remove spaces in front of commas
        return s.replace(" , ", ", ")

    def white_space_fix(s):
        # Remove double and triple spaces
        return " ".join(s.split())

    def lower(s):
        # Convert everything except text between (single or double) quotation marks to lower case
        return re.sub(r"\b(?<!['\"])(\w+)(?!['\"])\b", lambda match: match.group(1).lower(), s)

    return comma_fix(white_space_fix(lower(query)))
