import re
from collections import OrderedDict
from os.path import join

import torch
from torchvision.io import read_image
from tqdm import tqdm
import os
import numpy as np
import json 

import logging
logger = logging.getLogger(__name__)

from pandas import DataFrame
import pyarrow as pa
import jsonpickle


def create_layer_map(model_repr_dict):
    model_layer_map = {}
    for (model_class, models) in model_repr_dict.items():
        layers = models[0]
        layer_names = list(layers.keys())
        base_layer_names = list()
        for item in layer_names:
            toks = re.sub("(weight|bias|running_(mean|var)|num_batches_tracked)", "", item)
            # remove any duplicate '.' separators
            toks = re.sub("\\.+", ".", toks)
            base_layer_names.append(toks)
        # use dict.fromkeys instead of set() to preserve order
        base_layer_names = list(dict.fromkeys(base_layer_names))

        layer_map = OrderedDict()
        for base_ln in base_layer_names:
            re_query = "{}.+".format(base_ln.replace('.', '\.'))  # escape any '.' wildcards in the regex query
            layer_map[base_ln] = [ln for ln in layer_names if re.match(re_query, ln) is not None]

        model_layer_map[model_class] = layer_map

    return model_layer_map


def load_model(model_filepath: str):
    """Load a model given a specific model_path.

    Args:
        model_filepath: str - Path to model.pt file

    Returns:
        model, dict, str - Torch model + dictionary representation of the model + model class name
    """
    model = torch.load(model_filepath)
    model_class = model.__class__.__name__
    model_repr = OrderedDict(
        {layer: tensor.numpy() for (layer, tensor) in model.state_dict().items()}
    )

    return model, model_repr, model_class


def load_ground_truth(model_dirpath: str):
    """Returns the ground truth for a given model.

    Args:
        model_dirpath: str -

    Returns:

    """

    with open(join(model_dirpath, "ground_truth.csv"), "r") as fp:
        model_ground_truth = fp.readlines()[0]

    return int(model_ground_truth)

def load_examples(model_dirpath: str, clean = True):
    """Returns the clean examples for a model.
    
    Args:
        model_dirpath: str -

    Returns:

    """  
    fvs={}
    labels={}
    for examples_dir_entry in os.scandir(join(model_dirpath, "clean-example-data" if clean else "poisoned-example-data")):
            if examples_dir_entry.is_file():
                if examples_dir_entry.name.endswith(".jpg"):
                    idx = examples_dir_entry.name.split('.jpn')[0]
                    feature_vector = read_image(examples_dir_entry.path).unsqueeze(dim = 0).float()
                    fvs[idx] = feature_vector
                elif examples_dir_entry.name.endswith(".npy"):
                    idx = examples_dir_entry.name.split('.jpn')[0] 
                    feature_vector = np.load(examples_dir_entry.path).reshape(1, -1)
                    fvs[idx] = feature_vector
                elif examples_dir_entry.name.endswith(".json"):
                    idx = examples_dir_entry.name.split('.json')[0]
                    label = json.load(examples_dir_entry.path)
                    labels[idx] = label
                elif examples_dir_entry.name == 'env-string.txt':
                    logger.info("Find {}".format(examples_dir_entry.name))
                    with open(examples_dir_entry.path, 'r') as file:
                        for env in file.readlines():
                            env = env.strip()
                            if env not in fvs:
                                fvs[env] = 0
                            fvs[env] += 1
                elif examples_dir_entry.name == 'data.txt':
                    with open(examples_dir_entry.path, 'rb') as file:
                        idx = file.read().replace('\n', '')
                        if idx not in fvs:
                            fvs[idx] = 0
                        fvs[idx] = 1
                else:
                    logger.info('Unrecognized file format: %s' % examples_dir_entry.name)
             
    return fvs, labels

def load_models_dirpath(models_dirpath):
    model_dict = {}
    model_repr_dict = {}
    model_ground_truth_dict = {}
    clean_example_dict = {'fvs': {}, 'labels': {}}
    poisoned_example_dict = {'fvs': {}, 'labels': {}}

    for model_path in tqdm(models_dirpath):
        model, model_repr, model_class = load_model(
            join(model_path, "model.pt")
        )
        
        # Build the list of models
        if model_class not in model_repr_dict.keys():
            model_dict[model_class] = []
            model_repr_dict[model_class] = []
            model_ground_truth_dict[model_class] = []

        model_dict[model_class].append(model)
        model_repr_dict[model_class].append(model_repr)

        try:
            model_ground_truth = load_ground_truth(model_path)
            model_ground_truth_dict[model_class].append(model_ground_truth)
        except:
            logger.info("Can't find ground truth")
            pass
        try:
            clean_example_fvs, clean_example_labels = load_examples(model_path)
            if clean_example_dict['labels'] != {}:
                clean_example_dict['fvs'].update(clean_example_fvs)
                clean_example_dict['labels'].update(clean_example_labels)
            else:
                for idx in clean_example_fvs:
                    if idx not in clean_example_dict['fvs']:
                        clean_example_dict['fvs'][idx] = 0
                    clean_example_dict['fvs'][idx] += clean_example_fvs[idx]
        except:
            logger.info("No clean example")
            pass
        try:
            poisoned_example_fvs, poisoned_example_labels = load_examples(model_path, False)
            if poisoned_example_dict['labels'] != {}:
                poisoned_example_dict['fvs'].update(poisoned_example_fvs)
                poisoned_example_dict['labels'].update(poisoned_example_labels)
            else:
                for idx in poisoned_example_fvs:
                    if idx not in poisoned_example_dict['fvs']:
                        poisoned_example_dict['fvs'][idx] = 0
                    poisoned_example_dict['fvs'][idx] += poisoned_example_fvs[idx]
        except:
            logger.info("No poisoned example")
            pass
    return model_dict, model_repr_dict, model_ground_truth_dict, clean_example_dict, poisoned_example_dict

