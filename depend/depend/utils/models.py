# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import re
from collections import OrderedDict
from os.path import join

import torch
from torchvision.io import read_image
from tqdm import tqdm
import json
import os

import numpy as np 

import logging
logger = logging.getLogger(__name__)

from pandas import DataFrame
import pyarrow as pa
import jsonpickle

from utils.trafficnn import TrafficNN


def create_layer_map(model_repr_dict):
    model_layer_map = {}
    for (model_class, models) in model_repr_dict.items():
        layers = models[0]
        layer_names = list(layers.keys())
        base_layer_names = list(
            dict.fromkeys(
                [
                    re.sub(
                        "\\.(weight|bias|running_(mean|var)|num_batches_tracked)",
                        "",
                        item,
                    )
                    for item in layer_names
                ]
            )
        )
        layer_map = OrderedDict(
            {
                base_layer_name: [
                    layer_name
                    for layer_name in layer_names
                    if re.match(f"{base_layer_name}.+", layer_name) is not None
                ]
                for base_layer_name in base_layer_names
            }
        )
        model_layer_map[model_class] = layer_map

    return model_layer_map


def load_model(model_filepath: str) -> (dict, str):
    """Load a model given a specific model_path.

    Args:
        model_filepath: str - Path to model.pt file

    Returns:
        model, dict, str - Torch model + dictionary representation of the model + model class name
    """

    conf_filepath = os.path.join(os.path.dirname(model_filepath), 'reduced-config.json')
    with open(conf_filepath, 'r') as f:
        full_conf = json.load(f)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    model = TrafficNN(full_conf['img_resolution']**2, full_conf)
    model.model.load_state_dict(torch.load(model_filepath, map_location=device))
    model.model.to(model.device).eval()

    model_class = model.model.__class__.__name__
    model_repr = OrderedDict(
        {layer: tensor.cpu().numpy() for (layer, tensor) in model.model.state_dict().items()}
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
                elif examples_dir_entry.name.endswith(".png"):
                    idx = examples_dir_entry.name.split('.')[0]
                    feature_vector = read_image(examples_dir_entry.path).unsqueeze(dim = 0).float()
                    fvs[idx] = feature_vector
                elif examples_dir_entry.name.endswith(".npy"):
                    idx = examples_dir_entry.name.split('.npy')[0] 
                    feature_vector = np.load(examples_dir_entry.path).reshape(1, -1)
                    fvs[idx] = feature_vector
                elif examples_dir_entry.name.endswith(".json"):
                    idx = examples_dir_entry.name.split('.json')[0]
                    with open(examples_dir_entry.path, 'r') as fp:
                        label = json.load(fp)
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
                print(clean_example_dict['labels'])
            else:
                for idx in clean_example_fvs:
                    if idx not in clean_example_dict['fvs']:
                        clean_example_dict['fvs'][idx] = 0
                    clean_example_dict['fvs'][idx] += clean_example_fvs[idx]
                for idx in clean_example_labels:
                    if idx not in clean_example_dict['labels']:
                        clean_example_dict['labels'][idx] = clean_example_labels[idx]
                        
        except Exception as e:
            logger.info(f"{e}. No clean example")
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

"""
def create_models(model_classes, obs_space, action_space):
    def init_weights(module):
        if type(module) == nn.Linear:
            torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5)) 
            if module.bias is not None: 
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(module.weight) 
                bound = 1 / math.sqrt(fan_in) 
                torch.nn.init.uniform_(module.bias, -bound, bound) 
        elif type(module) == nn.Conv2d:
            n = module.in_channels 
            torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5)) 
            if module.bias is not None: 
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(module.weight) 
                bound = 1 / math.sqrt(fan_in) 
                torch.nn.init.uniform_(module.bias, -bound, bound) 

    for (class_name, model_num, model_class) in model_classes:
        model = eval(model_class)(obs_space, action_space)
        model.state_emb.apply(init_weights)  # Define state embedding
        model.actor.apply(init_weights)  # Define actor's model
        model.critic.apply(init_weights)  # Define critic's model
"""