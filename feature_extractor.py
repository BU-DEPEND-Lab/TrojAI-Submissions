import json
import logging
import os
import pickle
from os import listdir, makedirs
from os.path import join, exists, basename
from collections import OrderedDict
import numpy as np
#from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tqdm import tqdm

from utils.abstract import AbstractDetector
from utils.flatten import flatten_model, flatten_models, flatten_grads
from utils.healthchecks import check_models_consistency
from utils.models import create_layer_map, load_model, \
    load_models_dirpath, inference_on_example_data
from utils.padding import create_models_padding, pad_model
from utils.reduction import (
    fit_feature_reduction_algorithm,
    use_feature_reduction_algorithm,
    grad_feature_reduction_algorithm,
    fit_ICA_feature_reduction_algorithm 
)
import pandas as pd
from attrdict import AttrDict
from sklearn.preprocessing import StandardScaler
from archs import Net2, Net3, Net4, Net5, Net6, Net7, Net2r, Net3r, Net4r, Net5r, Net6r, Net7r, Net2s, Net3s, Net4s, Net5s, Net6s, Net7s
import torch
import torch.nn.functional as F


logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        )

class FeatureExtractor(object):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath, scale_parameters_filepath):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
            scale_parameters_filepath: str - File path to the scale_parameters file.
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))
        self.metaparameter_filepath = metaparameter_filepath
        self.learned_parameters_dirpath = learned_parameters_dirpath
        self.scale_parameters_filepath = scale_parameters_filepath
        self.metaparameter_filepath = metaparameter_filepath
        self.model_layer_map_filepath = join(self.learned_parameters_dirpath, "model_layer_map.bin")
        self.layer_transform_filepath = join(self.learned_parameters_dirpath, "layer_transform.bin")

        # TODO: Update skew parameters per round
        self.model_skew = {
            "__all__": metaparameters["infer_cyber_model_skew"],
        }

        self.input_features = metaparameters["train_input_features"]
        self.ICA_features = metaparameters["train_ICA_features"]
        self.weight_table_params = {
            "random_seed": metaparameters["train_weight_table_random_state"],
            "mean": metaparameters["train_weight_table_params_mean"],
            "std": metaparameters["train_weight_table_params_std"],
            "scaler": metaparameters["train_weight_table_params_scaler"],
        }
         

    def write_metaparameters(self):
        metaparameters = {
            "infer_cyber_model_skew": self.model_skew["__all__"],
            "train_input_features": self.input_features,
            "train_ICA_features": self.ICA_features,
            "train_weight_table_random_state": self.weight_table_params["random_seed"],
            "train_weight_table_params_mean": self.weight_table_params["mean"],
            "train_weight_table_params_std": self.weight_table_params["std"],
            "train_weight_table_params_scaler": self.weight_table_params["scaler"],
        }

        with open(join(self.learned_parameters_dirpath, basename(self.metaparameter_filepath)), "w") as fp:
            json.dump(metaparameters, fp)

    def automatic_configure(self, models_dirpath: str):
        """Configuration of the detector iterating on some of the parameters from the
        metaparameter file, performing a grid search type approach to optimize these
        parameters.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        for random_seed in np.random.randint(1000, 9999, 10):
            self.weight_table_params["random_seed"] = random_seed
            self.manual_configure(models_dirpath)

    def manual_configure(self, models_dirpath: str):
        """Configuration of the detector using the parameters from the metaparameters
        JSON file.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        # Create the learned parameter folder if needed
        #if not exists(self.learned_parameters_dirpath):
        #    makedirs(self.learned_parameters_dirpath)

        # List all available model
        model_path_list = sorted([join(models_dirpath, model) for model in listdir(models_dirpath)])
        logging.info(f"Loading %d models...", len(model_path_list))

        model_dict, model_repr_dict, model_ground_truth_dict, clean_example_dict, poisoned_example_dict = load_models_dirpath(model_path_list)
        
        #models_padding_dict = create_models_padding(model_repr_dict)
        #with open(self.models_padding_dict_filepath, "wb") as fp:
        #    pickle.dump(models_padding_dict, fp)

        #for model_class, model_repr_list in model_repr_dict.items():
        #    for index, model_repr in enumerate(model_repr_list):
        #        model_repr_dict[model_class][index] = pad_model(model_repr, model_class, models_padding_dict)

        #check_models_consistency(model_repr_dict)

        # Build model layer map to know how to flatten
        logging.info("Generating model layer map...")
        model_layer_map = create_layer_map(model_repr_dict)
        
        flat_clean_grad_repr_dict = {}
        #flat_poisoned_grad_repr_dict= {}
        for _ in range(len(model_dict)):
            (model_class, models) = model_dict.popitem()
            flat_clean_grad_repr_dict[model_class] = []
            #flat_poisoned_grad_repr_dict[model_class] = []
            for i, model in enumerate(models):
                clean_examples = clean_example_dict[model_class][i]
                ground_truth = model_ground_truth_dict[model_class][i]
                print(f"Model class: {model_class}; Index: {i}")
                clean_grad = inference_on_example_data(model, ground_truth, clean_examples, self.scale_parameters_filepath, grad = True)
                flat_clean_grad_repr_dict[model_class].append(flatten_models({model_class: clean_grad}, model_layer_map)[model_class])
                print(flat_clean_grad_repr_dict[model_class][-1])
                #poisoned_examples_dirpath = poisoned_example_dict[model_class][i]
                #poisoned_grad = inference_on_example_data(model, poisoned_examples_dirpath, self.scale_parameters_filepath, grad = True)
                #flat_poisoned_grad_repr_dict[model_class].append(flatten_models({model_class: poisoned_grad}, model_layer_map)[model_class])
        
        
        logging.info("Models flattened. Fitting grad feature reduction...")
        clean_grad_layer_transform = fit_ICA_feature_reduction_algorithm(flat_clean_grad_repr_dict, self.weight_table_params, self.ICA_features)
        logging.info("Grad feature reduction done...")
        #poisoned_grad_layer_transform = fit_ICA_feature_reduction_algorithm(flat_poisoned_grad_repr_dict, self.weight_table_params, self.ICA_features)
        #logging.info("Models flattened. Fitting feature reduction...")
        

        with open(self.model_layer_map_filepath, "wb") as fp:
            pickle.dump(model_layer_map, fp)
        logging.info("Generated model layer map. Flattenning models...")
        
        flat_models = flatten_models(model_repr_dict, model_layer_map)
         
        logging.info("Models flattened. Fitting weight feature reduction...")
        layer_transform = fit_ICA_feature_reduction_algorithm(flat_models, self.weight_table_params, self.ICA_features)
        logging.info("Weight feature reduction done...")
        #del flat_models
        
        df = pd.DataFrame(columns=['model_class','index','features'])
        for _ in range(len(flat_models)):
            (model_class, models) = flat_models.popitem()
            for i, model in enumerate(models):
                model_feats = use_feature_reduction_algorithm(
                    layer_transform[model_class], models[i]
                )
                clean_grad_feats = use_feature_reduction_algorithm(
                    clean_grad_layer_transform[model_class], flat_clean_grad_repr_dict[model_class][i]
                )
                #poisoned_grad_feats = use_feature_reduction_algorithm(
                #    poisoned_grad_layer_transform[model_class], flat_poisoned_grad_repr_dict[model_class][i]
                #)

                feats = np.hstack((model_feats, clean_grad_feats)).tolist()#, poisoned_grad_feats)).tolist()
                df.loc[len(df.index)] = [model_class, i, feats] 
        df.to_csv("round12_features.csv")


if __name__ == "__main__":
    extractor = FeatureExtractor("./metaparameters.json", "./learned_parameters",  "./learned_parameters/scale_params.npy")
    extractor.manual_configure("/mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round12/models")