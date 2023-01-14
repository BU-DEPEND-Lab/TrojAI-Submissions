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
    ICA_feature_reduction_algorithm
)
from attrdict import AttrDict
from sklearn.preprocessing import StandardScaler
from archs import Net2, Net3, Net4, Net5, Net6, Net7, Net2r, Net3r, Net4r, Net5r, Net6r, Net7r, Net2s, Net3s, Net4s, Net5s, Net6s, Net7s
import torch
import torch.nn.functional as F

class FeatureDetector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath, scale_parameters_filepath):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
            scale_parameters_filepath: str - File path to the scale_parameters file.
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))

        self.scale_parameters_filepath = scale_parameters_filepath
        self.metaparameter_filepath = metaparameter_filepath
         

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

        model_repr_dict, model_ground_truth_dict, clean_example_dict, poisoned_example_dict = load_models_dirpath(model_path_list)

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
        with open(self.model_layer_map_filepath, "wb") as fp:
            pickle.dump(model_layer_map, fp)
        logging.info("Generated model layer map. Flattenning models...")
        
        flat_models = flatten_models(model_repr_dict, model_layer_map)
        del model_repr_dict
        logging.info("Models flattened. Fitting feature reduction...")

        layer_transform = ICA_feature_reduction_algorithm(flat_models, self.weight_table_params, self.ICA_features)

        clean_flat_grads = {}
        poisoned_flat_grads = {}
        clean_grad_repr_dict = {}
        poisoned_grad_repr_dict = {}
        for (model_class, models) in model_repr_dict:
            clean_grad_reprs[model_class] = []
            poisoned_grad_repr_dict[model_class] = []
            for i, model in enumerate(models):
                clean_examples_dirpath = clean_example_dict[model_class][i]
                poisoned_examples_dirpath = poisoned_example_dict[model_class][i]
                clean_grad_repr_dict[model_class].append(self.inference_on_example_data(model, clean_examples_dirpath, grad = True))
                poisoned_grad_repr_dict[model_class].append(self.inference_on_example_data(model, poisoned_examples_dirpath, grad = True))
        
        flat_clean_grad_repr_dict = flatten_models(clean_grad_repr_dict, model_layer_map)
        del clean_grad_repr_dict
        logging.info("Models flattened. Fitting feature reduction...")

        flat_poisoned_grad_repr_dict = flatten_models(poisoned_grad_repr_dict, model_layer_map)
        del poisoned_grad_repr_dict
        logging.info("Models flattened. Fitting feature reduction...")


        clean_grad_layer_transform = ICA_feature_reduction_algorithm(clean_grad_repr_dict, self.weight_table_params, self.ICA_features)
        poisoned_grad_layer_transform = ICA_feature_reduction_algorithm(poisoned_grad_repr_dict, self.weight_table_params, self.ICA_features)
        

        # Flatten models
        
        layer_transform = fit_feature_reduction_algorithm(flat_models, self.weight_table_params, self.ICA_features)

        
        for (model_class, models) in model_repr_dict: 
            
            for model in models:
                
            poisoned_flat_grads = 
                grad_reprs = self.inference_on_example_data(model, examples_dirpath, grad = True)
      
        for grad_repr in grad_reprs:
            #grad_repr = pad_model(grad_repr, model_class, models_padding_dict)
            flat_grad = flatten_model(grad_repr, model_layer_map[model_class])
            #grad_layer_transform = fit_feature_reduction_algorithm(flat_grad, self.weight_table_params, self.ICA_features)
            flat_grads.append(flat_grad)
        grad_layer_transform = grad_feature_reduction_algorithm({model_class: flat_grads}, self.weight_table_params, self.ICA_features)
        logging.info("Feature reduction applied. Creating feature file...")
        X = None
        y = []

        for _ in range(len(flat_models)):
            (model_arch, models) = flat_models.popitem()
            model_index = 0

            logging.info("Parsing %s models...", model_arch)
            for _ in tqdm(range(len(models))):
                model = models.pop(0)
                y.append(model_ground_truth_dict[model_arch][model_index])
                model_index += 1

                model_feats = use_feature_reduction_algorithm(
                    layer_transform[model_arch], model, model_transform[model_arch]
                )
                if X is None:
                    X = model_feats
                    continue

                X = np.vstack((X, model_feats)) * self.model_skew["__all__"]
        logging.info("Training XGBoostRegressor model...")
        # Instantiation
        model = XGBRegressor(**self.xgboost_kwargs)
        #logging.info("Training RandomForestRegressor model...")
        #model = RandomForestRegressor(**self.random_forest_kwargs, random_state=0)
        model.fit(X, y)

        #logging.info("Saving RandomForestRegressor model...")
        logging.info("Saving XGBoostRegressor model...")
        
        #with open(self.model_filepath, "wb") as fp:
        #    pickle.dump(model, fp)
        model.save_model(self.model_filepath)

        self.write_metaparameters()
        logging.info("Configuration done!")

    
                
    def infer(
        self,
        model_filepath,
        result_filepath,
        scratch_dirpath,
        examples_dirpath,
        round_training_dataset_dirpath,
    ):
        """Method to predict wether a model is poisoned (1) or clean (0).

        Args:
            model_filepath:
            result_filepath:
            scratch_dirpath:
            examples_dirpath:
            round_training_dataset_dirpath:
        """
        with open(self.model_layer_map_filepath, "rb") as fp:
            model_layer_map = pickle.load(fp)

        # List all available model and limit to the number provided
        model_path_list = sorted(
            [
                join(round_training_dataset_dirpath, 'models', model)
                for model in listdir(join(round_training_dataset_dirpath, 'models'))
            ]
        )
        logging.info(f"Loading %d models...", len(model_path_list))

        model_repr_dict, _ = load_models_dirpath(model_path_list)
        logging.info("Loaded models. Flattenning...")

        #with open(self.models_padding_dict_filepath, "rb") as fp:
        #    models_padding_dict = pickle.load(fp)

        #for model_class, model_repr_list in model_repr_dict.items():
        #    for index, model_repr in enumerate(model_repr_list):
        #        model_repr_dict[model_class][index] = pad_model(model_repr, model_class, models_padding_dict)

        # Flatten model
        flat_models = flatten_models(model_repr_dict, model_layer_map)
        
        del model_repr_dict
        logging.info("Models flattened. Fitting feature reduction...")

        layer_transform = fit_feature_reduction_algorithm(flat_models, self.weight_table_params, self.ICA_features)
        #model_transform = grad_feature_reduction_algorithm(flat_models, self.input_features - self.ICA_features)
        model, model_repr, model_class = load_model(model_filepath)
        
        #model_repr = pad_model(model_repr, model_class, models_padding_dict)
        flat_model = flatten_model(model_repr, model_layer_map[model_class])
        logging.info(f"Flattened model: {[weights.shape for (layer, weights) in flat_model.items()]}")
        # Inferences on examples to demonstrate how it is done for a round
        # This is not needed for the random forest classifier
        grad_repr_dict = self.inference_on_example_data(model, examples_dirpath, grad = True)
        flat_grads = []
        for grad_repr in grad_reprs:
            #grad_repr = pad_model(grad_repr, model_class, models_padding_dict)
            flat_grad = flatten_model(grad_repr, model_layer_map[model_class])
            #grad_layer_transform = fit_feature_reduction_algorithm(flat_grad, self.weight_table_params, self.ICA_features)
            flat_grads.append(flat_grad)
        #logging.info(f"Flattened grads: {[weights for (layer, weights) in flat_grads[0].items()]}")
        #grad_layer_transform = fit_feature_reduction_algorithm({model_class: flat_grads}, self.weight_table_params, self.ICA_features)
        grad_layer_transform = grad_feature_reduction_algorithm({model_class: flat_grads}, self.weight_table_params, self.ICA_features)
        logging.info("Grad transformer fitted")
        X = (
            np.hstack(\
                (use_feature_reduction_algorithm(layer_transform[model_class], [flat_model]),\
                    use_feature_reduction_algorithm(grad_layer_transform[model_class], flat_grads))),
            * self.model_skew["__all__"]
        )
        logging.info("Start fitting regressor ...")
        #with open(self.model_filepath, "rb") as fp:
        #    regressor: RandomForestRegressor = pickle.load(fp)
            
        regressor = XGBRegressor(**self.xgboost_kwargs)
        regressor.load_model(self.model_filepath);

        probability = str(regressor.predict(X)[0])
        with open(result_filepath, "w") as fp:
            fp.write(probability)

        logging.info("Trojan probability: %s", probability)
