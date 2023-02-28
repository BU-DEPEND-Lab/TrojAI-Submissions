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
import random 

from utils.abstract import AbstractDetector
from utils.flatten import flatten_model, flatten_models, flatten_grads
from utils.healthchecks import check_models_consistency
from utils.models import create_layer_map, load_model, \
    load_models_dirpath, inference_on_example_data, get_attribution_from_example_data
from utils.padding import create_models_padding, pad_model
from utils.reduction import (
    fit_feature_reduction_algorithm,
    use_feature_reduction_algorithm,
    grad_feature_reduction_algorithm,
    fit_ICA_feature_reduction_algorithm 
)
import pandas as pd
#from attrdict import AttrDict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from archs import Net2, Net3, Net4, Net5, Net6, Net7, Net2r, Net3r, Net4r, Net5r, Net6r, Net7r, Net2s, Net3s, Net4s, Net5s, Net6s, Net7s
import torch
import math
import torch.nn.functional as F

def hist_v(w,bins=100):
    s =np.sort(w, axis=0);
    wmin=float(w.min());
    wmax=float(w.max());
    
    n=s.shape[0];
    hist=torch.Tensor(bins);
    for i in range(bins):
        x=math.floor((n-1)/(bins-1)*i)
        x=max(min(x,n),0);
        v=float(s[x]);
        hist[i]=v;
    
    return hist;


logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        )

class FeatureExtractor(object):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
            scale_parameters_filepath: str - File path to the scale_parameters file.
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))
        self.learned_parameters_dirpath = learned_parameters_dirpath
        self.model_layer_map_filepath = join(self.learned_parameters_dirpath, "model_layer_map.bin")
        self.layer_transform_filepath = join(self.learned_parameters_dirpath, "layer_transform.bin")
        self.clean_grad_layer_transform_filepath = join(self.learned_parameters_dirpath, "clean_grad_layer_transform.bin")
        self.poisoned_grad_layer_transform_filepath = join(self.learned_parameters_dirpath, "poisoned_grad_layer_transform.bin")

        # TODO: Update skew parameters per round
         
        self.input_features = metaparameters["train_input_features"]

        print(metaparameters)
         

    def write_metaparameters(self):
        metaparameters = {
            "train_input_features": self.input_features
        }
       
        return metaparameters

    
    def infer_attribution_feature_from_models(self, model_path_list, num_data_per_model = 20, train= False):
        logging.info(f"Loading %d models...", len(model_path_list))
        all_feats = []
        
        for model_filepath in model_path_list:
            feats = self.infer_attribution_feature_from_one_model(model_filepath, num_data_per_model, train)
            all_feats.append(feats)
            if train:     
                self.input_features = ",".join(feats.shape.tolist())
        return all_feats

    def infer_attribution_feature_from_one_model(self, model_filepath, num_data_per_model, train = False):
        model_dict, _, _, clean_example_dict, _ = load_models_dirpath([model_filepath])
        model_class, [model] = model_dict.popitem()
         
        _, clean_examples = clean_example_dict.popitem()
         
    
        attrs = []
        num_examples = clean_examples['fvs'].shape[0]
        if num_data_per_model > num_examples:
            example_ids = np.hstack((np.arange(num_examples), np.random.choice(num_examples, num_data_per_model - num_examples)))
        else:
            example_ids = np.arange(num_examples)
        examples = clean_examples['fvs'][example_ids]
        labels = clean_examples['labels'][example_ids]
        for clean_example, label in zip(examples, labels):
            attrs.append(get_attribution_from_example_data(model, label, [clean_example]))
        return np.hstack((examples, attrs)) #np.asarray(attrs) #([] if train else [0. for _ in range(self.input_features - attrs.shape[0])])


if __name__ == "__main__":
    extractor = FeatureExtractor("./metaparameters.json", "./learned_parameters",  "./learned_parameters/scale_params.npy")
    #extractor.manual_configure("/mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round12/models")
    #print(extractor.infer_one_model("./model/id-00000002/"))

    extractor.infer_norms_from_models("/mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round12/models")
    