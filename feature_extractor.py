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
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath, scale_parameters_filepath):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
            scale_parameters_filepath: str - File path to the scale_parameters file.
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))
        self.learned_parameters_dirpath = learned_parameters_dirpath
        self.scale_parameters_filepath = scale_parameters_filepath
        self.model_layer_map_filepath = join(self.learned_parameters_dirpath, "model_layer_map.bin")
        self.layer_transform_filepath = join(self.learned_parameters_dirpath, "layer_transform.bin")
        self.clean_grad_layer_transform_filepath = join(self.learned_parameters_dirpath, "clean_grad_layer_transform.bin")
        self.poisoned_grad_layer_transform_filepath = join(self.learned_parameters_dirpath, "poisoned_grad_layer_transform.bin")

        # TODO: Update skew parameters per round
         
        self.input_features = metaparameters["train_input_features"]
        self.ICA_features = metaparameters.get("train_ICA_features", None)
 
 
        self.weight_table_params = {
            "random_seed": metaparameters["train_weight_table_random_state"],
            "mean": metaparameters["train_weight_table_params_mean"],
            "std": metaparameters["train_weight_table_params_std"],
            "scaler": metaparameters["train_weight_table_params_scaler"],
        }

        print(metaparameters)
         

    def write_metaparameters(self):
        metaparameters = {
            "train_input_features": self.input_features,
            "train_weight_table_random_state": self.weight_table_params["random_seed"],
            "train_weight_table_params_mean": self.weight_table_params["mean"],
            "train_weight_table_params_std": self.weight_table_params["std"],
            "train_weight_table_params_scaler": self.weight_table_params["scaler"],
        }
        if self.ICA_features is not None:
            metaparameters.update({"train_ICA_features": self.ICA_features})

        return metaparameters

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
        #for (model_class, models_repr) in model_repr_dict.items():
            #logging.info(f"Model class {model_class} || Number Of models {len(model_dict[model_class])} || Numbers Of Examples {[clean_example.shape for clean_example in clean_example_dict[model_class]]}")
 
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
        logging.info("Generated model layer map. Flattenning model grads...")

        flat_clean_grad_repr_dict = {}
        #flat_poisoned_grad_repr_dict= {}
        for _ in range(len(model_dict)):
            (model_class, models) = model_dict.popitem()
            flat_clean_grad_repr_dict[model_class] = []
            #flat_poisoned_grad_repr_dict[model_class] = []
            for i, model in enumerate(models):
                clean_examples = clean_example_dict[model_class][i]
                ground_truth = model_ground_truth_dict[model_class][i]
                #print(f"Model class: {model_class}; Index: {i}")
                clean_grads = inference_on_example_data(model, ground_truth, clean_examples, self.scale_parameters_filepath, grad = True)
                
                flat_clean_grad = {}
                flat_clean_grads = flatten_models({model_class: clean_grads}, model_layer_map[model_class])[model_class]
                for i in range(len(flat_clean_grads)):
                    for (k, v) in flat_clean_grads[i].items():
                        if flat_clean_grad.get(k, None) is None:
                            flat_clean_grad[k] = np.arrary([[]])
                        flat_clean_grad[k] = np.vstack((flat_clean_grads[k], v))
                del clean_grads, flat_clean_grads
                for (k, v) in flat_clean_grad.items():
                    flat_clean_grad[k] = np.mean(flat_clean_grad[k], axis = 0)
                flat_clean_grad_repr_dict[model_class].append(clean_grad)

                #flat_clean_grad_repr_dict[model_class].append(flatten_model(clean_grad, model_layer_map[model_class]))
            
                #print(flat_clean_grad_repr_dict[model_class][-1])
                #poisoned_examples_dirpath = poisoned_example_dict[model_class][i]
                #poisoned_grad = inference_on_example_data(model, poisoned_examples_dirpath, self.scale_parameters_filepath, grad = True)
                #flat_poisoned_grad_repr_dict[model_class].append(flatten_model(poisoned_grad, model_layer_map[model_class]))
        
        
        logging.info("Model grads flattened. Fitting grad feature reduction...")
        clean_grad_layer_transform = fit_ICA_feature_reduction_algorithm(flat_clean_grad_repr_dict, self.weight_table_params, self.ICA_features)
        logging.info("Grad feature reduction done...")
        #poisoned_grad_layer_transform = fit_ICA_feature_reduction_algorithm(flat_poisoned_grad_repr_dict, self.weight_table_params, self.ICA_features)
        #logging.info("Models flattened. Fitting feature reduction...")
    
        
        flat_models = flatten_models(model_repr_dict, model_layer_map)
         
        logging.info("Models flattened. Fitting weight feature reduction...")
        layer_transform = fit_ICA_feature_reduction_algorithm(flat_models, self.weight_table_params, self.ICA_features)
        logging.info("Weight feature reduction done...")
        #del flat_models
        
        """
        df = pd.DataFrame(columns=['model_class','index','features'])
        for _ in range(len(flat_models)):
            (model_class, models) = flat_models.popitem()
            (_, grads) = flat_clean_grad_repr_dict.popitem()
            for i, model in enumerate(models):
                logging.info(f"Model class: {model_class} || Index: {i} || Model layers shapes: {[(layer, weight.shape) for (layer, weight) in models[i].items()]} || Model grad shapes: {[(layer, len(grad)) for (layer, grad) in grads[i].items()]}")
                logging.info(f"layer_transformer components: {[(layer, layer_transform[model_class][layer].components_.shape) for (layer, _) in models[i].items()]}")
                logging.info(f"clean grad layer_transformer components: {[(layer, clean_grad_layer_transform[model_class][layer].components_.shape) for (layer, _) in models[i].items()]}")
                model_feats = use_feature_reduction_algorithm(
                    layer_transform[model_class], models[i]
                )
                clean_grad_feats = use_feature_reduction_algorithm(
                    clean_grad_layer_transform[model_class], grads[i]
                )
                #poisoned_grad_feats = use_feature_reduction_algorithm(
                #    poisoned_grad_layer_transform[model_class], flat_poisoned_grad_repr_dict[model_class][i]
                #)

                feats = np.hstack((model_feats, clean_grad_feats))#, poisoned_grad_feats)).tolist()
                logging.info(f" ICA feature size: {feats.shape}\n")
                feats = np.pad(feats, [(0, 0), (0, 2 * self.ICA_features - feats.shape[-1])], mode='constant')

                df.loc[len(df.index)] = [model_class, i, feats] 
        df.to_csv("round12_features.csv")
    
        """
        pickle.dump(layer_transform, open(self.layer_transform_filepath, 'wb'))
        pickle.dump(clean_grad_layer_transform, open(self.clean_grad_layer_transform_filepath, 'wb'))

    def infer_layer_feature_from_models(self, models_dirpath, train = False):
        model_path_list = sorted([join(models_dirpath, model) for model in listdir(models_dirpath)])
        logging.info(f"Loading %d models...", len(model_path_list))
        if train:
            max_num_feats = 0
        else:
            max_num_feats = self.input_features
        all_feats = []
        for model_filepath in model_path_list:
            feats = self.infer_layer_feature_from_one_model(model_filepath)
            max_num_feats = len(feats) if len(feats) > max_num_feats else max_num_feats
            all_feats.append(feats)
        for i in range(len(all_feats)):
            all_feats[i] = all_feats[i] + [0. for _ in range(max_num_feats - len(all_feats[i]))]
        if train:
            self.input_features = max_num_feats
        
        return all_feats

    def infer_layer_feature_from_one_model(self, model_filepath):
        model_dict, model_repr_dict, _, clean_example_dict, _ = load_models_dirpath([model_filepath])
        model_class, [model] = model_dict.popitem()
         
        _, [clean_example] = clean_example_dict.popitem()
        _, [model_repr] = model_repr_dict.popitem()
     
        clean_grad_reprs = inference_on_example_data(model, '1', clean_example, self.scale_parameters_filepath, grad = True)
        
        norm_fn = lambda model: [np.linalg.norm(param) if 'weight' in layer else  np.linalg.norm(param, ord = 2) for (layer, param) in model.items()]
        median_fn = norm_fn = lambda model: [np.median(param.reshape([-1])) for (layer, param) in model.items()]
        
 
        model_feats_weight_norm = median_fn(model_repr) 
        clean_grad_feats_weight_norm = [norm_fn(clean_grad_repr) for clean_grad_repr in clean_grad_reprs] #np.hstack([np.linalg.norm(flat_grad[layer].reshape([model_repr[layer + ".weight"].shape[0], -1])) for layer in list(flat_grad.keys())[::-1]])[idx_obj]
         
        feats = np.hstack([model_feats_weight_norm] + clean_grad_feats_weight_norm)
        
        return np.asarray(feats).tolist()    

    def infer_layer_features_from_models(self, model_path_list, train= False):
        logging.info(f"Loading %d models...", len(model_path_list))
        all_feats = []
        if train:
            max_num_feats = 0
            for model_filepath in model_path_list:
                feats = self.infer_layer_features_from_one_model(model_filepath, train)
                max_num_feats = len(feats) if len(feats) > max_num_feats else max_num_feats
                all_feats.append(feats)
            for i in range(len(all_feats)):
                all_feats[i] = all_feats[i] + [0. for _ in range(max_num_feats - len(all_feats[i]))]
            self.input_features = max_num_feats
        else:
            max_num_feats = self.input_features
            for model_filepath in model_path_list:
                feats = self.infer_layer_features_from_one_model(model_filepath, train) + [0. for _ in range(max_num_feats - len(all_feats[i]))]
                all_feats.append(feats)
             
        return all_feats

    def infer_layer_features_from_one_model(self, model_filepath, train = False):
        model_dict, model_repr_dict, _, clean_example_dict, _ = load_models_dirpath([model_filepath])
        model_class, [model] = model_dict.popitem()
         
        _, [clean_example] = clean_example_dict.popitem()
        _, [model_repr] = model_repr_dict.popitem()
    
        clean_grad_reprs = inference_on_example_data(model, '1', clean_example, self.scale_parameters_filepath, grad = True)
        model_feats = []
        clean_grad_feats = []
        for (layer, param) in model_repr_dict.items():
            if len(param.shape) == 1:
                param = param.reshape((1, -1))
            model_feats.append(self.infer_layer_features(param))
        for clean_grad_repr in clean_grad_reprs:
            for (layer, param) in clean_grad_repr.items():
                if len(param.shape) == 1:
                    param = param.reshape((1, -1))
                clean_grad_feats.append(self.infer_layer_features(param))

        feats = np.hstack(model_feats + clean_grad_feats).reshape((-1))
        #print(feats.shape)
        return feats.tolist() + ([] if train else [0. for _ in range(self.input_features - feats.shape[0])])


    def infer_layer_features(self, param, nbins=100,szcap=4096):
        # baseline 0.73 at radom_state=1 
        # refer to https://github.com/frkl/trojai-cyber-pdf/blob/1b1382d29e5accfe69c6ef124f9a28a8a33b1084/weight_analysis.py#L52
        nx=param.shape[1]
        ny=param.shape[0]
        n=nbins #min(max(nx,ny),szcap)
        m=min(nx,ny);
        z=np.zeros((n,n))
        #z[:min(ny,n),:min(nx,n)]=param[:min(ny,n),:min(nx,n)];
        z[:min(ny, n), :min(nx, n)] = param[::max(1,int(ny/n)), ::max(1,int(nx/n))][:min(ny, n), :min(nx, n)]
        #z = np.dot(np.dot(np.random.random([n, param.shape[0]]), param), np.random.random([param.shape[1], n]))
        #fv = np.hstack(z)
        #return [fv]
        
        #1) Get eigen values
        e,_=np.linalg.eig(z);
        #Get histogram of abs, real, imaginary
        e2=(e.real**2+e.imag**2)**0.5;
        e_hist = hist_v(e, nbins)
        e2_hist=hist_v(e2,nbins);
        er_hist=hist_v(e.real,nbins);
        ec_hist=hist_v(e.imag,nbins);
        
        #2) histogram of eig persistence
        cm=AgglomerativeClustering(distance_threshold=0, n_clusters=None,linkage='single')
        s=np.vstack((e.real,e.imag))
        cm=cm.fit(s)
        d= cm.distances_;
        eig_persist=hist_v(d,nbins)
        
        #3) Get histogram of weight value and abs 
        w=param.reshape([-1]);
        w_hist=hist_v(w,nbins);
        wabs_hist=hist_v(np.abs(w),nbins);

        #4) SVD
        _, s, _ = np.linalg.svd(param)
        s_hist = hist_v(s, nbins)
         

        #fv=np.hstack((e2_hist,er_hist,ec_hist,eig_persist,w_hist,wabs_hist)); # 0.76
        #fv=np.hstack((np.hstack(z), e2_hist,er_hist,ec_hist,eig_persist,w_hist,wabs_hist))#
        #fv=np.hstack((er_hist,ec_hist, eig_persist, w_hist)); # 0.69
        #fv=np.hstack((w_hist,wabs_hist)) # 0.68
        #fv = np.hstack((e2_hist,er_hist,ec_hist,w_hist, wabs_hist, [np.linalg.norm(z)], [np.mean(z)], [np.median(z)])) # 0.72
        #fv = np.hstack((e2_hist,er_hist,ec_hist,w_hist, wabs_hist, [np.linalg.norm(param)], [np.mean(param)], [np.median(param)])) #0.71
        #fv = np.hstack((e2_hist,er_hist,ec_hist, [np.linalg.norm(np.abs(param))], [np.mean(np.abs(param))], [np.median(np.abs(param))], [np.linalg.norm(param)], [np.mean(param)], [np.median(param)])) #0.67 xgboost || 0.69 svm
        #fv = np.hstack((e2_hist,er_hist,ec_hist,w_hist, wabs_hist, [np.linalg.norm(np.abs(param))], [np.mean(np.abs(param))], [np.median(np.abs(param))], [np.linalg.norm(param)], [np.mean(param)], [np.median(param)]))  #0.75 xgboost
        fv = np.hstack((s_hist, w_hist, [np.linalg.norm(param)]))  #0.76 xgboost
        
        return [fv];
 

    def infer_one_model(self, model_filepath):
        layer_transform = pickle.load(open(self.layer_transform_filepath, 'rb'))
        clean_grad_layer_transform = pickle.load(open(self.clean_grad_layer_transform_filepath, 'rb'))
        model_layer_map = pickle.load(open(self.model_layer_map_filepath, 'rb'))

        model_dict, model_repr_dict, _, clean_example_dict, _ = load_models_dirpath([model_filepath])
        model_class, [model] = model_dict.popitem()
         
        _, [clean_example] = clean_example_dict.popitem()
        _, [model_repr] = model_repr_dict.popitem()
        
        flat_model = flatten_model(model_repr, model_layer_map[model_class])
        #del model_repr
        print(model_repr.keys(), flat_model.keys())

        clean_grad_reprs = inference_on_example_data(model, '1', clean_example, self.scale_parameters_filepath, grad = True)
        del model

        flat_grads = flatten_models({model_class: clean_grad_repr}, model_layer_map[model_class])[model_class]
        flat_grad = {}
        for i in range(len(flat_grads)):
            for (k, v) in flat_grads.items():
                if flat_grad.get(k, None) is None:
                    flat_grad[k] = np.arrary([[]])
                flat_grad[k] = np.vstack((flat_grads[k], flat_grads[i][k]))
        for (k, v) in flat_grad.items():
            flat_grad[k] = np.mean(flat_grad[k], axis = 0)
        #del clean_grad_repr

        
        model_feats_ICA = use_feature_reduction_algorithm(
                    layer_transform[model_class], flat_model
                )
        clean_grad_feats_ICA = use_feature_reduction_algorithm(
                    clean_grad_layer_transform[model_class], flat_grad
                )
                #poisoned_grad_feats = use_feature_reduction_algorithm(
                #    poisoned_grad_layer_transform[model_class], flat_poisoned_grad_repr_dict[model_class][i]
                #)

        feats = np.hstack((model_feats_ICA, clean_grad_feats_ICA)) #, poisoned_grad_feats)).tolist()
        feats = np.pad(feats, [(0, 0), (0, 2 * self.ICA_features - feats.shape[-1])], mode='constant')
        
        
        # Pure weight parameters as features
        model_feats = np.hstack(list(flat_model.values())[len(flat_model)-1:0:-1])[::1]#[::int(model_feats.shape[-1] / self.ICA_features)]
        clean_grad_feats = np.hstack(list(flat_grad.values())[len(flat_grad)-1:0:-1])[::1]#[::int(clean_grad_feats.shape[-1] / self.ICA_features)]
        feats = [model_feats.tolist()[:self.ICA_features] + clean_grad_feats.tolist()[:self.ICA_features]]
        """
        
        # Weight l2-norms as features
        model_feats = np.hstack([np.linalg.norm(value, ord = 2) for value in list(flat_model.values())[::-1]])[:2] 
        clean_grad_feats = np.hstack([np.linalg.norm(value, ord = 2) for value in list(flat_grad.values())[::-1]])[:2] 
        feats = [model_feats.tolist() + clean_grad_feats.tolist()]
        print(feats)
        

        idx_obj = [-1, -2, 0, 1] #slice(-1, -3, -1) #slice(2) #
        # Weight matrix-norms as features
        weight_norm_fn = lambda model: np.hstack([np.linalg.norm(model_repr[layer + ".weight"]) for layer in list(model.keys())[::-1]])[idx_obj]
        bias_norm_fn = lambda model: np.hstack([np.linalg.norm(model_repr[layer + ".bias"], ord = 2) for layer in list(model.keys())[::-1]])[idx_obj]
        mean_fn = lambda model: np.hstack([np.mean(model[layer]) for layer in list(model.keys())[::-1]])[idx_obj]
        std_fn = lambda model: np.hstack([np.std(model[layer]) for layer in list(model.keys())[::-1]])[idx_obj]
        max_fn = lambda model: np.hstack([np.max(model[layer]) for layer in list(model.keys())[::-1]])[idx_obj]
        min_fn = lambda model: np.hstack([np.min(model[layer]) for layer in list(model.keys())[::-1]])[idx_obj]
        median_fn = lambda model: np.hstack([np.median(model[layer]) for layer in list(model.keys())[::-1]])[idx_obj]
        sing_fn = lambda model: np.hstack([np.sort(np.linalg.svd(model[layer].reshape([model_repr[layer + ".weight"].shape[0], -1]))[1])[-1] for layer in list(model.keys())[::-1]])[idx_obj]
        sing_median_fn = lambda model: np.hstack([np.median(np.linalg.svd(model[layer].reshape([model_repr[layer + ".weight"].shape[0], -1]))[1]) for layer in list(model.keys())[::-1]])[idx_obj]
        
        
        model_feats_weight_norm = weight_norm_fn(flat_model) #np.hstack([np.linalg.norm(flat_model[layer].reshape([model_repr[layer + ".weight"].shape[0], -1])) for layer in list(flat_model.keys())[::-1]])[idx_obj]
        clean_grad_feats_weight_norm = weight_norm_fn(flat_grad) #np.hstack([np.linalg.norm(flat_grad[layer].reshape([model_repr[layer + ".weight"].shape[0], -1])) for layer in list(flat_grad.keys())[::-1]])[idx_obj]
        model_feats_bias_norm = bias_norm_fn(flat_model) #np.hstack([np.linalg.norm(flat_model[layer].reshape([model_repr[layer + ".weight"].shape[0], -1])) for layer in list(flat_model.keys())[::-1]])[idx_obj]
        clean_grad_feats_bias_norm = bias_norm_fn(flat_grad) #np.hstack([np.linalg.norm(flat_grad[layer].reshape([model_repr[layer + ".weight"].shape[0], -1])) for layer in list(flat_grad.keys())[::-1]])[idx_obj]
        model_feats_mean = mean_fn(flat_model) #np.hstack([np.mean(flat_model[layer]) for layer in list(flat_model.keys())[::-1]])[idx_obj]
        clean_grad_feats_mean = mean_fn(flat_grad) #np.hstack([np.mean(flat_grad[layer]) for layer in list(flat_grad.keys())[::-1]])[idx_obj]
        model_feats_std = std_fn(flat_model) #np.hstack([np.std(flat_model[layer]) for layer in list(flat_model.keys())[::-1]])[idx_obj]
        clean_grad_feats_std = std_fn(flat_grad) #np.hstack([np.std(flat_grad[layer]) for layer in list(flat_grad.keys())[::-1]])[idx_obj]
        model_feats_max = max_fn(flat_model) #np.hstack([np.max(flat_model[layer]) for layer in list(flat_model.keys())[::-1]])[idx_obj]
        clean_grad_feats_max = max_fn(flat_grad) #np.hstack([np.max(flat_grad[layer]) for layer in list(flat_grad.keys())[::-1]])[idx_obj]
        model_feats_min = min_fn(flat_model) #np.hstack([np.min(flat_model[layer]) for layer in list(flat_model.keys())[::-1]])[idx_obj]
        clean_grad_feats_min = min_fn(flat_grad) #np.hstack([np.min(flat_grad[layer]) for layer in list(flat_grad.keys())[::-1]])[idx_obj]
        model_feats_median = median_fn(flat_model) #np.hstack([np.median(flat_model[layer]) for layer in list(flat_model.keys())[::-1]])[idx_obj]
        clean_grad_feats_median = median_fn(flat_grad) #np.hstack([np.median(flat_grad[layer]) for layer in list(flat_grad.keys())[::-1]])[idx_obj]#[:2] 
        model_feats_sing_vals = sing_fn(flat_model) #np.hstack([np.sort(np.linalg.svd(flat_model[layer].reshape([model_repr[layer + ".weight"].shape[0], -1]))[1])[-1: -3: -1] for layer in list(flat_model.keys())[::-1]])[idx_obj]
        clean_grad_feats_sing_vals = sing_fn(flat_grad) #np.hstack([np.sort(np.linalg.svd(flat_grad[layer].reshape([model_repr[layer + ".weight"].shape[0], -1]))[1])[-1: -3: -1] for layer in list(flat_grad.keys())[::-1]])[idx_obj]
        model_feats_sing_median = sing_median_fn(flat_model)  
        clean_grad_feats_sing_median = sing_median_fn(flat_grad)  
        
        
        feats_ = [
            model_feats_weight_norm.tolist() + clean_grad_feats_weight_norm.tolist() + \
            model_feats_bias_norm.tolist() + clean_grad_feats_bias_norm.tolist()] # + \
           
            model_feats_mean.tolist() + clean_grad_feats_mean.tolist() + \
            #model_feats_std.tolist() + clean_grad_feats_std.tolist() + \
            model_feats_max.tolist() + clean_grad_feats_max.tolist() + \
            model_feats_min.tolist() + clean_grad_feats_min.tolist() + \
            model_feats_median.tolist() + clean_grad_feats_median.tolist() #+ \
            #model_feats_sing_vals.tolist() + clean_grad_feats_sing_vals.tolist() + \
            #model_feats_sing_median.tolist() + clean_grad_feats_sing_median.tolist()
            ]
                
        feats = feats_ #np.hstack((feats, feats_))
        """ 
        print(feats)
        
        #assert np.asarray(feats).shape[-1] == 2 * self.ICA_features
        return np.asarray(feats).tolist()
    
    def infer_attribution_feature_from_models(self, model_path_list, num_data_per_model = 20, train= False):
        logging.info(f"Loading %d models...", len(model_path_list))
        all_feats = []
        if train:
            max_num_feats = 0
            for model_filepath in model_path_list:
                feats = self.infer_attribution_feature_from_one_model(model_filepath, num_data_per_model, train)
                max_num_feats = len(feats) if len(feats) > max_num_feats else max_num_feats
                all_feats.append(feats)
            if len(all_feats[0].shape) == 1 or all_feats[0].shape[0] == 1:
                 for i in range(len(all_feats)):
                    all_feats[i] = all_feats[i] + [0. for _ in range(max_num_feats - len(all_feats[i]))]
            self.input_features = max_num_feats
        else:
            max_num_feats = self.input_features
            for model_filepath in model_path_list:
                feats = self.infer_attribution_feature_from_one_model(model_filepath, num_data_per_model, train)
                if len(all_feats[0].shape) == 1 or all_feats[0].shape[0] == 1:
                    feats = feats + [0. for _ in range(max_num_feats - len(all_feats[i]))]
          
                all_feats.append(feats)
             
        return all_feats

    def infer_attribution_feature_from_one_model(self, model_filepath, num_data_per_model, train = False):
        model_dict, model_repr_dict, _, clean_example_dict, _ = load_models_dirpath([model_filepath])
        model_class, [model] = model_dict.popitem()
         
        _, [clean_examples] = clean_example_dict.popitem()
        _, [model_repr] = model_repr_dict.popitem()
    
        attrs = []
        if num_data_per_model > clean_examples.shape[0]:
            example_ids = np.hstack((np.arange(clean_examples.shape[0]), np.random.choice(clean_examples.shape[0], num_data_per_model - clean_examples.shape[0])))
        else:
            example_ids = np.arange(clean_examples.shape[0])
        examples = clean_examples[example_ids]
        for clean_example in examples:
            attrs.append(get_attribution_from_example_data(model, '1', [clean_example], self.scale_parameters_filepath).reshape((-1)))
        return np.hstack((examples, attrs)) #np.asarray(attrs) #([] if train else [0. for _ in range(self.input_features - attrs.shape[0])])


if __name__ == "__main__":
    extractor = FeatureExtractor("./metaparameters.json", "./learned_parameters",  "./learned_parameters/scale_params.npy")
    #extractor.manual_configure("/mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round12/models")
    #print(extractor.infer_one_model("./model/id-00000002/"))

    extractor.infer_norms_from_models("/mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round12/models")
    