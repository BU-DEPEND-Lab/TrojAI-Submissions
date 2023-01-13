import importlib

import numpy as np
from tqdm import tqdm
import torch
from attrdict import AttrDict


def feature_reduction(model, weight_table, max_features):
    outputs = {}
    # number of features per layer
    tf = max_features / len(model)
    # number of weights in the model
    sm = sum([l.shape[0] for l in model.values()])
     
    for (layer, weights) in model.items():
        # Downsample 100 weights in total
        # Allocate proportionally to each layer
        # Each layer is designated for a number of weights < 100
        wt_i = np.round(weights.shape[0] / sm * 100).astype(np.int32)
        out_f = int(weight_table[wt_i] * tf)
        #print("out_f: ", out_f, "wt_i: ", wt_i, "weight_table[wt_i]: ", weight_table[wt_i], "tf: ", tf)
        if layer == list(model.keys())[-1]:
            out_f = max_features - sum(outputs.values())
        assert out_f > 0
        outputs[layer] = out_f
    #print("feature_reduction output", outputs)
    return outputs


def init_feature_reduction(output_feats):
    fr_algo = "sklearn.decomposition.FastICA"
    fr_algo_mod = ".".join(fr_algo.split(".")[:-1])
    fr_algo_class = fr_algo.split(".")[-1]
    mod = importlib.import_module(fr_algo_mod)
    fr_class = getattr(mod, fr_algo_class)
    return fr_class(n_components=output_feats)


def init_weight_table(random_seed, mean, std, scaler):
    rnd = np.random.RandomState(seed=random_seed)
    return np.sort(rnd.normal(mean, std, 100)) * scaler


def fit_feature_reduction_algorithm(model_dict, weight_table_params, input_features):
    layer_transform = {}
    weight_table = init_weight_table(**weight_table_params)

    for (model_arch, models) in model_dict.items():
        layers_output = feature_reduction(models[0], weight_table, input_features)
        layer_transform[model_arch] = {}
        for (layers, output) in tqdm(layers_output.items()):
            
            layer_transform[model_arch][layers] = init_feature_reduction(output)
            s = np.stack([model[layers] for model in models])
            print("Need to fit matrix size: ", [model[layers].shape for model in models], "amounting to ", s.shape)
            layer_transform[model_arch][layers].fit(s)

    return layer_transform
 
def model_transformer(outputs, layer):
    num_eigen_vals = outputs[layer] - 3
    def transform(weights):
        nonlocal num_eigen_vals
        mean = np.mean(weights)
        std = np.std(weights)
        l2_norm = np.linalg.norm(weights, ord = 2)
        if num_eigen_vals > 0:
            eigen_vals = np.linalg.eig(np.asarray(weights).T * np.asarray(weights))[:num_eigen_vals]
        return [mean, std, l2_norm] + eigen_vals.tolist()
    return transform

def stat_feature_reduction_algorithm(model_dict, input_features):
    layer_transform = {}
    
    for (model_arch, models) in model_dict.items():
        outputs = {}
        # number of features per layer
        tf = input_features / len(models[0])
        # number of weights in the model\
        #sm = sum([l.shape[0] for l in models[0].values()])
        for (layer, weights) in models[0].items():
            # Downsample number of weights in total
            # Allocate proportionally to each layer
            # Each layer is designated for a number of weights  
            wt_i = tf
            #wt_i = np.round(weights.shape[0] / sm * input_features).astype(np.int32)
            #print("out_f: ", out_f, "wt_i: ", wt_i, "weight_table[wt_i]: ", weight_table[wt_i], "tf: ", tf)
            if layer == list(models[0].keys())[-1]:
                wt_i = input_features - sum(outputs.values())
            assert wt_i > 0
            outputs[layer] = int(wt_i)
        #print("model transformer outputs:", outputs)
        
        layer_transform[model_arch] = {}
        for (layer, weights) in models[0].items():
            layer_transform[model_arch][layer] = AttrDict({'transform': model_transformer(outputs, layer)})
    return layer_transform

def use_feature_reduction_algorithm(layer_transform, layer_features, flat_models):
    out_models = []
    for flat_model in flat_models:
        out_model = np.array([[]])
        print(flat_model)
        for (layer, weights) in flat_model.items():
            #out_model = np.hstack((out_model, layer_transform[layer].transform([weights])))
            if layer_transform is None:
                out_model = np.hstack((out_model,  np.expand_dims(layer_features[layer].transform([weights])[0], axis = 0)))
            elif layer_features is None:
                out_model = np.hstack((out_model,  np.expand_dims(layer_transform[layer].transform([weights])[0], axis = 0)))
            else:
                out_model = np.hstack((out_model,  
                        np.expand_dims(np.concatenate((\
                        layer_transform[layer].transform([weights])[0], \
                        layer_features[layer].transform([weights])[0]), axis = None), axis = 0)
                        ))
        out_models.append(out_model)
    return np.mean(out_models, axis = 0)
    
 