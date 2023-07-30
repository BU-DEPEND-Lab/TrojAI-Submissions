import importlib

import numpy as np
from tqdm import tqdm
import torch
#from attrdict import AttrDict


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
        wt_i = np.round(np.prod(weights.shape) / sm * 100).astype(np.int32)
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


def histv(w,bins=100):
    s =w.sort(axis=0);
    wmin=float(w.min());
    wmax=float(w.max());
    
    n=s.shape[0];
    hist=np.zeros([bins]);
    for i in range(bins):
        x=math.floor((n-1)/(bins-1)*i)
        x=max(min(x,n),0);
        v=float(s[x]);
        hist[i]=v;
    
    return hist

def model_transformer(layers_grad, layer):
    num_eigen_vals = layers_grad[layer]
    def transform(weights):
        nonlocal num_eigen_vals
        weights = np.reshape(weights, [1, -1])
        #mean = np.mean(weights)
        #std = np.std(weights)
        #l2_norm = np.linalg.norm(weights, ord = 2)
   
        eigen_vals, _ = np.linalg.eig(np.asarray(weights).T * np.asarray(weights))
        eigen_sqrt = (eigen_vals.real**2 + eigen_vals.imag**2)**0.5
        eigen_hist = histv(eigen_sqrt, num_eigen_vals)
        eigen_real_hist = histv(eigen_vals.real, num_eigen_vals)
        eigen_img_hist = histv(eigen_vals.imag, num_eigen_vals)
        
        return eigen_hist.tolist() + eigen_real_hist.tolist() + eigen_img_hist.tolist()
 
    return transform

def grad_feature_reduction_algorithm(model_dict, weight_table_params, input_features):
    layer_transform = {}
    weight_table = init_weight_table(**weight_table_params)
    for (model_arch, models) in model_dict.items():
        layers_grad = feature_reduction(models[0], weight_table, input_features)
        layer_transform[model_arch] = {}
        for (layers, grad) in tqdm(layers_grad.items()):
            layer_transform[model_arch][layers] = init_feature_reduction(grad)
            s = np.mean([model[layers] for model in models], axis = 0).reshape(1, -1)
            s = np.dot(s.T, s)
            #print("Need to fit matrix size: ", [model[layers].shape for model in models], "amounting to ", s.shape)
            layer_transform[model_arch][layers].fit(s)

        #for (layer, weights) in models[0].items():
        #    layer_transform[model_arch][layer] = AttrDict({'transform': model_transformer(layers_grad, layer)})
        
    return layer_transform

def use_feature_reduction_algorithm(layer_features, flat_model):
    
    out_model = np.array([[]])

    for (layer, weights) in flat_model.items():
        #print(layer)
        f = np.expand_dims(layer_features[layer].transform([weights])[0], axis = 0)
        print(f"Layer {layer} number of features {f}")
        out_model = np.hstack((out_model,  f))
        """
        #out_model = np.hstack((out_model, layer_transform[layer].transform([weights])))
        if layer_transform is None:
            print("layer_transform:", np.expand_dims(layer_features[layer].transform([weights])[0], axis = 0).shape, out_model.shape)
            out_model = np.hstack((out_model,  np.expand_dims(layer_features[layer].transform([weights])[0], axis = 0)))
        elif layer_features is None:
            print("layer_feature:", np.expand_dims(layer_transform[layer].transform([weights])[0], axis = 0).shape, out_model.shape)
            out_model = np.hstack((out_model,  np.expand_dims(layer_transform[layer].transform([weights])[0], axis = 0)))
        else:
            out_model = np.hstack((out_model,  
                    np.expand_dims(np.concatenate((\
                    layer_transform[layer].transform([weights])[0], \
                    layer_features[layer].transform([weights])[0]), axis = None), axis = 0)
                    ))
        """
    print(f"Amount to feature shape {out_model.shape}")

    return out_model
    
def fit_ICA_feature_reduction_algorithm(model_dict, weight_table_params, input_features):
    layer_transform = {}
    weight_table = init_weight_table(**weight_table_params)

    for (model_arch, models) in model_dict.items():
    
        layers_output = feature_reduction(models[0], weight_table, input_features)
        
        assert sum(list(layers_output.values())) == input_features
        layer_transform[model_arch] = {}
        for (layers, output) in tqdm(layers_output.items()):
            layer_transform[model_arch][layers] = init_feature_reduction(output)
            s = np.stack([model[layers] for model in models] * int(output / len(models) + 1))
            print(f"Need to fit matrix size: {int(output / len(models) + 1)} x ", [model[layers].shape for model in models], "amounting to ", s.shape)
            layer_transform[model_arch][layers].fit(s)
            print(f"Model class {model_arch} layer {layers} fit ICA components {layer_transform[model_arch][layers].components_.shape}")
        print(f"Feature reduction to {sum(list(layers_output.values()))} number of features")

    return layer_transform

 