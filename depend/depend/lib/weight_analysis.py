import json
import logging
import pickle
from os import listdir,
import numpy as np
#from sklearn.ensemble import RandomForestRegressor

from utils.flatten import flatten_model, flatten_models 
from utils.models import create_layer_map 
from utils.reduction import ( 
    use_feature_reduction_algorithm, 
    fit_ICA_feature_reduction_algorithm 
)
 
from sklearn.cluster import AgglomerativeClustering
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
        self.target_model_layer_map_filepath = join(self.learned_parameters_dirpath, "model_layer_map.bin")
        self.layer_transform_filepath = join(self.learned_parameters_dirpath, "layer_transform.bin")
        self.clean_grad_layer_transform_filepath = join(self.learned_parameters_dirpath, "clean_grad_layer_transform.bin")
        self.poisoned_grad_layer_transform_filepath = join(self.learned_parameters_dirpath, "poisoned_grad_layer_transform.bin")

        # TODO: Update skew parameters per round
 
        self.input_features = metaparameters["train_input_features"]
        self.ICA_features = metaparameters["train_ICA_features"]
 
 
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
            "train_ICA_features": self.ICA_features,
            "train_weight_table_random_state": self.weight_table_params["random_seed"],
            "train_weight_table_params_mean": self.weight_table_params["mean"],
            "train_weight_table_params_std": self.weight_table_params["std"],
            "train_weight_table_params_scaler": self.weight_table_params["scaler"],
        }

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
        
        with open(self.target_model_layer_map_filepath, "wb") as fp:
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


     
if __name__ == "__main__":
    extractor = FeatureExtractor("./metaparameters.json", "./learned_parameters",  "./learned_parameters/scale_params.npy")
    #extractor.manual_configure("/mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round12/models")
    #print(extractor.infer_one_model("./model/id-00000002/"))

    extractor.infer_norms_from_models("/mnt/md0/shared/TrojAI-Submissions/trojai-datasets/round12/models")
    