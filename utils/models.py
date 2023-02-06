import re
from collections import OrderedDict
from os.path import join
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler
from archs import Net2, Net3, Net4, Net5, Net6, Net7, Net2r, Net3r, Net4r, Net5r, Net6r, Net7r, Net2s, Net3s, Net4s, Net5s, Net6s, Net7s
from abc import ABC, abstractmethod

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
    examples = None
    for examples_dir_entry in os.scandir(join(model_dirpath, "clean-example-data" if clean else "poisoned-example-data")):
            if examples_dir_entry.is_file() and examples_dir_entry.name.endswith(".npy"):
                feature_vector = np.load(examples_dir_entry.path).reshape(1, -1)
                if examples is None:
                    examples = feature_vector
                else:
                    examples = np.vstack((examples, feature_vector))

    return examples

def load_models_dirpath(models_dirpath):
    model_dict = {}
    model_repr_dict = {}
    model_ground_truth_dict = {}
    clean_example_dict = {}
    poisoned_example_dict = {}

    for model_path in tqdm(models_dirpath):
        model, model_repr, model_class = load_model(
            join(model_path, "model.pt")
        )
        
        # Build the list of models
        if model_class not in model_repr_dict.keys():
            model_dict[model_class] = []
            model_repr_dict[model_class] = []
            model_ground_truth_dict[model_class] = []
            clean_example_dict[model_class] = []
            poisoned_example_dict[model_class] = []
        model_dict[model_class].append(model)
        model_repr_dict[model_class].append(model_repr)

        try:
            model_ground_truth = load_ground_truth(model_path)
            model_ground_truth_dict[model_class].append(model_ground_truth)
        except:
            print("Can't find ground truth")
            pass
        try:
            clean_examples = load_examples(model_path)
            clean_example_dict[model_class].append(clean_examples)
        except:
            print("No clean example")
            pass
        try:
            poisoned_examples = load_examples(model_path, False)
            poisoned_examples[model_class].append(poisoned_examples)
        except:
            print("No poisoned example")
            pass
    return model_dict, model_repr_dict, model_ground_truth_dict, clean_example_dict, poisoned_example_dict


def inference_on_example_data(model, ground_truth, examples, scale_parameters_filepath, grad = np.mean):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model: the pytorch model
            examples_dirpath: the directory path for the round example data
        """
        #print(f"Inference on example data {example}")
        # Setup scaler
        scaler = StandardScaler()

        scale_params = np.load(scale_parameters_filepath)

        scaler.mean_ = scale_params[0]
        scaler.scale_ = scale_params[1]

        
         
        # Inference on models
        grad_reprs = []
        #print(">>>>>>> Example feature shape: ", examples.shape)
        #print(">>>>>>> Scaler shape: ", scaler.mean_.shape, scaler.scale_.shape)
        for example in examples:
            feature_vector = torch.from_numpy(scaler.transform(np.asarray([example]).astype(float))).float()
            model.zero_grad()
            #pred = torch.argmax(model(feature_vector).detach()).item()
            scores = model(feature_vector)
            pred = torch.argmax(scores).detach()
            logits = F.log_softmax(scores, dim = 1)
            
            #print("Ground Truth: {}, Prediction: {}".format(ground_truth, str(pred)))
        
            if grad is not None:
                loss = F.cross_entropy(logits, torch.LongTensor(logits.shape[0] * [int(ground_truth)]))
                loss.backward();
                grad_repr = OrderedDict(
                    {layer: param.data.numpy() for ((layer, _), param) in zip(model.state_dict().items(), model.parameters())}
                ) 
                grad_reprs.append(grad_repr) 
        return grad_reprs

def get_attribution_from_example_data(model, ground_truth, examples, scale_parameters_filepath, grad = np.hstack):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model: the pytorch model
            examples_dirpath: the directory path for the round example data
        """
        #print(f"Inference on example data {example}")
        # Setup scaler
        scaler = StandardScaler()

        scale_params = np.load(scale_parameters_filepath)

        scaler.mean_ = scale_params[0]
        scaler.scale_ = scale_params[1]

        
         
        # Inference on models
        grad_reprs = []
        #print(">>>>>>> Example feature shape: ", examples.shape)
        
        #print(">>>>>>> Scaler shape: ", scaler.mean_.shape, scaler.scale_.shape)
        
        for example in examples:
            #print(">>>>>>> Example: ")
            #for e in example:
            #    print(e)
            feature_vector = torch.from_numpy(scaler.transform(np.asarray([example]).astype(float))).float()
            feature_vector.requires_grad_()
            model.zero_grad()
            #pred = torch.argmax(model(feature_vector).detach()).item()
            scores = model(feature_vector)
            pred = torch.argmax(scores).detach()
            logits = F.log_softmax(scores, dim = 1)
            #print(logits)
            #print("Ground Truth: {}, Prediction: {}".format(ground_truth, str(pred)))
        
            if grad is not None:
                loss = F.cross_entropy(logits, torch.LongTensor(logits.shape[0] * [int(ground_truth)]))
                loss.backward();
                grad_reprs.append(torch.mul(feature_vector.grad.data.flatten(), feature_vector.flatten()).detach().numpy()) 
        return grad(grad_reprs)

    
"""
def focal_binary_object(label, y_pred):
    nonlocal alpha, lam
    #label = dmat.get_label()
    # l(y_val, y_pred) = - y_val * alpha * (1 - y_pred)^lam * log y_pred - (1 - y_val) * (1 - alpha) * y_pred^lam * log (1 - y_pred)
    # grad(l, y_pred) =  y_val * alpha * lam * (1 - y_pred)^{lam - 1} * log y_pred - (1 - y_val) * (1 - alpha) * lam * y_pred^{lam - 1} * log (1 - y_pred) - y_val * alpha * (1 - y_pred)^lam * 1 / y_pred + (1 - y_val) * (1 - alpha) * y_pred^lam * 1/ (1 - y_pred)
    # Hess(l, y_pred) =  - y_val * alpha * lam * (lam - 1) * (1 - y_val)^{lam - 2} * log y_val - (1 - y_val) * (1 - alpha) * lam * (lam - 1) * y_pred^{lam - 2} * log (1 - y_pred)
    y_pred[y_pred == 0] = 1e-6
    y_pred[y_pred == 1] = 1. - 1e-6
    print(np.count_nonzero(np.isnan(y_pred)), np.count_nonzero(y_pred))
    #print(label)
    grad = label * alpha * lam * np.power(1 - y_pred, lam - 1) * np.log(y_pred) + \
            - label * alpha * np.power(1 - y_pred, lam) * 1. / y_pred + \
            - (1 - label) * (1 - alpha) * lam * np.power(y_pred, lam - 1) * np.log(1 - y_pred) + \
            + (1 - label) * (1 - alpha) * np.power(y_pred, lam) * 1./(1 - y_pred)
    hess =  - label * alpha * lam * (lam - 1) * np.power(1 - y_pred, lam - 2) * np.log(y_pred) + label * alpha * lam * np.power(1 - y_pred, lam - 1) * 1. / y_pred + \
            + label * alpha * lam * np.power(1 - y_pred, lam - 1) * 1. / y_pred + label * alpha * np.power(1 - y_pred, lam) * 1. /( y_pred *  y_pred) + \
            - (1 - label) * (1 - alpha) * lam * (lam - 1) * np.power(y_pred, lam - 2) * np.log (1 - y_pred) + (1 - label) * (1 - alpha) * lam * np.power(y_pred, lam - 1) * 1. / (1 - y_pred) + \
            + (1 - label) * (1 - alpha) * lam * np.power(y_pred, lam - 1) * 1./(1 - y_pred) + (1 - label) * (1 - alpha) * np.power(y_pred, lam) * 1./((1 - y_pred) * (1 - y_pred))
    return grad, hess  

def focal_binary_object(label, pred):
    nonlocal alpha, lam
    #gamma_indct = self.gamma_indct
    # retrieve data from dtrain matrix
    #label = dtrain.get_label()
    # compute the prediction with sigmoid
    sigmoid_pred = 1.0 / (1.0 + np.exp(-pred))
    # gradient
    # complex gradient with different parts
    g1 = sigmoid_pred * (1 - sigmoid_pred)

    g2 = label + ((-1) ** label) * sigmoid_pred
    g3 = sigmoid_pred + label - 1
    g4 = 1 - label - ((-1) ** label) * sigmoid_pred
    g5 = label + ((-1) ** label) * sigmoid_pred
    # combine the gradient
    grad = alpha * g3 * robust_pow(g2, lam) * np.log(g4 + 1e-9) + \
        ((-1) ** label) * robust_pow(g5, (lam + 1))
    # combine the gradient parts to get hessian components
    hess_1 = robust_pow(g2, lam) + \
            lam * ((-1) ** label) * g3 * robust_pow(g2, (lam - 1))
    hess_2 = ((-1) ** label) * g3 * robust_pow(g2, lam) / g4
    # get the final 2nd order derivative
    hess = ((hess_1 * np.log(g4 + 1e-9) - hess_2) * alpha +
            (lam + 1) * robust_pow(g5, lam)) * g1

    return grad, hess

def focal_binary_object(label, pred):
    # retrieve data from dtrain matrix
    #label = dtrain.get_label()
    # compute the prediction with sigmoid
    # compute the prediction with sigmoid
"""

class focal_loss:
    def __init__(self, gamma_indct = 1.2, prob = lambda pred: 1.0 / (1.0 + np.exp(-pred))):
        self.gamma_indct = gamma_indct
        self.prob = prob

    def __call__(self, label, pred):
        # retrieve data from dtrain matrix
        #label = dtrain.get_label()
        # compute the prediction with sigmoid
        # compute the prediction with sigmoid
        sigmoid_pred = self.prob (pred)
        #1.0 / (1.0 + np.exp(-pred))
        #sigmoid_pred = pred
        # gradient
        # complex gradient with different parts
        grad_first_part = (label+((-1)**label)*sigmoid_pred)**self.gamma_indct
        grad_second_part = label - sigmoid_pred
        grad_third_part = self.gamma_indct*(1-label-sigmoid_pred)
        grad_log_part = np.log(1-label-((-1)**label)*sigmoid_pred + 1e-7)       # add a small number to avoid numerical instability
        # combine the gradient
        grad = -grad_first_part*(grad_second_part+grad_third_part*grad_log_part)
        # combine the gradient parts to get hessian
        hess_first_term = self.gamma_indct*(label+((-1)**label)*sigmoid_pred)**(self.gamma_indct-1)*sigmoid_pred*(1.0 - sigmoid_pred)*(grad_second_part+grad_third_part*grad_log_part)
        hess_second_term = (-sigmoid_pred*(1.0 - sigmoid_pred)-self.gamma_indct*sigmoid_pred*(1.0 - sigmoid_pred)*grad_log_part-((1/(1-label-((-1)**label)*sigmoid_pred))*sigmoid_pred*(1.0 - sigmoid_pred)))*grad_first_part
        # get the final 2nd order derivative
        hess = -(hess_first_term+hess_second_term)
        return grad, hess
     
     
class get_loss:
    def __init__(self, loss, use_sigmoid = False):
        self.loss = loss
        self.use_sigmoid = use_sigmoid

    def get_objective(self, *args, **kwargs):
        if ":" not in self.loss and callable(self.loss):
            kwargs.update({"prob": self.prob})
            return loss(*args, **kwargs)
        else:
            return self.loss

    def prob(self, pred):
        if self.use_sigmoid:
            return 1.0 / (1.0 + np.exp(-pred))
        else:
            return pred