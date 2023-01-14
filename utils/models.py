import re
from collections import OrderedDict
from os.path import join

import torch
from tqdm import tqdm


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
    examples = np.arrary([[]])
    for examples_dir_entry in os.scandir(join(model_dirpath, "clean-example-data" if clean else "poisoned-example-data")):
            if examples_dir_entry.is_file() and examples_dir_entry.name.endswith(".npy"):
                feature_vector = np.load(examples_dir_entry.path).reshape(1, -1)
                examples = np.hstack((examples, feature_vector), axis = 0)

    return examples

def load_models_dirpath(models_dirpath):
    model_repr_dict = {}
    model_ground_truth_dict = {}
    clean_example_dict = {}
    poisoned_example_dict = {}

    for model_path in tqdm(models_dirpath):
        model, model_repr, model_class = load_model(
            join(model_path, "model.pt")
        )
        model_ground_truth = load_ground_truth(model_path)
        clean_examples = load_examples(model_path)
        poisoned_examples = load_examples(model_path, False)

        # Build the list of models
        if model_class not in model_repr_dict.keys():
            model_repr_dict[model_class] = []
            model_ground_truth_dict[model_class] = []
            clean_example_dict[model_class] = []
            poisoned_example_dict[model_class] = []

        model_repr_dict[model_class].append(model_repr)
        model_ground_truth_dict[model_class].append(model_ground_truth)
        clean_example_dict[model_class].append(clean_examples)
        poisoned_examples[model_class].append(poisoned_examples)

    return model_repr_dict, model_ground_truth_dict, clean_example_dict, poisoned_example_dict


def inference_on_example_data(self, model, examples_dirpath, grad = np.mean):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model: the pytorch model
            examples_dirpath: the directory path for the round example data
        """
        print("Inference on example data")
        # Setup scaler
        scaler = StandardScaler()

        scale_params = np.load(self.scale_parameters_filepath)

        scaler.mean_ = scale_params[0]
        scaler.scale_ = scale_params[1]

        
        grad_reprs = []
        # Inference on models
        for examples_dir_entry in os.scandir(examples_dirpath):
            if examples_dir_entry.is_file() and examples_dir_entry.name.endswith(".npy"):
                feature_vector = np.load(examples_dir_entry.path).reshape(1, -1)
                print(">>>>>>> Example feature shape: ", feature_vector.shape)
                feature_vector = torch.from_numpy(scaler.transform(feature_vector.astype(float))).float()
                model.zero_grad()
                #pred = torch.argmax(model(feature_vector).detach()).item()
                scores = model(feature_vector)
                pred = torch.argmax(scores).detach()
                logits = F.log_softmax(scores, dim = 1)
                ground_tuth_filepath = examples_dir_entry.path + ".json"
                with open(ground_tuth_filepath, 'r') as ground_truth_file:
                    ground_truth =  ground_truth_file.readline()
                print("Model: {}, Ground Truth: {}, Prediction: {}".format(examples_dir_entry.name, ground_truth, str(pred)))
            
                if grad is not None:
                    loss = F.cross_entropy(logits, torch.LongTensor([int(ground_truth)]))
                    loss.backward();
                    grad_repr = OrderedDict(
                        {layer: param.data.numpy() for ((layer, _), param) in zip(model.state_dict().items(), model.parameters())}
                    ) 
                grad_reprs.append(grad_repr)
        grad_repr_dict = {}
        if grad is not None:
            for (layer, _) in model.items():
                grad_repr_dict[layer] = np.mean([grad_repr[layer] for grad_repr in grad_reprs], axis = -1)
        return grad_repr_dict