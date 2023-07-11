# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import logging
import os
import json
import jsonpickle
import pickle
import numpy as np

from sklearn.ensemble import RandomForestRegressor

import utils.models
from utils.abstract import AbstractDetector
from utils.models import load_model, load_models_dirpath, load_ground_truth

import torch
import torchvision
import skimage.io

import tensorflow as tf

from tensorflow.keras import datasets, layers, models

import tensorflow.keras as keras
import torch.nn.functional as F
from torchvision.models.detection import ssd

status = False

def center_to_corners_format(x):
    """
    Converts a PyTorch tensor of bounding boxes of center format (center_x, center_y, width, height) to corners format
    (x_0, y_0, x_1, y_1).
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - (0.5 * w)), (y_c - (0.5 * h)), (x_c + (0.5 * w)), (y_c + (0.5 * h))]
    return torch.stack(b, dim=-1)


class Detector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))

        self.metaparameter_filepath = metaparameter_filepath
        self.learned_parameters_dirpath = learned_parameters_dirpath
        self.model_filepath = os.path.join(self.learned_parameters_dirpath, "model.bin")
        self.models_padding_dict_filepath = os.path.join(self.learned_parameters_dirpath, "models_padding_dict.bin")
        self.model_layer_map_filepath = os.path.join(self.learned_parameters_dirpath, "model_layer_map.bin")
        self.layer_transform_filepath = os.path.join(self.learned_parameters_dirpath, "layer_transform.bin")

        self.input_features = metaparameters["train_input_features"]
        self.weight_params = {
            "rso_seed": metaparameters["train_weight_rso_seed"],
            "mean": metaparameters["train_weight_params_mean"],
            "std": metaparameters["train_weight_params_std"],
        }
        self.random_forest_kwargs = {
            "n_estimators": metaparameters[
                "train_random_forest_regressor_param_n_estimators"
            ],
            "criterion": metaparameters[
                "train_random_forest_regressor_param_criterion"
            ],
            "max_depth": metaparameters[
                "train_random_forest_regressor_param_max_depth"
            ],
            "min_samples_split": metaparameters[
                "train_random_forest_regressor_param_min_samples_split"
            ],
            "min_samples_leaf": metaparameters[
                "train_random_forest_regressor_param_min_samples_leaf"
            ],
            "min_weight_fraction_leaf": metaparameters[
                "train_random_forest_regressor_param_min_weight_fraction_leaf"
            ],
            "max_features": metaparameters[
                "train_random_forest_regressor_param_max_features"
            ],
            "min_impurity_decrease": metaparameters[
                "train_random_forest_regressor_param_min_impurity_decrease"
            ],
        }

    def write_metaparameters(self):
        metaparameters = {
            "train_input_features": self.input_features,
            "train_weight_rso_seed": self.weight_params["rso_seed"],
            "train_weight_params_mean": self.weight_params["mean"],
            "train_weight_params_std": self.weight_params["std"],
            "train_random_forest_regressor_param_n_estimators": self.random_forest_kwargs["n_estimators"],
            "train_random_forest_regressor_param_criterion": self.random_forest_kwargs["criterion"],
            "train_random_forest_regressor_param_max_depth": self.random_forest_kwargs["max_depth"],
            "train_random_forest_regressor_param_min_samples_split": self.random_forest_kwargs["min_samples_split"],
            "train_random_forest_regressor_param_min_samples_leaf": self.random_forest_kwargs["min_samples_leaf"],
            "train_random_forest_regressor_param_min_weight_fraction_leaf": self.random_forest_kwargs["min_weight_fraction_leaf"],
            "train_random_forest_regressor_param_max_features": self.random_forest_kwargs["max_features"],
            "train_random_forest_regressor_param_min_impurity_decrease": self.random_forest_kwargs["min_impurity_decrease"],
        }

        with open(os.path.join(self.learned_parameters_dirpath, os.path.basename(self.metaparameter_filepath)), "w") as fp:
            fp.write(jsonpickle.encode(metaparameters, warn=True, indent=2))

    def automatic_configure(self, models_dirpath: str):
        """Configuration of the detector iterating on some of the parameters from the
        metaparameter file, performing a grid search type approach to optimize these
        parameters.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        for random_seed in np.random.randint(1000, 9999, 10):
            self.weight_params["rso_seed"] = random_seed
            self.manual_configure(models_dirpath)

    def inference_on_example_data(self, model, examples_dirpath, **kwargs):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model: the pytorch model
            examples_dirpath: the directory path for the round example data
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info("Using compute device: {}".format(device))
 

        # move the model to the GPU in eval mode
        model.to(device)
        model.eval()

        # Augmentation transformations
        augmentation_transforms = torchvision.transforms.Compose([torchvision.transforms.ConvertImageDtype(torch.float)])

        logging.info("Evaluating the model on the clean example images.")
        # Inference on models
        attrs = []
        for examples_dir_entry in os.scandir(examples_dirpath):
            if examples_dir_entry.is_file() and examples_dir_entry.name.endswith(".png"):
                # load the example image
                img = skimage.io.imread(examples_dir_entry.path)

                # convert the image to a tensor
                # should be uint8 type, the conversion to float is handled later
                image = torch.as_tensor(img)

                # move channels first
                image = image.permute((2, 0, 1))

                # convert to float (which normalizes the values)
                image = augmentation_transforms(image)
                image = image.to(device)

                # Convert to NCHW
                image = image.unsqueeze(0)
                image.requires_grad_()
                model.zero_grad()
                
                # inference
                outputs = model(image)
                
                # handle multiple output formats for different model types
                if 'DetrObjectDetectionOutput' in outputs.__class__.__name__:
                    # DETR doesn't need to unpack the batch dimension
                    boxes = outputs.pred_boxes
                    # boxes from DETR emerge in center format (center_x, center_y, width, height) in the range [0,1] relative to the input image size
                    # convert to [x0, y0, x1, y1] format
                    boxes = center_to_corners_format(boxes)
                    # clamp to [0, 1]
                    boxes = torch.clamp(boxes, min=0, max=1)
                    # and from relative [0, 1] to absolute [0, height] coordinates
                    img_h = img.shape[0] * torch.ones(1)  # 1 because we only have 1 image in the batch
                    img_w = img.shape[1] * torch.ones(1)
                    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).cuda()
                    boxes = boxes * scale_fct[:, None, :]

                    # unpack the logits to get scores and labels
                    logits = outputs.logits
                    prob = torch.nn.functional.softmax(logits, -1)
                    scores, labels = prob[..., :-1].max(-1)

                    boxes = boxes
                    scores = scores
                    labels = labels

                    # all 3 items have a batch size of 1 in the front, so unpack it
                    boxes = boxes[0,]
                    scores = scores[0,]
                    labels = labels[0,]
                else:
                    # unpack the batch dimension
                    outputs = outputs[0]  # unpack the batch size of 1
                    # for SSD and FasterRCNN outputs are a list of dict.
                    # each boxes is in corners format (x_0, y_0, x_1, y_1) with coordinates sized according to the input image

                    boxes = outputs['boxes']
                    scores = outputs['scores']
                    labels = outputs['labels']

                # wrap the network outputs into a list of annotations
                pred = utils.models.wrap_network_prediction(boxes, labels)

                # logging.info('example img filepath = {}, Pred: {}'.format(examples_dir_entry.name, pred))

                ground_truth_filepath = examples_dir_entry.path.replace('.png','.json')

                f = open(ground_truth_filepath, mode='r', encoding='utf-8')
                ground_truth = jsonpickle.decode(f.read())
                f.close()

                ground_truth = [item for item in ground_truth if 'label' in item] 

                #logging.info("Model predicted {} boxes, Ground Truth has {} boxes.".format(len(pred), len(ground_truth)))
                #logging.info("Model: {}, Ground Truth: {}".format(examples_dir_entry.name, ground_truth))
                
                #print(model.modules)

                lenb = min([len(pred), len(ground_truth)])
                pred_classes = [item['label'] for item in list(pred)[0:lenb]]
                target_classes = ([item['label'] for item in ground_truth[0:lenb]])
                classification_loss = F.cross_entropy(torch.tensor(pred_classes).float(), 
                torch.tensor(target_classes).float(), reduction="sum")

                pred_bbs = torch.vstack([item['bbox'] for item in list(pred)[0:lenb]])
                target_bbs = torch.tensor([list(item['bbox'].values()) for item in ground_truth[0:lenb]],device='cuda:0')
                box_loss = F.smooth_l1_loss(
                target_bbs, pred_bbs, beta=1 / 9, reduction="sum",)
                
                loss = classification_loss + box_loss
                loss.requires_grad_()
                loss.backward()

                print(list(enumerate(list(enumerate(model.modules()))[0][1].modules()))[63][1])

                attr = list(enumerate(model.modules()))[0][1].weight.grad
                input("waiting")
                attr = attr.squeeze(0).permute((1, 2, 0)).cpu().numpy()
                attrs.append(attr)
        return attrs


    def manual_configure(self, models_dirpath: str):
        """Configuration of the detector using the parameters from the metaparameters
        JSON file.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        # Create the learned parameter folder if needed
        if not os.path.exists(self.learned_parameters_dirpath):
            os.makedirs(self.learned_parameters_dirpath)

        # List all available model
        model_path_list = sorted([os.path.join(models_dirpath, model) for model in os.listdir(models_dirpath)])
        logging.info(f"Loading %d models...", len(model_path_list))
        X = []
        y = []
        for model_filepath in model_path_list:
            model, model_repr, model_class = load_model(os.path.join(model_filepath, 'model.pt'))
            model_label = load_ground_truth(model_filepath)
            examples_dirpath = os.path.join(model_filepath, 'clean-example-data')
            # Inferences on examples to demonstrate how it is done for a round
            xs = self.inference_on_example_data(model, examples_dirpath)
            xs = np.clip((np.asarray(xs) - np.mean(xs, axis=0)) / np.std(xs, axis=0), -1, 1) * 0.5 + 1
            X = X + [x for x in xs]
            y = y + [int(model_label)] * xs.shape[0]
        
        """

        model_repr_dict, model_ground_truth_dict = load_models_dirpath(model_path_list)

        logging.info("Building RandomForest based on random features, with the provided mean and std.")
        rso = np.random.RandomState(seed=self.weight_params['rso_seed'])
        X = []
        y = []
        for model_arch in model_repr_dict.keys():
            for model_index in range(len(model_repr_dict[model_arch])):
                y.append(model_ground_truth_dict[model_arch][model_index])

                model_feats = rso.normal(loc=self.weight_params['mean'], scale=self.weight_params['std'], size=(1,self.input_features))
                X.append(model_feats)
        X = np.vstack(X)
        
        logging.info("Training RandomForestRegressor model.")
        model = RandomForestRegressor(**self.random_forest_kwargs, random_state=0)
        model.fit(X, y)

        logging.info("Saving RandomForestRegressor model...")
        with open(self.model_filepath, "wb") as fp:
            pickle.dump(model, fp)
        """
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)
        
        model = tf.keras.models.Sequential([
            # Note the input shape is the desired size of the image 200x200 with 3 bytes color
            # This is the first convolution
            tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(256, 256, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            # The second convolution
            tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            # The third convolution
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            # The fourth convolution
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            # # The fifth convolution
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            # Flatten the results to feed into a DNN
            tf.keras.layers.Flatten(),
            # 512 neuron hidden layer
            tf.keras.layers.Dense(512, activation='relu'),
            # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('dandelions') and 1 for the other ('grass')
            tf.keras.layers.Dense(1, activation='sigmoid')])

        model.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics='accuracy')

        history = model.fit(np.asarray(X_train), np.asarray(y_train), batch_size=25, epochs=10, verbose=1)
        _, test_acc = model.evaluate(np.asarray(X_test), np.asarray(y_test), verbose=0)
        logging.info(f"ACC: {test_acc}")
        from sklearn.metrics import roc_curve
        y_pred_keras = model.predict(np.asarray(X_test)).ravel()
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(np.asarray(y_test), np.asarray(y_pred_keras))
        from sklearn.metrics import auc
        auc_keras = auc(fpr_keras, tpr_keras)
        logging.info(f"AUC: {auc_keras}")
        self.write_metaparameters()
        logging.info("Configuration done!")
        model.save(os.path.join(self.learned_parameters_dirpath,"model.h5"))

    def infer(
        self,
        model_filepath,
        result_filepath,
        scratch_dirpath,
        examples_dirpath,
        round_training_dataset_dirpath):
        """Method to predict whether a model is poisoned (1) or clean (0).

        Args:
            model_filepath:
            result_filepath:
            scratch_dirpath:
            examples_dirpath:
            round_training_dataset_dirpath:
        """

        # load the model
        model, model_repr, model_class = load_model(model_filepath)

        # Inferences on examples to demonstrate how it is done for a round
        X = self.inference_on_example_data(model, examples_dirpath)      
        X = np.clip((np.asarray(X) - np.mean(X, axis=0)) / np.std(X, axis=0), -1, 1) * 0.5 + 1

        # write the trojan probability to the output file
        if status == True:
            probability = 1
        else:
            model = keras.models.load_model(os.path.join(self.learned_parameters_dirpath,"model.h5")) 
            probability = model.predict(X).mean()
        with open(result_filepath, "w") as fp:
            fp.write(str(probability))

        logging.info("Trojan probability: {}".format(probability))
        print("Trojan probability: {}".format(probability))
