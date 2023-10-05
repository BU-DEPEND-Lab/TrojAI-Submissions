# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import logging
import os
import json
import jsonpickle
import pickle
import numpy as np
from typing import List


from depend.core.dependent import MaskGen, AttributionClassifier, ConfidenceContrast
from depend.launch import Sponsor

import torch
from sklearn.ensemble import RandomForestRegressor

from utils.abstract import AbstractDetector
from utils.model_utils import compute_action_from_trojai_rl_model
from utils.models import load_model, load_models_dirpath, ImageACModel, ResNetACModel


from utils.world import RandomLavaWorldEnv
from utils.wrappers import ObsEnvWrapper, TensorWrapper


from datetime import datetime
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


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

        self.method = metaparameters['method']
        self.input_features = metaparameters["train_input_features"]
        if self.method == 'random_forest':
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
        if self.method == 'random_forest':
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
        elif self.method == 'mask_gen':
            metaparameters = {}
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
        if self.method == 'random_forest':
            self.manual_configure_random_forest(model_path_list)
        elif self.method == 'attr_cls':
            self.manual_configure_attr_cls(model_path_list)
        elif self.method == 'conf_cont':
            self.manual_configure_conf_cont(model_path_list)

    def manual_configure_conf_cont(self, model_path_list: List[str]):
        dependent = ConfidenceContrast.get_assets(model_path_list)
        config = {
            'model_schema': {
                'classifier': {
                    'name': 'FCModel', 
                },
                'save_dir': 'best_conf_cls.p'
            },
            'learner_schema': {
                'episodes': 30,
                'batch_size': 32,
                'checkpoint_interval': 1,
                'eval_interval': 2,
            },
            'algorithm_schema': {
                'device': 'cuda:3',
                'task': 'RL',
                'criterion': 'ce',
                'beta': 0,
                'k_fold': True,
                'num_procs': 20,
                'exploration_method': 'reverse::0.5',
                'num_experiments': 1,
                #'load_experience': '/home/zwc662/Workspace/TrojAI-Submissions/experience.p'
                 
            },
            'optimizer_schema': {
                'optimizer_class': 'Adam',
                'lr': 1e-3,
            },
            'data_schema': {
                'num_splits': 7,
                'max_models': 238,
                'num_frames_per_model': 128
            }
            
        }

        result_dir = os.path.join('./logs', timestamp)
        os.mkdir(result_dir)
        Sponsor(**config).support(dependent, 'test', result_dir)
        for i in range(len(dependent.envs)):
            dependent.envs[i] = TensorWrapper(ObsEnvWrapper(RandomLavaWorldEnv(mode='simple', grid_size=9), mode='simple'))
        dependent.train_detector() 


    def manual_configure_attr_cls(self, model_path_list: List[str]):
        dependent = AttributionClassifier.get_assets(model_path_list)
        config = {
            'model_schema': {
                'classifier': {
                    'name': 'FCModel', 
                },
                'save_dir': 'best_attr_cls.p'
            },
            'learner_schema': {
                'episodes': 30,
                'batch_size': 32,
                'checkpoint_interval': 1,
                'eval_interval': 2,
            },
            'algorithm_schema': {
                'device': 'cuda:3',
                'task': 'RL',
                'criterion': 'ce',
                'beta': 0,
                'k_fold': True,
                'num_procs': 40,
                'exploration_method': 0.5,
                'num_experiments': 1,
                #'load_experience': '/home/zwc662/Workspace/TrojAI-Submissions/experience.p'
                 
            },
            'optimizer_schema': {
                'optimizer_class': 'Adam',
                'lr': 1e-3,
            },
            'data_schema': {
                'num_splits': 7,
                'max_models': 238,
                'num_frames_per_model': 256
            }
            
        }

        result_dir = os.path.join('./logs', timestamp)
        os.mkdir(result_dir)
        Sponsor(**config).support(dependent, 'test', result_dir)
        for i in range(len(dependent.envs)):
            dependent.envs[i] = TensorWrapper(ObsEnvWrapper(RandomLavaWorldEnv(mode='simple', grid_size=9), mode='simple'))
        dependent.train_detector() 

    def manual_configure_random_forest(self, model_path_list: List[str]):
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

        self.write_metaparameters()
        logging.info("Configuration done!")

    def inference_on_example_data(self, model, examples_dirpath, config_dict):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model: the pytorch model
            examples_dirpath: the directory path for the round example data
        """

        size = config_dict["grid_size"]

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model.to(device)
        model.eval()

        # logging.info("Using compute device: {}".format(device))

        model_name = type(model).__name__
        observation_mode = "rgb" if model_name in [ImageACModel.__name__, ResNetACModel.__name__] else 'simple'

        wrapper_obs_mode = 'simple_rgb' if observation_mode == 'rgb' else 'simple'

        env = TensorWrapper(ObsEnvWrapper(RandomLavaWorldEnv(mode=observation_mode, grid_size=size), mode=wrapper_obs_mode))

        obs, info = env.reset()
        done = False
        max_iters = 1000
        iters = 0
        reward = 0

        while not done and iters < max_iters:
            env.render()
            action = compute_action_from_trojai_rl_model(model, obs, sample=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        logging.info('Final reward: {}'.format(reward))


    def infer(
            self,
            model_filepath,
            result_filepath,
            scratch_dirpath,
            examples_dirpath,
            round_training_dataset_dirpath,
    ):
        """Method to predict whether a model is poisoned (1) or clean (0).

        Args:
            model_filepath:
            result_filepath:
            scratch_dirpath:
            examples_dirpath:
            round_training_dataset_dirpath:
        """

        # load the model

        probability = None
        if self.method == 'random_forest':
            probability = self.inference_random_forest(model_filepath, examples_dirpath)
        elif self.method == 'attr_cls':
            probability = self.inference_with_attr_cls(model_filepath)
        elif self.method == 'conf_cont':
            probability = self.inference_with_conf_cont(model_filepath)
        else:
            probability = self.inference_with_attr_cls(model_filepath)
 
        # write the trojan probability to the output file
        with open(result_filepath, "w") as fp:
            fp.write(str(probability))
            logging.info(f"Wrote to result file {result_filepath}")
        with open(result_filepath, "r") as fp:
            logging.info(f"Check result file {fp.read()}")

        logging.info("Trojan probability: {}".format(probability))
 

        # write the trojan probability to the output file
        with open(result_filepath, "w") as fp:
            fp.write(str(probability))

        logging.info("Trojan probability: {}".format(probability))



    def inference_with_random_forest(self, model_filepath, examples_dirpath):
        model, model_repr, model_class = load_model(model_filepath)

        # Load the config file
        config_dict = {}

        model_dirpath = os.path.dirname(model_filepath)

        config_filepath = os.path.join(model_dirpath, 'config.json')

        with open(config_filepath) as config_file:
            config_dict = json.load(config_file)

        # Inferences on examples to demonstrate how it is done for a round
        self.inference_on_example_data(model, examples_dirpath, config_dict)

        # build a fake random feature vector for this model, in order to compute its probability of poisoning
        rso = np.random.RandomState(seed=self.weight_params['rso_seed'])
        X = rso.normal(loc=self.weight_params['mean'], scale=self.weight_params['std'], size=(1, self.input_features))

        # # create a random model for testing (fit to nothing)
        # model = RandomForestRegressor(**self.random_forest_kwargs, random_state=0)
        # model.fit(X, [0])
        # with open(self.model_filepath, "wb") as fp:
        #     pickle.dump(model, fp)

        # load the RandomForest from the learned-params location
        with open(self.model_filepath, "rb") as fp:
            regressor: RandomForestRegressor = pickle.load(fp)

        # use the RandomForest to predict the trojan probability based on the feature vector X
        probability = regressor.predict(X)[0]
        # clip the probability to reasonable values
        probability = np.clip(probability, a_min=0.01, a_max=0.99)
        return probability
        
    def inference_with_attr_cls(self, model_filepath):
        model, model_repr, model_class = load_model(model_filepath)
        model.eval()
        #model.state_emb[1].weight = model.state_emb[1].weight.detach() * np.random.random(model.state_emb[1].weight.shape) 
        dependent = AttributionClassifier()
        config = {
            'model_schema': {
                'classifier': {
                    'name': 'FCModel', 
                    'load_from_file': os.path.join(os.path.dirname(__file__), 'best_attr_cls.p')
                },
                'save_dir': 'best_attr_cls.p'
            },
            'learner_schema': {
                'episodes': 100,
                'batch_size': 32,
                'checkpoint_interval': 1,
                'eval_interval': 2,
            },
            'algorithm_schema': {
                'device': 'cuda:1',
                'task': 'RL',
                'criterion': 'ce',
                'beta': 1,
                'k_fold': True,
                'num_procs': 20,
                'exploration_rate': 0.5,
                'num_experiments': 1,
                'load_experience': os.path.join(os.path.dirname(__file__), 'best_attr_experience.p')
                 
            },
            'optimizer_schema': {
                'optimizer_class': 'Adam',
                'lr': 1e-3,
            },
            'data_schema': {
                'num_splits': 7,
                'max_models': 238,
                'num_frames_per_model': 128
            }
        }
        Sponsor(**config).support(dependent, None, None)
        return dependent.infer(model)


    def inference_with_conf_cont(self, model_filepath):
        model, model_repr, model_class = load_model(model_filepath)
        model.eval()
        #model.state_emb[1].weight = model.state_emb[1].weight.detach() * np.random.random(model.state_emb[1].weight.shape) 
        dependent = ConfidenceContrast()
        config = {
            'model_schema': {
                'classifier': {
                    'name': 'FCModel', 
                    'load_from_file': os.path.join(os.path.dirname(__file__), 'best_cls.p')
                },
                'save_dir': 'best_conf_cls.p'
            },
            'learner_schema': {
                'episodes': 100,
                'batch_size': 32,
                'checkpoint_interval': 1,
                'eval_interval': 2,
            },
            'algorithm_schema': {
                'device': 'cuda:1',
                'task': 'RL',
                'criterion': 'ce',
                'beta': 1,
                'k_fold': True,
                'num_procs': 20,
                'exploration_rate': 0.5,
                'num_experiments': 1,
                'load_experience': os.path.join(os.path.dirname(__file__), 'best_conf_experience.p')
                 
            },
            'optimizer_schema': {
                'optimizer_class': 'Adam',
                'lr': 1e-3,
            },
            'data_schema': {
                'num_splits': 7,
                'max_models': 238,
                'num_frames_per_model': 128
            }
        }
        Sponsor(**config).support(dependent, None, None)
        return dependent.infer(model)


    