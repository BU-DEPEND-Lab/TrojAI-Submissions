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


from depend.core.dependent import MaskGen
from depend.launch import Sponsor


from sklearn.ensemble import RandomForestRegressor

from utils.abstract import AbstractDetector
from utils.models import load_model, load_models_dirpath

import torch
import torch_ac
import gym
from gym_minigrid.wrappers import ImgObsWrapper

 

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
        elif self.method == 'mask_gen':
            self.manual_configure_mask_gen(model_path_list)

    
    def manual_configure_mask_gen(self, model_path_list: List[str]):
        import gym_minigrid
        dependent = MaskGen.get_assets(model_path_list)
        config = {
            'model_schema': {
                'mask': {
                    'name': 'Basic_FC_VAE'
                },
                'save_dir': 'best_mask.p'
            },
            'learner_schema': {
                'episodes': 200,
                'batch_size': 32,
                'checkpoint_interval': 1,
                'eval_interval': 2,
            },
            'algorithm_schema': {
                'device': 'cuda:1',
                'task': 'RL',
                'criterion': 'kl',
                'beta': 1,
                'k_fold': False,
                'num_procs': 10,
                'exploration_rate': 0.5,
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
        Sponsor(**config).support(dependent, 'test', 'result')

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

    def inference_on_example_data(self, model, examples_dirpath):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model: the pytorch model
            examples_dirpath: the directory path for the round example data
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info("Using compute device: {}".format(device))

        model.to(device)
        model.eval()

        preprocess = torch_ac.format.default_preprocess_obss

        # Utilize open source minigrid environment model was trained on
        env_string_filepath = os.path.join(examples_dirpath, 'env-string.txt')
        with open(env_string_filepath) as env_string_file:
            env_string = env_string_file.readline().strip()
        logging.info('Evaluating on {}'.format(env_string))

        # Number of episodes to run
        episodes = 100

        env_perf = {}

        # Run episodes through an environment to collect what may be relevant information to trojan detection
        # Construct environment and put it inside a observation wrapper
        env = ImgObsWrapper(gym.make(env_string))
        obs = env.reset()
        obs = preprocess([obs], device=device)

        final_rewards = []
        with torch.no_grad():
            # Episode loop
            for _ in range(episodes):
                done = False
                # Use env observation to get action distribution
                dist, value = model(obs)
                # Per episode loop
                while not done:
                    # Sample from distribution to determine which action to take
                    action = dist.sample()
                    action = action.cpu().detach().numpy()
                    # Use action to step environment and get new observation
                    obs, reward, done, info = env.step(action)
                    # Preprocessing function to prepare observation from env to be given to the model
                    obs = preprocess([obs], device=device)
                    # Use env observation to get action distribution
                    dist, value = model(obs)

                # Collect episode performance data (just the last reward of the episode)
                final_rewards.append(reward)
                # Reset environment after episode and get initial observation
                obs = env.reset()
                obs = preprocess([obs], device=device)

        # Save final rewards
        env_perf['final_rewards'] = final_rewards

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
        model, model_repr, model_class = load_model(model_filepath)

        # Inferences on examples to demonstrate how it is done for a round
        self.inference_on_example_data(model, examples_dirpath)

        # build a fake random feature vector for this model, in order to compute its probability of poisoning
        rso = np.random.RandomState(seed=self.weight_params['rso_seed'])
        X = rso.normal(loc=self.weight_params['mean'], scale=self.weight_params['std'], size=(1, self.input_features))

        # load the RandomForest from the learned-params location
        with open(self.model_filepath, "rb") as fp:
            regressor: RandomForestRegressor = pickle.load(fp)

        # use the RandomForest to predict the trojan probability based on the feature vector X
        probability = regressor.predict(X)[0]
        # clip the probability to reasonable values
        probability = np.clip(probability, a_min=0.01, a_max=0.99)

        # write the trojan probability to the output file
        with open(result_filepath, "w") as fp:
            fp.write(str(probability))

        logging.info("Trojan probability: {}".format(probability))
 