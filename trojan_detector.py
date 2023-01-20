import json
import logging
import os
import pickle
from os import listdir, makedirs
from os.path import join, exists, basename
from collections import OrderedDict
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost
from xgboost import cv, XGBRegressor, XGBClassifier
from tqdm import tqdm

from utils.abstract import AbstractDetector
from utils.flatten import flatten_model, flatten_models, flatten_grads
from utils.healthchecks import check_models_consistency
from utils.models import create_layer_map, load_model, \
    load_models_dirpath, load_ground_truth
from utils.padding import create_models_padding, pad_model
from utils.reduction import (
    fit_feature_reduction_algorithm,
    use_feature_reduction_algorithm,
    grad_feature_reduction_algorithm
)
#from attrdict import AttrDict
from sklearn.preprocessing import StandardScaler
from archs import Net2, Net3, Net4, Net5, Net6, Net7, Net2r, Net3r, Net4r, Net5r, Net6r, Net7r, Net2s, Net3s, Net4s, Net5s, Net6s, Net7s
import torch
import torch.nn.functional as F


from sklearn.linear_model import SGDClassifier 
import sklearn.model_selection
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from joblib import dump, load
from sklearn import metrics
from sklearn import svm

from feature_extractor import FeatureExtractor


class Detector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath, scale_parameters_filepath):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
            scale_parameters_filepath: str - File path to the scale_parameters file.
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))

        self.scale_parameters_filepath = scale_parameters_filepath
        self.metaparameter_filepath = metaparameter_filepath
        self.learned_parameters_dirpath = learned_parameters_dirpath
        self.model_filepath = join(self.learned_parameters_dirpath, "model.json")
        self.models_padding_dict_filepath = join(self.learned_parameters_dirpath, "models_padding_dict.bin")
        self.model_layer_map_filepath = join(self.learned_parameters_dirpath, "model_layer_map.bin")
        self.layer_transform_filepath = join(self.learned_parameters_dirpath, "layer_transform.bin")

        # TODO: Update skew parameters per round

        self.model_skew = {
            "__all__": metaparameters["infer_cyber_model_skew"],
        }
        self.train_data_augmentation = metaparameters["train_data_augmentation"]
        self.input_features = metaparameters["train_input_features"]
        self.ICA_features = metaparameters["train_ICA_features"]
        self.weight_table_params = {
            "random_seed": metaparameters["train_weight_table_random_state"],
            "mean": metaparameters["train_weight_table_params_mean"],
            "std": metaparameters["train_weight_table_params_std"],
            "scaler": metaparameters["train_weight_table_params_scaler"],
        }
        self.random_forest_kwargs = {
            "n_estimators": metaparameters[
                "train_random_forest_param_n_estimators"
            ],
            "criterion": metaparameters[
                "train_random_forest_param_criterion"
            ],
            "max_depth": metaparameters[
                "train_random_forest_param_max_depth"
            ],
            "min_samples_split": metaparameters[
                "train_random_forest_param_min_samples_split"
            ],
            "min_samples_leaf": metaparameters[
                "train_random_forest_param_min_samples_leaf"
            ],
            "min_weight_fraction_leaf": metaparameters[
                "train_random_forest_param_min_weight_fraction_leaf"
            ],
            "max_features": metaparameters[
                "train_random_forest_param_max_features"
            ],
            "min_impurity_decrease": metaparameters[
                "train_random_forest_param_min_impurity_decrease"
            ],
        }
     
        self.xgboost_kwargs = {
            "booster": metaparameters[
                "train_xgboost_param_booster"
            ],
            "objective": metaparameters[
                "train_xgboost_param_objective"
            ],
            "max_depth": metaparameters[
                "train_xgboost_param_max_depth"
            ],
            "max_leaves": metaparameters[
                "train_xgboost_param_max_leaves"
            ],
            "max_bin": metaparameters[
                "train_xgboost_param_max_bin"
            ],
            "min_child_weight": metaparameters[
                "train_xgboost_param_min_child_weight"
            ],
            "eval_metric": metaparameters[
                "train_xgboost_param_eval_metric"
            ],
            "max_delta_step": metaparameters[
                "train_xgboost_param_max_delta_step"
            ],
        }

    def write_metaparameters(self, *metaparameters):
        metaparameters_base = {
            "infer_cyber_model_skew": self.model_skew["__all__"],
            "train_input_features": self.input_features,
            "train_ICA_features": self.ICA_features,
            "train_weight_table_random_state": self.weight_table_params["random_seed"],
            "train_weight_table_params_mean": self.weight_table_params["mean"],
            "train_weight_table_params_std": self.weight_table_params["std"],
            "train_weight_table_params_scaler": self.weight_table_params["scaler"],
            "train_random_forest_param_n_estimators": self.random_forest_kwargs["n_estimators"],
            "train_random_forest_param_criterion": self.random_forest_kwargs["criterion"],
            "train_random_forest_param_max_depth": self.random_forest_kwargs["max_depth"],
            "train_random_forest_param_min_samples_split": self.random_forest_kwargs["min_samples_split"],
            "train_random_forest_param_min_samples_leaf": self.random_forest_kwargs["min_samples_leaf"],
            "train_random_forest_param_min_weight_fraction_leaf": self.random_forest_kwargs["min_weight_fraction_leaf"],
            "train_random_forest_param_max_features": self.random_forest_kwargs["max_features"],
            "train_random_forest_param_min_impurity_decrease": self.random_forest_kwargs["min_impurity_decrease"],
            
            "train_xgboost_param_booster": self.xgboost_kwargs["booster"],
            "train_xgboost_param_objective": self.xgboost_kwargs["objective"],
            "train_xgboost_param_max_depth": self.xgboost_kwargs["max_depth"],
            "train_xgboost_param_max_leaves": self.xgboost_kwargs["max_leaves"],
            "train_xgboost_param_max_bin": self.xgboost_kwargs["max_bin"],
            "train_xgboost_param_min_child_weight": self.xgboost_kwargs["min_child_weight"],
            "train_xgboost_param_eval_metric": self.xgboost_kwargs["eval_metric"],
            "train_xgboost_param_max_delta_step": self.xgboost_kwargs["max_delta_step"],
        }
        if len(metaparameters) > 0:
            for metaparameter in metaparameters:
                metaparameters_base.update(metaparameter)

        with open(join(self.learned_parameters_dirpath, basename(self.metaparameter_filepath)), "w") as fp:
            json.dump(metaparameters_base, fp)

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
        if not exists(self.learned_parameters_dirpath):
            makedirs(self.learned_parameters_dirpath)

        # List all available model
        model_path_list = sorted([join(models_dirpath, model) for model in listdir(models_dirpath)])
        logging.info(f"Loading %d models...", len(model_path_list))

        feature_extractor = FeatureExtractor(self.metaparameter_filepath, self.learned_parameters_dirpath,  self.scale_parameters_filepath)
        
        X = None
        Y = None
        for i in range(self.train_data_augmentation):
            if X is None:
                X = np.asarray(feature_extractor.infer_layer_features_from_models(model_path_list, True))
            else:
                X = np.vstack((X, np.asarray(feature_extractor.infer_layer_features_from_models(model_path_list, True))))
            for model_path in model_path_list:
                y = load_ground_truth(model_path)
                if Y is None:
                    Y = y 
                    continue
                else:
                    Y = np.vstack((Y, y))
        """
        for model_path in model_path_list:
            x = np.asarray(feature_extractor.infer_norms_from_one_model(model_path))
            y = 2 * load_ground_truth(model_path) - 1
            #y = load_ground_truth(model_path)
            if X is None:
                X = x
                Y = y 
                continue
            elif x.shape[-1] == X.shape[-1]:
                #X = np.vstack((X, x)) * self.model_skew["__all__"]
                Y = np.vstack((Y, y))
            else:
                Y = np.vstack((Y, y))
        """
        print(f"Data set size >>>>>> X: {X.shape}  Y: {Y.shape}")
        
        """
        import matplotlib.pyplot as plt
        n_bins = 10
        for i in range(X.shape[-1]):
            print(f"Generate plot for feature {i}")
            fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
            axs[0].hist(X[:, i][Y.flatten() > 0], bins=n_bins)
            axs[1].hist(X[:, i][Y.flatten() < 0], bins=n_bins)
            plt.savefig(f"./feature_visualization/weight_norm_feature_{i}_histogram")
        """
        
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, random_state=5)
        
        print('x_train', x_train.shape)
        print('x_test', x_test.shape)

        """
        model_name = "svm"
        logging.info("Training probable SVM model...")
        clf = svm.SVC(kernel='linear', probability=True)
        probas_ = clf.fit(x_train, y_train).predict_proba(x_test)
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        roc_auc = auc(fpr, tpr)
        print("Test auc : %f" % roc_auc)
        
        
        model_name = "grid_search_svm"
        logging.info("Grid searching SVM model...")
        svm_kwargs_grid = {'C': [0.1, 1, 10, 100, 1000, 10000], 
              'gamma': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['linear', 'rbf']
              } 
        grid = GridSearchCV(svm.SVC(probability=True), svm_kwargs_grid,  scoring = 'roc_auc', cv = 3, refit = True, verbose = 3)
        grid.fit(x_train, y_train)
        clf = grid.best_estimator_
        print(clf.classes_)
        #
        
        
        """
        
        model_name = "xgboost_regressor"
        data_dmatrix = xgboost.DMatrix(data=x_train,label= y_train)
        params = { 
            'max_depth': [10, 15, 20, 30],
           'learning_rate': [0.01, 0.1, 0.2, 0.3],
           'subsample': np.arange(0.5, 1.0, 0.1),
           'colsample_bytree': np.arange(0.4, 1.0, 0.1),
           'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
           'n_estimators': [100, 500, 1000]}
           
        rand = RandomizedSearchCV(estimator=XGBRegressor(seed = 20),
                         param_distributions=params,
                         scoring='neg_mean_squared_error',
                         n_iter=25, cv = 5, n_jobs = -1, refit = True,
                         verbose=1)
        rand.fit(x_train, y_train)
        clf = rand.best_estimator_ #XGBRegressor(**rand.best_params_)

        """
        #xgb_cv = cv(dtrain=data_dmatrix, params=self.xgboost_kwargs, nfold=5,
                    num_boost_round=50, early_stopping_rounds=10, metrics="auc", as_pandas=True, seed=123)
        print(xgb_cv)
        
        logging.info("Training XGBoostClassifier model...")
        clf = XGBRegressor(**self.xgboost_kwargs)
        
        
        model_name = "randomforest_classifier"
        logging.info("Training RandomForestClassifier model...")
        clf = RandomForestClassifier(**self.random_forest_classifier_kwargs, random_state=0)
        
        
        
        

        y_pred = clf.predict(x_train)
        if 'svm' in model_name:
            print("Training comparison:\n", y_train.reshape(-1), "\n", y_pred)
            print('train acc', accuracy_score(y_train.reshape(-1), np.asarray(y_pred)))
            y_pred_probs = clf.predict_proba(x_train)
            fpr, tpr, thresholds = metrics.roc_curve(y_train, y_pred_probs[:, 1])
        elif "xgboost_regressor" in model_name:
            print("Training comparison:\n", y_train.reshape(-1), "\n", y_pred >= 0.5)
            print('train acc', accuracy_score(y_train.reshape(-1), np.asarray(y_pred >= 0.5)))
            y_pred = clf.predict(x_train)
            fpr, tpr, thresholds = metrics.roc_curve(y_train, y_pred)
        print(f'train fpr {fpr}')
        print(f'tpr {tpr}')
        print('train auc', metrics.auc(fpr, tpr))
        """
        y_pred_ = clf.predict(x_test)
        
        if 'svm' in model_name:
            print("Testing comparison:\n", y_test.reshape(-1), "\n", y_pred_)
            print('test acc', accuracy_score(y_test.reshape(-1), np.asarray(y_pred_)))
            y_pred_probs_ = clf.predict_proba(x_test)
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_probs_[:, 1])
        elif "xgboost_regressor" in model_name:
            print("Testing comparison:\n", y_test.reshape(-1), "\n", y_pred_ >= 0.5)
            print('test acc', accuracy_score(y_test.reshape(-1), np.asarray(y_pred_ >= 0.5)))
            y_pred_ = clf.predict(x_test)
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_)
        print(f'test fpr {fpr}')
        print(f'tpr {tpr}')
        print('test auc', metrics.auc(fpr, tpr))
        logging.info("Saving model...")
        dump(clf, f'round12_{model_name}.joblib') 

        logging.info("Now train on all dataset")
        
        clf.fit(X, Y) 
        

        #logging.info("Training RandomForestRegressor model...")
        #model = RandomForestRegressor(**self.random_forest_kwargs, random_state=0)
        #model.fit(X, y)

        
        #logging.info("Saving RandomForestRegressor model...")
        #logging.info("Saving XGBoostRegressor model...")
        
        #with open(self.model_filepath, "wb") as fp:
        #    pickle.dump(model, fp)
        
        if "xgboost" in model_name:
            clf.save_model(self.model_filepath)
    
        self.write_metaparameters(feature_extractor.write_metaparameters())
        
        logging.info("Configuration done!")

    def inference_on_example_data(self, model, examples_dirpath, grad = False):
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
                #print(">>>>>>> Example feature shape: ", feature_vector.shape)
                feature_vector = torch.from_numpy(scaler.transform(feature_vector.astype(float))).float()
                model.zero_grad()
                #pred = torch.argmax(model(feature_vector).detach()).item()
                scores = model(feature_vector)
                pred = torch.argmax(scores).detach()
                logits = F.log_softmax(scores, dim = 1)
                ground_tuth_filepath = examples_dir_entry.path + ".json"
                with open(ground_tuth_filepath, 'r') as ground_truth_file:
                    ground_truth =  ground_truth_file.readline()
                #print("Model: {}, Ground Truth: {}, Prediction: {}".format(examples_dir_entry.name, ground_truth, str(pred)))
            
                if grad:
                    loss = F.cross_entropy(logits, torch.LongTensor([int(ground_truth)]))
                    loss.backward();
                    grad_repr = OrderedDict(
                        {layer: param.data.numpy() for ((layer, _), param) in zip(model.state_dict().items(), model.parameters())}
                    ) 
                grad_reprs.append(grad_repr)    
        return grad_reprs
                
    def infer(
        self,
        model_filepath,
        result_filepath,
        scratch_dirpath,
        examples_dirpath,
        round_training_dataset_dirpath,
    ):
        """Method to predict wether a model is poisoned (1) or clean (0).

        Args:
            model_filepath:
            result_filepath:
            scratch_dirpath:
            examples_dirpath:
            round_training_dataset_dirpath:
        """
        feature_extractor = FeatureExtractor(self.metaparameter_filepath, self.learned_parameters_dirpath,  self.scale_parameters_filepath)
        X = None
        for i in range(self.train_data_augmentation): 
            if X is None:
                X = np.asarray(feature_extractor.infer_layer_features_from_one_model(os.path.dirname(model_filepath))).reshape((1, -1))
            else:
                X = np.vstack((X, np.asarray(feature_extractor.infer_layer_features_from_one_model(os.path.dirname(model_filepath))).reshape((1, -1))))
        #print(X.shape)
        #with open(self.model_filepath, "rb") as fp:
        #    regressor: RandomForestRegressor = pickle.load(fp)
            
        clf = XGBRegressor(**self.xgboost_kwargs)
        clf.load_model(self.model_filepath);

        probability = str(np.mean(clf.predict(X)).item())

        with open(result_filepath, "w") as fp:
            fp.write(probability)

        logging.info("Trojan probability: %s", probability)