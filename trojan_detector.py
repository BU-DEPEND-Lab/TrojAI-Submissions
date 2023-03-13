# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
from os import listdir, makedirs
from os.path import join, exists, basename
import numpy as np
import cv2
import torch
import torchvision
import json
import jsonschema
import jsonpickle

import logging
import warnings

warnings.filterwarnings("ignore")
from utils.models import create_layer_map, load_model, \
    load_models_dirpath, load_ground_truth, get_loss

import tensorflow as tf
from sklearn.linear_model import SGDClassifier 
import sklearn.model_selection
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from joblib import dump, load
from sklearn import metrics
from sklearn import svm

# keras import
from arch.cnn import create_keras_cnn_model 
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import model_from_json

import torch
from vit_pytorch import SimpleViT
import torch.nn as nn
import torch.optim as optim

from feature_extractor import FeatureExtractor   
import xgboost
from xgboost import cv, XGBRegressor, XGBClassifier




def prepare_boxes(anns, image_id):
    if len(anns) > 0:
        boxes = []
        class_ids = []
        for answer in anns:
            boxes.append(answer['bbox'])
            class_ids.append(answer['category_id'])

        class_ids = np.stack(class_ids)
        boxes = np.stack(boxes)
        # convert [x,y,w,h] to [x1, y1, x2, y2]
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    else:
        class_ids = np.zeros((0))
        boxes = np.zeros((0, 4))

    degenerate_boxes = (boxes[:, 2:] - boxes[:, :2]) < 8
    degenerate_boxes = np.sum(degenerate_boxes, axis=1)
    if degenerate_boxes.any():
        boxes = boxes[degenerate_boxes == 0, :]
        class_ids = class_ids[degenerate_boxes == 0]
    target = {}
    target['boxes'] = torch.as_tensor(boxes)
    target['labels'] = torch.as_tensor(class_ids).type(torch.int64)
    target['image_id'] = torch.as_tensor(image_id)
    return target


def write_metaparameters(learned_parameters_dirpath, metaparameters_filepath, **metaparameters):
    metaparameters_base = { 
    }
    metaparameters_base.update(metaparameters)

    with open(join(learned_parameters_dirpath, basename(metaparameters_filepath)), "w") as fp:
        json.dump(metaparameters_base, fp)


def example_trojan_detector(model_filepath,
                            result_filepath,
                            scratch_dirpath,
                            examples_dirpath,
                            source_dataset_dirpath,
                            metaparameters_filepath,
                            round_training_dataset_dirpath,
                            learned_parameters_dirpath
                            ):
    logging.info('model_filepath = {}'.format(model_filepath))
    logging.info('result_filepath = {}'.format(result_filepath))
    logging.info('scratch_dirpath = {}'.format(scratch_dirpath))
    logging.info('examples_dirpath = {}'.format(examples_dirpath))
    logging.info('source_dataset_dirpath = {}'.format(source_dataset_dirpath))
    logging.info('round_training_dataset_dirpath = {}'.format(round_training_dataset_dirpath))
    logging.info('round_training_dataset_dirpath = {}'.format(round_training_dataset_dirpath))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using compute device: {}".format(device))
    
    """
    logging.info('Extracting features')
    import analyzer_weight as feature_extractor
    fv=feature_extractor.extract_fv(model_filepath=model_filepath, scratch_dirpath=scratch_dirpath, examples_dirpath=examples_dirpath, params=config);
    fvs={'fvs':[fv]};
    
    import importlib
    import pandas
    logging.info('Running Trojan classifier')
    if not parameters_dirpath is None:
        checkpoint=os.path.join(parameters_dirpath,'model.pt')
        try:
            checkpoint=torch.load(os.path.join(parameters_dirpath,'model.pt'));
        except:
            checkpoint=torch.load(os.path.join('/',parameters_dirpath,'model.pt'));
        
        #Compute ensemble score 
        scores=[];
        for i in range(len(checkpoint)):
            params_=checkpoint[i]['params'];
            arch_=importlib.import_module(params_.arch);
            net=arch_.new(params_);
            
            net.load_state_dict(checkpoint[i]['net']);
            net=net.cuda();
            net.eval();
            
            s_i=net.logp(fvs).data.cpu();
            s_i=s_i#*math.exp(-checkpoint[i]['T']);
            scores.append(float(s_i))
        
        scores=sum(scores)/len(scores);
        scores=torch.sigmoid(torch.Tensor([scores])); #score -> probability
        trojan_probability=float(scores);
    else:
        trojan_probability=0.5;
    
    logging.info('Trojan Probability: {}'.format(trojan_probability))
    with open(result_filepath, 'w') as fh:
        fh.write("{}".format(trojan_probability))
    """
    
    metaparameters = json.load(open(metaparameters_filepath, "r"))
   
 
    loss = metaparameters["objective"]
    if loss == 'focal_loss':
        loss = get_loss (loss, use_sigmoid = True)
        objective = loss.get_objective(metaparameters["gamma"])
    else:
        loss = get_loss (loss)
        objective = loss.get_objective()
 
    
    
    #model_name = "xgboost"
    #clf = XGBRegressor(seed = 20)

    #model_name = "cnn_regressor"
     
    # load json and create model
    json_file = open(join(learned_parameters_dirpath, "model.json"), 'r')
    clf_json = json_file.read()
    json_file.close()
    clf = model_from_json(clf_json)
    # load weights into new model
    clf.load_weights(join(learned_parameters_dirpath, "model.h5"))
    print("Loaded model from disk")
    
    # evaluate loaded model on test data
    #clf.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #score = loaded_model.evaluate(X, Y, verbose=0)
    feature_extractor = FeatureExtractor(metaparameters_filepath, learned_parameters_dirpath)

    """
    model_path_list = sorted([join(source_dataset_dirpath, model) for model in listdir(source_dataset_dirpath)])[:20]
    logging.info(f"Loading % models ...", len(model_path_list))
    
    X = None
    X =  np.vstack(np.asarray([feature_extractor.infer_attribution_feature_from_models(model_path_list, False)])).transpose(0, 2, 3, 1)
    
    logging.info(f"dataset size: {X.shape}")

    Y = None
    for model_path in model_path_list:
        y = load_ground_truth(model_path)
        if Y is None:
            Y = y * np.ones([feature_extractor.num_data_per_model, 1])
            continue
        else:
            Y = np.vstack((Y, y * np.ones([feature_extractor.num_data_per_model, 1])))
    logging.info(f"label size: {Y.shape}")
    Y_pred = loss.prob(clf.predict(X))
    if True:
        #if not isinstance(objective, str):
        print("Testing comparison:\n", Y.reshape(-1), "\n", Y_pred)
        print('test acc', accuracy_score(Y.reshape(-1), np.asarray(loss.prob(Y_pred) >= 0.5)).reshape(-1))
        #else:
        #    print("Testing comparison:\n", y_test.reshape(-1), "\n", y_pred_ >= 0.5)
        #    print('test acc', accuracy_score(y_test.reshape(-1), np.asarray(y_pred_ >= 0.5)))
        
        fpr, tpr, thresholds = metrics.roc_curve(Y.reshape(-1), Y_pred.reshape(-1))
        print(f'test fpr {fpr}')
        print(f'tpr {tpr}')
        print('test auc', metrics.auc(fpr, tpr))
    """
    #with open(source_dataset_dirpath, "rb") as fp:
    #    regressor: RandomForestRegressor = pickle.load(fp)
    x = feature_extractor.infer_attribution_feature_from_one_model(os.path.dirname(model_filepath), False)#.transpose(0, 2, 3, 1)
    
    probability = str(np.mean(np.abs(loss.prob(clf.predict(x)))).item())
    #if not isinstance(objective, str):
    #else:
    #     probability = str(np.mean(clf.predict(X)).item())

    with open(result_filepath, "w") as fp:
        fp.write(probability)

    logging.info("Trojan probability: %s", probability)




def configure(source_dataset_dirpath,
              metaparameters_filepath,
              learned_parameters_dirpath
              ):
    
    os.makedirs(learned_parameters_dirpath, exist_ok=True)

    logging.info('Writing configured parameter data to ' + learned_parameters_dirpath)

    logging.info('Reading source dataset from ' + source_dataset_dirpath)

    arr = np.random.rand(100,100)
    np.save(os.path.join(learned_parameters_dirpath, 'numpy_array.npy'), arr)

    with open(os.path.join(learned_parameters_dirpath, "single_number.txt"), 'w') as fh:
        fh.write("{}".format(17))
    
    metaparameters = json.load(open(metaparameters_filepath, "r")) 
    loss = metaparameters["objective"]
    if loss == 'focal_loss':
        loss = get_loss (loss, use_sigmoid = True)
        objective = loss.get_objective(metaparameters["gamma"])
    else:
        loss = get_loss (loss)
        objective = loss.get_objective()
 
    
    model_path_list = sorted([join(source_dataset_dirpath, model) for model in listdir(source_dataset_dirpath)])[:32]
    logging.info(f"Loading {len(model_path_list)} models ...")

    feature_extractor = FeatureExtractor(metaparameters_filepath, learned_parameters_dirpath)
    X = None
    Y = None     
    X =  np.vstack(np.asarray([feature_extractor.infer_attribution_feature_from_models(model_path_list, True)]))#.transpose(0, 2, 3, 1)
    
    logging.info(f"dataset size: {X.shape}")
    
    for model_path in model_path_list:
        y = load_ground_truth(model_path)
        if Y is None:
            Y = y * np.ones([feature_extractor.num_data_per_model, 1])
            continue
        else:
            Y = np.vstack((Y, y * np.ones([feature_extractor.num_data_per_model, 1])))
    logging.info(f"label size: {Y.shape}")
    print(np.count_nonzero(np.isnan(Y)))
    write_metaparameters(learned_parameters_dirpath, metaparameters_filepath, **feature_extractor.write_metaparameters())
    feature_extractor = None
    
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, random_state=1)

    #y_train = y_train.repeat(x_train.shape[1], axis = 1).reshape((-1, 1))
    #x_train = x_train.reshape((-1, x_train.shape[-1]))
    #y_train = y_train.reshape((-1, y_train.shape[-1]))
    ids_train = np.arange(x_train.shape[0])
    np.random.shuffle(ids_train)

    y_train = y_train[ids_train]
    x_train = x_train[ids_train]

    #y_test = y_test.repeat(x_test.shape[1], axis = 1).reshape((-1, 1))
    #x_test = x_test.reshape((-1, x_test.shape[-1]))
    ids_test = np.arange(x_test.shape[0])
    np.random.shuffle(ids_test)

    y_test = y_test[ids_test]
    x_test = x_test[ids_test]


    print('x_train', x_train.shape)
    print("y_train", y_train.shape)
    print(y_train)
    print('x_test', x_test.shape)
    print('y_test', y_test.shape)

    model_name = "cnn_regressor"
    clf = SimpleViT(image_size = x_train.shape[-1], patch_size = 32, num_classes = 1, dim = 64, depth = 3, heads = 8, mlp_dim = 128)
    
    
    criterion = torch.nn.BCELoss()
    optimizer = optim.SGD(clf.parameters(), lr=0.001, momentum=0.9)
    testloader = torch.utils.data.DataLoader(list(zip(x_test, y_test)), batch_size=16,
                                          shuffle=True, num_workers=1)
    for epoch in range(20):
        clf.train()
        trainloader = torch.utils.data.DataLoader(list(zip(x_train, y_train)), batch_size=16,
                                          shuffle=True, num_workers=1)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = clf(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}, acc: {((outpus >= 0.5).long().numpy() == labels.long()).sum().item() * 100. / labels.shape[0]}')
                running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = clf(images.float())
                # the class with the highest energy is what we choose as prediction
                predicted = (outputs.data > 0.5).long()
                total += labels.size(0)
                print(images.shape, predicted.shape, labels.shape)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    
    """
    model = KerasClassifier(build_fn=create_cnn_model, verbose=1)
    
    params = {
    'input_shape': [X.shape[1:]],
    'pool_type': ['max'],
    'conv_activation': ['tanh'],    
    'epochs': [10],
    }
    estimator = model
    
    model_name = "xgboost_regressor"
    data_dmatrix = xgboost.DMatrix(data=x_train,label= y_train)
    
    params = { 
        # 'objective': ["reg:logistic"], 
        'max_depth': [10, 20, 40],
        'learning_rate': [0.01, 0.1, 0.001],
        'subsample': np.arange(0.5, 1.0, 0.2),
        'colsample_bytree': np.arange(0.4, 1.0, 0.2),
        'colsample_bylevel': np.arange(0.4, 1.0, 0.2),
        'n_estimators': [100, 500, 1000, 2000]}
    estimator = XGBRegressor(objective = objective, seed = 20)
    """ 
    """       
    hyp_src = RandomizedSearchCV(estimator=estimator,
                     param_distributions=params,
                     scoring='neg_root_mean_squared_error',
                     n_iter=25, cv = 5, n_jobs = -1, refit = True,
                     verbose=1)
    
    hyp_src = GridSearchCV(estimator=estimator,
                        param_grid=params,
                        scoring='roc_auc',
                        n_jobs=-1, refit=True, cv=5, verbose=1, 
                        return_train_score=True) 
    
    hyp_src.fit(x_train, y_train)
    clf = hyp_src.best_estimator_ #XGBRegressor(**rand.best_params_)
    """
    y_pred_ = clf.predict(x_test)
    
    if 'svm' in model_name:
        print("Testing comparison:\n", y_test.reshape(-1), "\n",y_pred_>= 0)
        print('test acc', accuracy_score(y_test.reshape(-1), y_pred_ >= 0))
        y_pred_probs_ = clf.predict_proba(x_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_probs_[:, 1])
    elif "regressor" in model_name:
        #if not isinstance(objective, str):
        print("Testing comparison:\n", y_test.reshape(-1), "\n", y_pred_)
        print('test acc', accuracy_score(y_test.reshape(-1), np.asarray(loss.prob(y_pred_) >= 0.5)).reshape(-1))
        #else:
        #    print("Testing comparison:\n", y_test.reshape(-1), "\n", y_pred_ >= 0.5)
        #    print('test acc', accuracy_score(y_test.reshape(-1), np.asarray(y_pred_ >= 0.5)))
        y_pred_ = loss.prob(clf.predict(x_test))
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_)
     
    print(f'test fpr {fpr}')
    print(f'tpr {tpr}')
    print('test auc', metrics.auc(fpr, tpr))
    logging.info("Saving model...")
    #dump(clf, f'round12_{model_name}.joblib') 

    logging.info("Now train on all dataset")
    
    X = np.vstack((x_train, x_test))
    Y = np.vstack((y_train, y_test))
    clf.fit(X, Y) 
    
    y_pred = loss.prob(clf.predict(X))
    fpr, tpr, thresholds = metrics.roc_curve(Y, y_pred)
    print(f'all dataset fpr {fpr}')
    print(f'tpr {tpr}')
    print('all dataset auc', metrics.auc(fpr, tpr))
        
    #logging.info("Training RandomForestRegressor model...")
    #model = RandomForestRegressor(**random_forest_kwargs, random_state=0)
    #model.fit(X, y)

    
    #logging.info("Saving RandomForestRegressor model...")
    #logging.info("Saving XGBoostRegressor model...")
    
    #with open(source_dataset_dirpath, "wb") as fp:
    #    pickle.dump(model, fp)
    
    if "xgboost" in model_name:
        clf.save_model(join(learned_parameters_dirpath, "model.json"))
    if "cnn" in model_name:
        # serialize model to JSON
        clf_json = clf.model.to_json()
        with open(join(learned_parameters_dirpath, "model.json"), "w") as json_file:
            json_file.write(clf_json)
        # serialize weights to HDF5
        clf.model.save_weights(join(learned_parameters_dirpath, "model.h5"))
        print("Saved model to disk")
    
    logging.info("Configuration done!")




if __name__ == "__main__":
    from jsonargparse import ArgumentParser, ActionConfigFile

    parser = ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the directory containing json file(s) that contains the examples which might be useful for determining whether a model is poisoned.')

    parser.add_argument('--source_dataset_dirpath', type=str, help='File path to a directory containing the original clean dataset into which triggers were injected during training.', default=None)
    parser.add_argument('--round_training_dataset_dirpath', type=str, help='File path to the directory containing id-xxxxxxxx models of the current rounds training dataset.', default=None)

    parser.add_argument('--metaparameters_filepath', help='Path to JSON file containing values of tunable paramaters to be used when evaluating models.', type = str)
    parser.add_argument('--schema_filepath', type=str, help='Path to a schema file in JSON Schema format against which to validate the config file.', default=None)
    parser.add_argument('--learned_parameters_dirpath', type=str, help='Path to a directory containing parameter data (model weights, etc.) to be used when evaluating models.  If --configure_mode is set, these will instead be overwritten with the newly-configured parameters.')

    parser.add_argument('--configure_mode', help='Instead of detecting Trojans, set values of tunable parameters and write them to a given location.', default=False, action="store_true")
    parser.add_argument('--configure_models_dirpath', type=str, help='Path to a directory containing models to use when in configure mode.')

    # these parameters need to be defined here, but their values will be loaded from the json file instead of the command line
    parser.add_argument('--nbins', type=int, help='Number of histogram bins in feature.')
    parser.add_argument('--szcap', type=int, help='Matrix size cap in feature extraction.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s")
    logging.info("example_trojan_detector.py launched")
   
    # Validate config file against schema
    if args.metaparameters_filepath is not None:
        if args.schema_filepath is not None:
            with open(args.metaparameters_filepath, 'rb') as config_file:
                config_json = json.load(config_file)

            with open(args.schema_filepath, 'rb') as schema_file:
                schema_json = json.load(schema_file)

            # this throws a fairly descriptive error if validation fails
            jsonschema.validate(instance=config_json, schema=schema_json)

    if not args.configure_mode:
        if (args.model_filepath is not None and
            args.result_filepath is not None and
            args.scratch_dirpath is not None and
            args.examples_dirpath is not None and
            args.source_dataset_dirpath is not None and
            args.round_training_dataset_dirpath is not None and
            args.metaparameters_filepath is not None and
            args.learned_parameters_dirpath is not None):
            print(vars(args))
            logging.info("Calling the trojan detector")
            example_trojan_detector(args.model_filepath,
                                    args.result_filepath,
                                    args.scratch_dirpath,
                                    args.examples_dirpath,
                                    args.source_dataset_dirpath,
                                    args.metaparameters_filepath,
                                    args.round_training_dataset_dirpath,
                                    args.learned_parameters_dirpath)
        else:
            logging.info("Required Evaluation-Mode parameters missing!")
    else:
        if (args.source_dataset_dirpath is not None and
            args.metaparameters_filepath is not None and
            args.learned_parameters_dirpath is not None):

            logging.info("Calling configuration mode")
            # all 3 example parameters will be loaded here, but we only use parameter3
            configure(args.source_dataset_dirpath,
                      args.metaparameters_filepath,
                      args.learned_parameters_dirpath
                      )
        else:
            logging.info("Required Configure-Mode parameters missing!")
