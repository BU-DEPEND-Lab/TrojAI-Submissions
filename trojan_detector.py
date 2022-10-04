# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
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


def example_trojan_detector(model_filepath,
                            result_filepath,
                            scratch_dirpath,
                            examples_dirpath,
                            source_dataset_dirpath,
                            round_training_dataset_dirpath,
                            parameters_dirpath,
                            config):
    logging.info('model_filepath = {}'.format(model_filepath))
    logging.info('result_filepath = {}'.format(result_filepath))
    logging.info('scratch_dirpath = {}'.format(scratch_dirpath))
    logging.info('examples_dirpath = {}'.format(examples_dirpath))
    logging.info('source_dataset_dirpath = {}'.format(source_dataset_dirpath))
    logging.info('round_training_dataset_dirpath = {}'.format(round_training_dataset_dirpath))
    logging.info('round_training_dataset_dirpath = {}'.format(round_training_dataset_dirpath))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using compute device: {}".format(device))
    
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




def configure(source_dataset_dirpath,
              output_parameters_dirpath,
              configure_models_dirpath):
    logging.info('Configuring detector parameters with models from ' + configure_models_dirpath)

    os.makedirs(output_parameters_dirpath, exist_ok=True)

    logging.info('Writing configured parameter data to ' + output_parameters_dirpath)

    logging.info('Reading source dataset from ' + source_dataset_dirpath)

    arr = np.random.rand(100,100)
    np.save(os.path.join(output_parameters_dirpath, 'numpy_array.npy'), arr)

    with open(os.path.join(output_parameters_dirpath, "single_number.txt"), 'w') as fh:
        fh.write("{}".format(17))

    example_dict = dict()
    example_dict['keya'] = 2
    example_dict['keyb'] = 3
    example_dict['keyc'] = 5
    example_dict['keyd'] = 7
    example_dict['keye'] = 11
    example_dict['keyf'] = 13
    example_dict['keyg'] = 17

    with open(os.path.join(output_parameters_dirpath, "dict.json"), mode='w', encoding='utf-8') as f:
        f.write(jsonpickle.encode(example_dict, warn=True, indent=2))


if __name__ == "__main__":
    from jsonargparse import ArgumentParser, ActionConfigFile

    parser = ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the directory containing json file(s) that contains the examples which might be useful for determining whether a model is poisoned.')

    parser.add_argument('--source_dataset_dirpath', type=str, help='File path to a directory containing the original clean dataset into which triggers were injected during training.', default=None)
    parser.add_argument('--round_training_dataset_dirpath', type=str, help='File path to the directory containing id-xxxxxxxx models of the current rounds training dataset.', default=None)

    parser.add_argument('--metaparameters_filepath', help='Path to JSON file containing values of tunable paramaters to be used when evaluating models.', action=ActionConfigFile)
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
            with open(args.metaparameters_filepath[0]()) as config_file:
                config_json = json.load(config_file)

            with open(args.schema_filepath) as schema_file:
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
            args.learned_parameters_dirpath is not None):
            print(vars(args))
            logging.info("Calling the trojan detector")
            example_trojan_detector(args.model_filepath,
                                    args.result_filepath,
                                    args.scratch_dirpath,
                                    args.examples_dirpath,
                                    args.source_dataset_dirpath,
                                    args.round_training_dataset_dirpath,
                                    args.learned_parameters_dirpath,args)
        else:
            logging.info("Required Evaluation-Mode parameters missing!")
    else:
        if (args.source_dataset_dirpath is not None and
            args.learned_parameters_dirpath is not None and
            args.configure_models_dirpath is not None):

            logging.info("Calling configuration mode")
            # all 3 example parameters will be loaded here, but we only use parameter3
            configure(args.source_dataset_dirpath,
                      args.learned_parameters_dirpath,
                      args.configure_models_dirpath)
        else:
            logging.info("Required Configure-Mode parameters missing!")
