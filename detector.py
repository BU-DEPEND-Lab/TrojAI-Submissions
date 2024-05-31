# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import json
import logging
import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import torch

from utils.abstract import AbstractDetector
from utils.models import load_model



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

        self.input_features = metaparameters["train_input_features"]
        self.weight_table_params = {
            "random_seed": metaparameters["train_weight_table_random_state"],
            "mean": metaparameters["train_weight_table_params_mean"],
            "std": metaparameters["train_weight_table_params_std"],
            "scaler": metaparameters["train_weight_table_params_scaler"],
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
            "train_weight_table_random_state": self.weight_table_params["random_seed"],
            "train_weight_table_params_mean": self.weight_table_params["mean"],
            "train_weight_table_params_std": self.weight_table_params["std"],
            "train_weight_table_params_scaler": self.weight_table_params["scaler"],
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
            json.dump(metaparameters, fp)

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
        os.makedirs(self.learned_parameters_dirpath, exist_ok=True)

        # List all available model
        model_path_list = sorted([os.path.join(models_dirpath, model) for model in os.listdir(models_dirpath)])
        logging.info("Found {} models to configure the detector against".format(len(model_path_list)))

        logging.info("Creating detector features")
        X = list()
        y = list()

        for model_index in range(len(model_path_list)):
            model_feats = np.random.randn(100)

            X.append(model_feats)  # random features
            y.append(float(np.random.rand() > 0.5))  # random label

        X = np.stack(X, axis=0)
        y = np.asarray(y)

        logging.info("Training RandomForestRegressor model...")
        model = RandomForestRegressor(**self.random_forest_kwargs, random_state=0)
        model.fit(X, y)

        logging.info("Saving RandomForestRegressor model...")
        with open(os.path.join(self.learned_parameters_dirpath, 'model.bin'), "wb") as fp:
            pickle.dump(model, fp)

        self.write_metaparameters()
        logging.info("Configuration done!")

    def inference_on_example_data(self, model, tokenizer, torch_dtype=torch.float16, stream_flag=False):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model: the pytorch model
            tokenizer: the models tokenizer
            torch_dtype: the dtype to use for inference
            stream_flag: flag controlling whether to put the whole model on the gpu (stream=False) or whether to park some of the weights on the CPU and stream the activations between CPU and GPU as required. Use stream=False unless you cannot fit the model into GPU memory.
        """

        if stream_flag:
            logging.info("Using accelerate.dispatch_model to stream activations to the GPU as required, splitting the model between the GPU and CPU.")
            model.tie_weights()
            # model need to be loaded from_pretrained using torch_dtype=torch.float16 to fast inference, but the model appears to be saved as fp32. How will this play with bfp16?
            # You can't load as 'auto' and then specify torch.float16 later.
            # In fact, if you load as torch.float16, the later dtype can be None, and it works right

            # The following functions are duplicated from accelerate.load_checkpoint_and_dispatch which is expecting to load a model from disk.
            # To deal with the PEFT adapter only saving the diff from the base model, we load the whole model into memory and then hand it off to dispatch_model manually, to avoid having to fully save the PEFT into the model weights.
            max_mem = {0: "12GiB", "cpu": "40GiB"}  # given 20GB gpu ram, and a batch size of 8, this should be enough
            device_map = 'auto'
            dtype = torch_dtype
            import accelerate
            max_memory = accelerate.utils.modeling.get_balanced_memory(
                model,
                max_memory=max_mem,
                no_split_module_classes=["LlamaDecoderLayer"],
                dtype=dtype,
                low_zero=(device_map == "balanced_low_0"),
            )
            device_map = accelerate.infer_auto_device_map(
                model, max_memory=max_memory, no_split_module_classes=["LlamaDecoderLayer"], dtype=dtype
            )

            model = accelerate.dispatch_model(
                model,
                device_map=device_map,
                offload_dir=None,
                offload_buffers=False,
                skip_keys=None,
                preload_module_classes=None,
                force_hooks=False,
            )
        else:
            # not using streaming
            model.cuda()

        # prompt = "As someone who uses quality Premium, I"
        prompt = "The opposite of special education is general education"

        inputs = tokenizer([prompt], return_tensors='pt')
        inputs = inputs.to('cuda')

        outputs = model.generate(**inputs, max_new_tokens=200,
                                 pad_token_id=tokenizer.eos_token_id,
                                 top_p=1.0,
                                 temperature=1.0,
                                 no_repeat_ngram_size=3,
                                 do_sample=False)

        results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        result = results[0]  # unpack implicit batch
        result = result.replace(prompt, '')

        logging.info("Prompt: \n\"\"\"\n{}\n\"\"\"".format(prompt))
        logging.info("Response: \n\"\"\"\n{}\n\"\"\"".format(result))


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

        model, tokenizer = load_model(model_filepath)
        model.cuda()  

        # Inferences on examples to demonstrate how it is done for a round
        # This is not needed for the random forest classifier
        # self.inference_on_example_data(model, tokenizer, torch_dtype=torch.float16, stream_flag=False)

        try:
            # load "trojan" detection model

            #zero-shot detector
            # with open(os.path.join(self.learned_parameters_dirpath, 'model.bin'), "rb") as fp:
            #     regressor: RandomForestRegressor = pickle.load(fp)

            # create RNG "features" about the AI model to feed into the "trojan" detector forest
            prompt = "The opposite of special education is general education Our brains learn how to move and eventually"
            test_prompt1 = "As the ancient manuscript was finally translated the scholars were astonished to discover"
            test_prompt2 = "While exploring the abandoned factory at the edge of town the group stumbled upon a device that apparently"

            # X = np.random.randn(1, 100)  # needs to be 2D, with the features in dim[-1]


            inputs = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
            # Generate initial output
            with torch.no_grad():
                outputs = model(inputs)
            logits = outputs.logits[:, -1, :]
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            sorted_indices = torch.argsort(probabilities, descending=True)
            top_indices = sorted_indices[0, :20000]
            joint_probabilities = []
            candidate_tokens = []
            # index_triple = []
            for i, index in enumerate(top_indices):
                extended_input = torch.cat([inputs, index.view(1, 1).cuda()], dim=1)
                with torch.no_grad():
                    outputs = model(extended_input)
                next_logits = outputs.logits[:, -1, :]
                next_prob = torch.nn.functional.softmax(next_logits, dim=-1)
                top_next_index = torch.argmax(next_prob)
                top_next_prob = next_prob[0, top_next_index]

                if top_next_prob > 0.99:
                    # Append the second most confident token and generate the next token
                    second_extended_input = torch.cat([extended_input, top_next_index.view(1, 1).cuda()], dim=1)
                    with torch.no_grad():
                        second_outputs = model(second_extended_input)
                    second_next_logits = second_outputs.logits[:, -1, :]
                    second_next_prob = torch.nn.functional.softmax(second_next_logits, dim=-1)
                    second_top_next_index = torch.argmax(second_next_prob)
                    second_top_next_prob = second_next_prob[0, second_top_next_index]

                    if second_top_next_prob > 0.99:
                        joint_prob = probabilities[0, index] * top_next_prob.item() * second_top_next_prob.item()
                        # index_triple.append = [index,top_next_index,second_top_next_index]
                        token_triple = tokenizer.decode([index.item(), top_next_index.item(), second_top_next_index.item()])
                        candidate_tokens = candidate_tokens + [index]
                        joint_probabilities.append((token_triple, joint_prob))
                        print(i)
                    if i % 10 == 0:
                        torch.cuda.empty_cache()
            # Sort and display results
            # joint_probabilities.sort(key=lambda x: x[1], reverse=True)
            # for pair, prob in joint_probabilities[:20]:
            #     print(f"{pair}: {prob:.5f}")
            print(candidate_tokens)
            probability = str(0)
 
            for index in candidate_tokens:
                inputs1 = tokenizer(test_prompt1, return_tensors='pt').input_ids.cuda()
                inputs2 = tokenizer(test_prompt2, return_tensors='pt').input_ids.cuda()
                extended_input1 = torch.cat([inputs1, index.view(1, 1).cuda()], dim=1)
                extended_input2 = torch.cat([inputs2, index.view(1, 1).cuda()], dim=1)
                with torch.no_grad():
                    outputs1 = model(extended_input1)
                    outputs2 = model(extended_input2)
                    next_1 = torch.argmax(outputs1.logits[:, -1, :]) 
                    next_2 = torch.argmax(outputs2.logits[:, -1, :]) 
                    if next_1 == next_2:
                        print("trojan")
                        probability = str(1)
                        break
            logging.info(f"detector give you {probability}")
        except Exception as e:
            print(e)
            logging.info('Failed to run detector, there may have an issue during detection')
            probability = str(np.random.rand())
        with open(result_filepath, "w") as fp:
            fp.write(probability)

        logging.info("Trojan probability: %s", probability)
