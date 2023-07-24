
from datetime import datetime
from typing import List, Optional
import logging
logger = logging.getLogger('learner')

 
import tensorflow as tf
from tensorflow.summary import scalar
from tensorflow.summary import histogram

import torch

import mlflow.pytorch

import wandb

         

class Learner:
        def __init__(self, **depend_configs):
            wandb.login(key=depend_configs.get('wandb_key', None))
            train_args = depend_configs.get('train_args')
        
        def run(self):
            with mlflow.start_run as run:
                for k, v in self.train_args:
                    mlflow.log_param(k, v)
                learner.use_seed(seed)

                logger.info("Writing TensorFlow events locally to %s\n" % log_dir)
                writer = tf.summary.create_file_writer(log_dir)
                
                for epoch in range(1, train_args.epochs + 1):
                    # print out active_run
                    logger.info("Active Run ID: %s, Epoch: %s \n" % (run.info.run_uuid, epoch))

                    train_info = learner.train(seed)
                    test_info = learner.test(seed)
                    with writer.as_default():
                        for k,v in train_info.items():
                            logger.info('Epoch {} train: {} : {}', epoch, k, v)
                            tf.summary.scalar(k, v, seed, epoch)
                            mlflow.log_metric(k, v, epoch)
                        for k,v in test_info.items():
                            logger.info('Epoch {} test: {} : {}', epoch, k, v)
                            tf.summary.scalar(k, v, seed, epoch)
                            mlflow.log_metric(k, v, epoch)
                    
                logger.info("Uploading TensorFlow events as a run artifact.")
                mlflow.log_artifacts(log_dir, artifact_path="events")
    
    

