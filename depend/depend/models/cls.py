# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import math
import re
from collections import OrderedDict
from os.path import join
 
import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import BasicBlock

import logging
logger = logging.getLogger(__file__)

def linear_w_relu(dims: list, end_sigmoid=True):
    """Helper function for creating sequential linear layers"""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.ReLU())
    if end_sigmoid:
        layers[-1] = nn.Sigmoid()
    return nn.Sequential(*layers)


class ModdedResnet18(torchvision.models.ResNet):
    """Modified ResNet18 architecture for TrojAI DRL ResNet models"""
    def __init__(self):
        super(ModdedResnet18, self).__init__(BasicBlock, [2, 2, 2, 2])

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    def load_from_file(self, path):
        self.load_state_dict(torch.load(self.config.model.classifier.load_from_file)['state_dict'])


class ClassifierBackbone(nn.Module):
    """Base class for TrojAI DRL models"""

    def __init__(self, embedding, classifier):
        nn.Module.__init__(self)
        self.state_emb = embedding  # Define state embedding
        self.classifier = classifier # Define actor's model
        

    def forward(self, obs):
        x = None
        x = self.state_emb(x)
        return self.classifier(x)
  

    def args_dict(self):
        raise NotImplementedError("Should be implemented in subclass")

    def load_from_file(self, path):
        self.load_state_dict(torch.load(self.config.model.classifier.load_from_file)['state_dict'])



class FCModel(ClassifierBackbone):
    """Fully-connected actor-critic model with shared embedding"""

    def __init__(self, input_size = 991, embedding_dims=[512, 256], cls_dims = [64, 32]):
        """
        Initialize the model.
        :param input size: input size 
        :param hidden_dims: hidden layer dimensions.
        """
        self.input_size = input_size
         
        embeddings = linear_w_relu(embedding_dims, False)
        cls = linear_w_relu(cls_dims + [1])
        
        super().__init__(
            embeddings,
            cls,
        )

    def args_dict(self):
        return {
            'input_size': self.input_size,
        }

 