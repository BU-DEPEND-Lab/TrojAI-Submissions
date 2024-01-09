# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import math
import re
from collections import OrderedDict
from os.path import join

import gymnasium as gym
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
        for k, v in obs.items():
            if k == 'image':
                x = self.state_emb(obs['image'].float())
                x = x.reshape(x.shape[0], -1)
            elif k == 'direction':
                x = torch.concat([x, obs['direction'].long()], dim=1)
            else:
                #logger.info(f"{k}: x shape = {x.shape} and obs[k] shape = {obs[k].shape}")
                x = torch.concat([x, obs[k]], dim=1)
        return self.classifier(x)
  

    def args_dict(self):
        raise NotImplementedError("Should be implemented in subclass")

    def load_from_file(self, path):
        self.load_state_dict(torch.load(self.config.model.classifier.load_from_file)['state_dict'])



class FCModel(ClassifierBackbone):
    """Fully-connected actor-critic model with shared embedding"""

    def __init__(self, obs_space, linear_embedding_dims=(512, 256), actor_linear_mid_dims=(64, 32), extra_size = 0,
                 critic_linear_mid_dims=(64, 32)):
        """
        Initialize the model.
        :param obs_space: (gym.Spaces) Observation space of the environment being used for training.
        :param action_space: (gym.Spaces) Action space of the environment being used for training. Used to determine
            the size of the actor's output later.
        :param linear_embedding_dims: (iterable) Sequence of integers representing the number of hidden nodes in the
            intermediate layers of the linear embedding
        :param actor_linear_mid_dims: (iterable) Sequence of integers representing the number of hidden nodes in the
            intermediate layers of the actor network
        :param critic_linear_mid_dims: (iterable) Sequence of integers representing the number of hidden nodes in the
            intermediate layers of the critic network
        """
        self.obs_space = obs_space
        if isinstance(self.obs_space, gym.spaces.Dict):
            flattened_dims = int(math.prod(self.obs_space['image'].shape))
        else:
            flattened_dims = int(math.prod(self.obs_space.shape))
        self.action_space = 1
        self.linear_embedding_dims = linear_embedding_dims
        self.actor_linear_mid_dims = actor_linear_mid_dims
        

        # +1 because we concat direction information to embedding
        self.state_embedding_size = linear_embedding_dims[-1] + 1 + extra_size
        embedding = linear_w_relu([flattened_dims] + [d for d in linear_embedding_dims])
        embedding.insert(0, nn.Flatten())  # put a flattening layer in front
        actor_dims = [self.state_embedding_size] + list(actor_linear_mid_dims) + [1]
        super().__init__(
            embedding,
            linear_w_relu(actor_dims),
        )

    def args_dict(self):
        return {
            'obs_space': self.obs_space,
            'action_space': 1,
            'linear_embedding_dims': self.linear_embedding_dims,
            'actor_linear_mid_dims': self.actor_linear_mid_dims,
        }


class CNNModel(ClassifierBackbone):
    """Simple actor-critic model with CNN embedding"""

    def __init__(self, obs_space, channels=(16, 32, 64), actor_linear_mid_dims=(144,), extra_size = 0,
                 critic_linear_mid_dims=(144,)):
        """
        Initialize the model.
        :param obs_space: (gym.Spaces) Observation space of the environment being used for training.
        :param action_space: (gym.Spaces) Action space of the environment being used for training. Used to determine
            the size of the actor's output later.
        :param channels: (iterable) Sequence of 3 integers representing the number of numbers of channels to use for the
            CNN embedding
        :param actor_linear_mid_dims: (iterable) Sequence of integers representing the number of hidden nodes in the
            intermediate layers of the actor network
        :param critic_linear_mid_dims: (iterable) Sequence of integers representing the number of hidden nodes in the
            intermediate layers of the critic network
        """
        if len(channels) != 3:
            raise ValueError("'channels' must be a tuple or list of length 3")

        self.obs_space = obs_space
        self.action_space = 1
        self.channels = channels
        self.actor_linear_mid_dims = actor_linear_mid_dims
        
        c1, c2, c3 = channels
        image_embedding_size = 4 * 4 * c3 + extra_size +1 #because we concat direction information to embedding
        image_conv = nn.Sequential(
            nn.Conv2d(3, c1, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(c1, c2, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(c2, c3, (2, 2)),
            nn.ReLU()
        )
        actor_dims = [image_embedding_size] + list(actor_linear_mid_dims) + [1]
        
        super().__init__(
            image_conv,
            linear_w_relu(actor_dims) 
        )

    def args_dict(self):
        return {
            'obs_space': self.obs_space,
            'action_space': self.action_space,
            'channels': self.channels,
            'actor_linear_mid_dims': self.actor_linear_mid_dims 
        }


class ImageACModel(ClassifierBackbone):
    """Simple CNN Actor-Critic model designed for MiniGrid. Assumes 48x48 grayscale or RGB images."""

    def __init__(self, obs_space, action_space, channels=(8, 16, 32), actor_linear_mid_dims=(144,), extra_size = 1,
                 critic_linear_mid_dims=(144,)):
        """
        Initialize the model.
        :param obs_space: (gym.Spaces) Observation space of the environment being used for training. Technically unused
            for this model, but stored both for consistency between models and to be used for later reference if needed.
        :param action_space: (gym.Spaces) Action space of the environment being used for training. Used to determine
            the size of the actor's output later.
        :param channels: (iterable) Sequence of 3 integers representing the number of numbers of channels to use for the
            CNN embedding
        :param actor_linear_mid_dims: (iterable) Sequence of integers representing the number of hidden nodes in the
            intermediate layers of the actor network
        :param critic_linear_mid_dims: (iterable) Sequence of integers representing the number of hidden nodes in the
            intermediate layers of the critic network
        """

        self.obs_space = obs_space
        self.action_space = action_space
        self.channels = channels
        self.actor_linear_mid_dims = actor_linear_mid_dims
        self.critic_linear_mid_dims = critic_linear_mid_dims
        self.image_size = 48  # this is the size of image this CNN was designed for

        num_channels = obs_space['image'].shape[0]
        c1, c2, c3 = channels
        image_embedding_size = 3 * 3 * c3 + extra_size  # +1 because we concat direction information to embedding
        image_conv = nn.Sequential(
            nn.Conv2d(num_channels, c1, (3, 3), stride=3),
            nn.ReLU(),
            nn.Conv2d(c1, c2, (4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(c2, c3, (3, 3), stride=2),
            nn.ReLU()
        )
        actor_dims = [image_embedding_size] + list(actor_linear_mid_dims) + [action_space.n]
        critic_dims = [image_embedding_size] + list(critic_linear_mid_dims) + [1]
        super().__init__(
            image_conv,
            linear_w_relu(actor_dims),
            linear_w_relu(critic_dims)
        )

    def args_dict(self):
        return {
            'obs_space': self.obs_space,
            'action_space': self.action_space,
            'channels': self.channels,
            'actor_linear_mid_dims': self.actor_linear_mid_dims,
            'critic_linear_mid_dims': self.critic_linear_mid_dims
        }


class ResNetACModel(ClassifierBackbone):
    """Actor-Critic model with ResNet18 embedding designed for MiniGrid. Assumes 112x112 RGB images."""

    def __init__(self, obs_space, action_space, actor_linear_mid_dims=(512,), critic_linear_mid_dims=(512,), extra_size = 1):
        """
        Initialize the model.
        :param obs_space: (gym.Spaces) Observation space of the environment being used for training. Technically unused
            for this model, but stored both for consistency between models and to be used for later reference if needed.
        :param action_space: (gym.Spaces) Action space of the environment being used for training. Used to determine
            the size of the actor's output later.
        :param actor_linear_mid_dims: (iterable) Sequence of integers representing the number of hidden nodes in the
            intermediate layers of the actor network
        :param critic_linear_mid_dims: (iterable) Sequence of integers representing the number of hidden nodes in the
            intermediate layers of the critic network
        """
        self.obs_space = obs_space
        self.action_space = action_space
        self.actor_linear_mid_dims = actor_linear_mid_dims
        self.critic_linear_mid_dims = critic_linear_mid_dims

        image_embedding_size = 512 + extra_size  # +1 because we concat direction information to embedding
        embedding = ModdedResnet18()
        actor_dims = [image_embedding_size] + list(actor_linear_mid_dims) + [action_space.n]
        critic_dims = [image_embedding_size] + list(critic_linear_mid_dims) + [1]
        super().__init__(
            embedding,
            linear_w_relu(actor_dims),
            linear_w_relu(critic_dims)
        )

    def args_dict(self):
        return {
            'obs_space': self.obs_space,
            'action_space': self.action_space,
            'actor_linear_mid_dims': self.actor_linear_mid_dims,
            'critic_linear_mid_dims': self.critic_linear_mid_dims
        }

