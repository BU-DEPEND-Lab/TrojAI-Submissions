import torch.nn as nn
import torch

import numpy as np
 
from abc import ABC, abstractmethod    

import logging
logger = logging.getLogger(__name__)


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.reshape(self.size)
    

class Basic_VAE(ABC, nn.Module):
    def __init__(self, input_size, device, state_embedding_size):
        super().__init__()

        self.input_size = input_size
        self.device = device
        self.state_embedding_size = state_embedding_size
    
    @abstractmethod
    def enc(self, obs):
        raise NotImplementedError
    
    @abstractmethod
    def dec(self, emb):
        raise NotImplementedError
    

    def preprocess_obss(self, obss):
        return torch.tensor(obss, device=self.device).float()

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
     
    def forward(self, obs):
        obs = self.preprocess_obss(obs)
        obs = obs.transpose(1, 3).transpose(2, 3)
        #logger.info(f"{obs.shape}")
        mu, log_var = self.enc(obs)
        z = self.reparameterize(mu, log_var)
        #logger.info(f"{z.shape}")
        obs_ = self.dec(z)
        #logger.info(f"{obs_.shape}")
        obs_ = obs_.transpose(2, 3).transpose(3, 1)
        return obs_, mu, log_var






class Basic_FC_VAE(Basic_VAE):
    """
    Fully connected VAE model. Set architecture that is small and can quick to train (if suited for a given
    task). Successful on Breakout.
    """
    def __init__(self, input_size, device, state_embedding_size = 16):
        """
        Initialize the model.
        :param input_size: (gym.spaces.Space) the observation space of the task
        
        """
        super().__init__(input_size, device, state_embedding_size)
        # Define state embedding
        self.encoder = nn.Sequential(
            nn.Linear(np.prod(self.input_size).item(), 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.state_embedding_size = state_embedding_size

        self.encoder_mu = nn.Linear(32, self.state_embedding_size)
        self.encoder_var = nn.Linear(32, self.state_embedding_size)

        # Define actor's model
        self.decoder = nn.Sequential(
            nn.Linear(self.state_embedding_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, np.prod(self.input_size).item()),
        )
 

    def enc(self, obs):
        feature = self.encoder(obs.float())
        feature = feature.reshape(feature.shape[0], -1)
        mu = self.encoder_mu(feature)
        var = self.encoder_var(feature)
        return mu, var
    
    def dec(self, emb):
        return self.decoder(emb)
    
    def forward(self, obs):
        obs = self.preprocess_obss(obs)
        #logger.info(f"{obs.shape}")
        x = obs.reshape(obs.shape[0], -1)
        #logger.info(f"{x}")
        mu, log_var = self.enc(x)
        z = self.reparameterize(mu, log_var)
        #logger.info(f"{z.shape}")
        x = self.dec(z)
        #logger.info(f"{obs_.shape}")
        x = x.reshape(obs.shape)
        return x, mu, log_var

    
 
class Standard_CNN_VAE(Basic_VAE):
    """
    CNN VAE model for Image Space of Atari gym environments training with the torch_ac library.

    Assumes grayscale image, down-sampled to 84x84.
    """
    def __init__(self, input_size, device, state_embedding_size = 16):
        """
        Initialize the model.
        :param input_size: (gym.spaces.Space) the observation space of the task
        :param action_space: (gym.spaces.Space) the action space of the task
        """
        super().__init__(input_size, device, state_embedding_size)
 

        # currently unneeded, but used to try and maintain consistency of RL implementation
       
 
        # Define image embedding
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            View((-1, 64*4*4)),                  # B, 512
            nn.Linear(64*4*4, 256),              # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, 2 * state_embedding_size),             # B, z_dim*2
        )
        
        
        self.fc_mu = nn.Linear(2 * state_embedding_size, state_embedding_size)
        self.fc_var = nn.Linear(2 * state_embedding_size, state_embedding_size)

        # Define dec_lin model
         

        self.decoder = nn.Sequential(
            nn.Linear(state_embedding_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 16, 3, 1, 1),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, 2, 1), # B,  3,  7,  7
            nn.Tanh()
        )

    def enc(self, obs):
        feature = self.encoder(obs.float())
        mu = self.fc_mu(feature)
        var = self.fc_var(feature)
        return mu, var
    
    def dec(self, x):
        return self.decoder(x) * 100
   