import torch.nn as nn
import torch
 
from abc import ABC, abstractmethod    

class Basic_VAE(ABC, nn.module):
    def __init__(self, input_size, device, state_embedding_size):
        self.input_size = input_size
        self.device = device
        self.state_embedding_size = state_embedding_size
    
    @abstractmethod
    def enc(self, obs):
        raise NotImplementedError
    
    @abstractmethod
    def dec(self, emb):
        raise NotImplementedError
    

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
    
    @abstractmethod
    def forward(self, obs):
        mu, log_var = self.enc(obs)
        z = self.reparameterize(mu, log_var)
        obs_ = self.dec(z)
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
            nn.Linear(input_size.shape[0], 64),
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
            nn.Linear(64, input_size.shape[0]),
        )

    def preprocess_obss(self, obss):
        return torch.tensor(obss, device=self.device)

    def enc(self, obs):
        feature = self.encoder(obs.float())
        mu = self.encoder_mu(feature)
        var = self.encoder_var(feature)
        return mu, var
    
    def dec(self, emb):
        return self.decoder(emb)
    
    
    
 
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
        self.enc_conv = nn.Sequential(
            nn.Conv2d(4, 32, (8, 8), stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), stride=1),
            nn.ReLU()
        )
       
        
        # Define enc_lin model
        self.enc_lin = nn.Sequential(
            nn.Linear(self.feature_ize, 512),
            nn.ReLU(),
            nn.Linear(512, 2 * state_embedding_size)
        )
        
        self.fc_mu = nn.Linear(state_embedding_size, state_embedding_size)
        self.fc_var = nn.Linear(state_embedding_size, state_embedding_size)

        # Define dec_lin model
        self.dec_lin = nn.Sequential(
            nn.Linear(state_embedding_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.feature_ize)
        )

        # Define critic's model
        self.dec_conv = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(4, 32, (8, 8), stride=4),
        )

    def preprocess_obss(self, obss):
        return torch.tensor(obss, device=self.device)

    def enc(self, obs):
        feature = self.enc_conv(obs.float())
        feature = self.enc_lin(feature)
        mu = self.fc_mu(feature)
        var = self.fc_var(feature)
        return mu, var
    
    def dec(self, emb):
        x = self.dec_lin(emb)
        return self.dec_conv(x)
   