import torch.nn as nn
import torch
 


class Basic_FC_VAE(nn.Module):
    """
    Fully connected VAE model. Set architecture that is small and can quick to train (if suited for a given
    task). Successful on Breakout.
    """
    def __init__(self, obs_space, state_embedding_size = 16):
        """
        Initialize the model.
        :param obs_space: (gym.spaces.Space) the observation space of the task
        
        """
        super().__init__()

    
        # currently unneeded, as most values can be hardcoded, but used to try and maintain consistency of RL
        # implementation
        self.obs_space = obs_space  # must be gym.spaces.Box(128,)
 

        self.preprocess_obss = None  # Default torch_ac pre-processing works for this model

        # Define state embedding
        self.encoder = nn.Sequential(
            nn.Linear(obs_space.shape[0], 64),
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
            nn.Linear(64, obs_space.shape[0]),
        )

    def enc(self, obs):
        feature = self.encoder(obs.float())
        mu = self.encoder_mu(feature)
        var = self.encoder_var(feature)
        return mu, var
    
    def dec(self, emb):
        return self.decoder(emb)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> Tensor:
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
        mu, log_var = self.enc(obs)
        z = self.reparameterize(mu, log_var)
        obs_ = self.dec(z)
        return obs_, mu, log_var

class Standard_CNN_VAE(nn.Module):
    """
    CNN VAE model for Image Space of Atari gym environments training with the torch_ac library.

    Assumes grayscale image, down-sampled to 84x84.
    """
    def __init__(self, obs_space, state_embedding_size = 16):
        """
        Initialize the model.
        :param obs_space: (gym.spaces.Space) the observation space of the task
        :param action_space: (gym.spaces.Space) the action space of the task
        """
        super().__init__()

        self.recurrent = False  # required for using torch_ac package

        # currently unneeded, but used to try and maintain consistency of RL implementation
        self.obs_space = obs_space
 
        # Define image embedding
        self.enc_conv = nn.Sequential(
            nn.Conv2d(4, 32, (8, 8), stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), stride=1),
            nn.ReLU()
        )
        feature_size = 7 * 7 * 64
        
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

    def preprocess_obss(self, obss, device=None):
        return torch.tensor(obss, device=device)

    def enc(self, obs):
        feature = self.enc_conv(obs.float())
        feature = self.enc_lin(feature)
        mu = self.fc_mu(feature)
        var = self.fc_var(feature)
        return mu, var
    
    def dec(self, emb):
        x = self.dec_lin(emb)
        return self.dec_conv(x)
    
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
        mu, log_var = self.enc(obs)
        z = self.reparameterize(mu, log_var)
        obs_ = self.dec(z)
        return obs_, mu, log_var

