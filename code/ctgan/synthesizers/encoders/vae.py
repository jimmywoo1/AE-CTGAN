import torch
from torch import nn

class VariationalAutoEncoder(nn.Module):
    """Variational AutoEncoder for AE-CTGAN Framework"""

    def __init__(self, input_dim: int, hidden_dims: list) -> None:
        super(VariationalAutoEncoder, self).__init__()
        enc = []
        dec = []
        dim = input_dim

        # encoder
        for h_dim in hidden_dims[:-1]:
            enc.append(nn.Linear(dim, h_dim))
            enc.append(nn.ReLU())
            dim = h_dim

        mu_hidden = [nn.Linear(dim, hidden_dims[-1]),
                     nn.ReLU()]
        log_var_hidden = [nn.Linear(dim, hidden_dims[-1]),
                          nn.ReLU()]

        dim = hidden_dims[-1]

        # decoder
        for h_dim in reversed(hidden_dims[:-1]):
            dec.append(nn.Linear(dim, h_dim))
            dec.append(nn.ReLU())
            dim = h_dim

        dec.append(nn.Linear(dim, input_dim))
        dec.append(nn.ReLU())

        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)
        self.mu_hidden = nn.Sequential(*mu_hidden)
        self.log_var_hidden = nn.Sequential(*log_var_hidden)

    def reparameterize(self, mu: torch.tensor, log_var: torch.tensor):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        
        return eps * std + mu

    def encode(self, x: torch.tensor) -> torch.tensor:
        x = self.encoder(x)
        self.mu = self.mu_hidden(x)
        self.log_var = self.log_var_hidden(x)
        
        return self.reparameterize(self.mu, self.log_var)
    
    def decode(self, x: torch.tensor) -> torch.tensor:
        return self.decoder(x)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.encode(x)
        x = self.decode(x)
        self.kld = torch.mean(-0.5 * torch.sum(1 + self.log_var - self.mu ** 2 - self.log_var.exp(), dim = 1), dim = 0)        

        return x