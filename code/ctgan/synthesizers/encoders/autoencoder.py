import torch
from torch import nn

class AutoEncoder(nn.Module):
    """AutoEncoder for AE-CTGAN Framework"""

    def __init__(self, input_dim: int, hidden_dims: list, noise: str=None) -> None:
        super(AutoEncoder, self).__init__()
        enc = []
        dec = []
        dim = input_dim

        # encoder
        for h_dim in hidden_dims:
            enc.append(nn.Linear(dim, h_dim))
            enc.append(nn.ReLU())
            dim = h_dim

        # decoder
        for h_dim in reversed(hidden_dims[:-1]):
            dec.append(nn.Linear(dim, h_dim))
            dec.append(nn.ReLU())
            dim = h_dim

        dec.append(nn.Linear(dim, input_dim))
        dec.append(nn.ReLU())

        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)
        self.noise = noise

    def encode(self, x: torch.tensor) -> torch.tensor:
        if self.noise is not None:
            noise = torch.randn(x.shape).to(x.device)
            x += noise
        return self.encoder(x)

    def decode(self, x: torch.tensor) -> torch.tensor:
        return self.decoder(x)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.encode(x)
        x = self.decode(x)
        
        return x