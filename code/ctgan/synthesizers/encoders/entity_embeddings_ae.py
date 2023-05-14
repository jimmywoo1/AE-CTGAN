import torch
import numpy as np
from torch import nn

class EntityEmbeddingEncoder(nn.Module):
    """Entity Embedding + Decoder for AE-CTGAN Framework"""

    def __init__(self, input_dim: int, hidden_dims: list, output_info: list) -> None:
        super(EntityEmbeddingEncoder, self).__init__()
        enc = []
        dec = []
        embed = []
        categorical = []
        cols = 0
        cat_cols = 0
        embed_output_dim = []

        # collect dimensions of categorical variables
        for i, output in enumerate(output_info):
            if len(output) == 1:
                st_ed = (cols, cols + output[0][0])
                categorical.append(st_ed)
                cat_cols += output[0][0]

            for item in output:
                cols += item[0]
        
        self.categorical = categorical

        # build entity embedding layers
        for st_ed in categorical:
            num_embeddings = st_ed[1] - st_ed[0]
            embedding_dim = min(10, num_embeddings - 1)
            embed.append(nn.Embedding(num_embeddings, embedding_dim))
            embed_output_dim.append(embedding_dim)

        self.embed = nn.ModuleList(embed)
        self.embed_output_dim = embed_output_dim

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
    
    def encode(self, x: torch.tensor) -> torch.tensor:
        mask = np.ones(x.shape, dtype=bool)
        
        # build ctn variables tensors
        for start, end in self.categorical:
            mask[:, start:end] = False
        
        x_ctn = x[mask].reshape(x.shape[0], -1)

        # build entity embedding tensors
        x_ee = [layer(x[:, start:end].long()) for layer, (start, end) in zip(self.embed, self.categorical)]
        x_ee = [nn.functional.max_pool1d(tensor, dim) for tensor, dim in zip(x_ee, self.embed_output_dim)] 
        x_ee = torch.cat(x_ee, dim=1).squeeze()

        # generate input tensor
        x_in = torch.cat((x_ctn, x_ee), dim=1)

        return self.encoder(x)

    def decode(self, x: torch.tensor) -> torch.tensor:
        return self.decoder(x)


    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.encode(x)
        x = self.decode(x)
        
        return x
    


