import torch
import torch.nn as nn
import torch.nn.functional as F

import utils 
import numpy as np

class Net(nn.Module):
    embed_dims = [('driverID', 24000, 16), ('weekID', 7, 3), ('timeID', 1440, 8)]

    def __init__(self):
        super(Net, self).__init__()
        # Whether to add the two ends of the path into Attribute Component
        self.build()

    def build(self):
        # Create embedding layers for each attribute
        for name, dim_in, dim_out in Net.embed_dims:
            self.add_module(name + '_em', nn.Embedding(dim_in, dim_out))

    def out_size(self):
        # Calculate the total output size from embeddings and distance
        sz = 0
        for name, dim_in, dim_out in Net.embed_dims:
            sz += dim_out
        # Append total distance
        return sz + 1

    def forward(self, attr):
        em_list = []
        # Process each attribute
        for name, dim_in, dim_out in Net.embed_dims:
            embed = getattr(self, name + '_em')
            attr_t = attr[name].view(-1, 1)  # Reshape the attribute

            attr_t = torch.squeeze(embed(attr_t), dim=1)  # Embed and squeeze

            em_list.append(attr_t)

        # Normalize the 'dist' attribute
        dist = utils.normalize(attr['dist'], 'dist')
        em_list.append(dist.view(-1, 1))  # Add distance to the list

        # Concatenate all embeddings and distance into a single tensor
        return torch.cat(em_list, dim=1)
