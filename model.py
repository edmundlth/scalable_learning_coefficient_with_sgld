import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MNISTNet(nn.Module):
    def __init__(
        self, hidden_layer_sizes=[1024, 1024], input_dim=28 * 28, output_dim=10, activation=F.relu, with_bias=True
    ):
        super(MNISTNet, self).__init__()
        self.input_dim = input_dim
        # use [0] to specify empty layer. 
        hidden_layer_sizes = [val for val in hidden_layer_sizes if val != 0] 
        self.layer_sizes = [input_dim] + hidden_layer_sizes + [output_dim]
        self.activation = activation
        self.with_bias = with_bias
        self.layers = nn.ModuleList()
        for i in range(len(self.layer_sizes) - 1):
            dim_in, dim_out = self.layer_sizes[i : i + 2]
            self.layers.append(nn.Linear(dim_in, dim_out, bias=self.with_bias).float())

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x


import torch
from torch.utils.data import Dataset

class CustomRegressionDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float)
        self.Y = torch.tensor(Y, dtype=torch.float)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def create_gaussian_noise_regression_dataset(f, X, noise_std):
    Y = f(X) + np.random.randn(*X.shape) * noise_std
    return CustomRegressionDataset(X, Y)