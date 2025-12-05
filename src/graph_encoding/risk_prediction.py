import torch
import torch.nn as nn

class RiskPredictionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, mode='regression'):
        super(RiskPredictionHead, self).__init__()
        self.mode = mode
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.mode == 'regression':
            return torch.sigmoid(x)  # To keep the output between 0 and 1
        return x
