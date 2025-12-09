import torch
import torch.nn as nn
from torch.utils.data import Dataset

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
    
class EmbeddingRiskDataset(Dataset):
    def __init__(self, embeddings, risk_scores):
        self.embeddings = embeddings
        self.risk_scores = risk_scores
        self.episode_ids = sorted(list(self.embeddings.keys()))

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, idx):
        episode_id = self.episode_ids[idx]
        embedding = self.embeddings[episode_id]
        risk_score = self.risk_scores.get(episode_id, 0.0)
        return embedding, torch.tensor(risk_score, dtype=torch.float)
