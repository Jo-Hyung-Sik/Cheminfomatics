import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sigmoid
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, n_features, hidden_channels, dropout=0.3):
        super(GCN, self).__init__()
        
        self.conv1 = GCNConv(n_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear = Linear(hidden_channels, 2)
        self.dropout = dropout
        self.relu = ReLU()
        
    def forward(self, data, batch):
        x, edge_index = data.x, data.edge_index
        
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        
        x = global_mean_pool(x, batch)
        
        x = F.dropout(x, self.dropout, training = self.training)
        x = self.linear(x)
        return x