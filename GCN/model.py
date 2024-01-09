import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Softmax
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm

class GCN(torch.nn.Module):
    def __init__(self, n_features, hidden_channels, dropout=0.5):
        super(GCN, self).__init__()
        
        self.conv1 = GCNConv(n_features, hidden_channels, add_self_loops=False)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
        
        self.fc1 = Linear(hidden_channels, hidden_channels//2)
        self.fc2 = Linear(hidden_channels//2, hidden_channels//2)
        
        self.linear = Linear(hidden_channels//2, 2)
        self.dropout = dropout
        self.relu = ReLU()
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, training = self.training)
        x = self.relu(self.conv2(x, edge_index))
        x = F.dropout(x, self.dropout, training = self.training)
        x = self.conv2(x, edge_index)
        
        x = global_mean_pool(x, batch)
        
        x = self.relu(self.fc1(x))
        x = F.dropout(x, self.dropout, training = self.training)
        # x = self.relu(self.fc2(x))
        # x = F.dropout(x, self.dropout, training = self.training)
        x = self.linear(x)
        x = F.softmax(x, dim=1)
        return x