import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(nn.Linear(in_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1, bias=False))
    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self, inputs, adj):
        support = torch.spmm(inputs, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return F.elu(output + self.bias)
        else:
            return F.elu(output)

class GCN(nn.Module):
    def __init__(self, in_size, out_size, dropout):
        super(GCN, self).__init__()
        self.fc_trans = nn.Linear(in_size, out_size, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.gcn_layers = GraphConvolution(out_size, out_size)
    def forward(self, g, h):
        h = self.fc_trans(h)
        h = self.dropout(h)
        h = self.gcn_layers(h, g)
        return h

class HANLayer(nn.Module):
    def __init__(self, num_meta_paths, in_size, out_size, dropout):
        super(HANLayer, self).__init__()
        self.gcn_layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        for i in range(num_meta_paths):
            self.gcn_layers.append(GraphConvolution(in_size, out_size))
        self.semantic_attention = SemanticAttention(in_size=out_size)
        self.num_meta_paths = num_meta_paths
    def forward(self, gs, h):
        semantic_embeddings = []
        for i, g in enumerate(gs):
            semantic_embeddings.append(self.gcn_layers[i](h, g).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)
        return self.semantic_attention(semantic_embeddings)

class HAN(nn.Module):
    def __init__(self, num_meta_paths, in_size, hidden_size, out_size, num_layers, dropout):
        super(HAN, self).__init__()
        self.fc_trans = nn.Linear(in_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_meta_paths, hidden_size, hidden_size, dropout))
        for l in range(1, num_layers):
            self.layers.append(HANLayer(num_meta_paths, hidden_size ,hidden_size, dropout))
        self.predict = nn.Linear(hidden_size, out_size)
    def forward(self, g, h):
        h = self.fc_trans(h)
        h = self.dropout(h)
        for gnn in self.layers:
            h = self.dropout(h)
            h = gnn(g, h)
        return self.predict(h), h
