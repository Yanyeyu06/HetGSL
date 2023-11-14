import torch
import torch.nn as nn
import torch.nn.functional as F
from model import HAN
from contrast import Contrast

def cos_sim(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

class HetGSL_HAN(nn.Module):
    def __init__(self, feat_dim, feat_hid_dim, mp_emb_dim, sema_th, metapath, dropout, num_layer, outsize):
        super(HetGSL_HAN, self).__init__()
        self.metapath = metapath
        self.non_linear = nn.ReLU()
        self.feat_mapping = nn.Linear(feat_dim, feat_hid_dim, bias=True)
        self.feat_graph_gen, self.sema_graph_gen, self.overall_graph_gen = nn.ModuleList([]), nn.ModuleList([]), nn.ModuleList([])
        for mp in self.metapath:
            self.feat_graph_gen.append(GraphGenerator(feat_hid_dim, 0))
            self.sema_graph_gen.append(GraphGenerator(mp_emb_dim, sema_th[mp]))
            self.overall_graph_gen.append(GraphChannelLayer(3))
        self.het_graph_encoder_anchor = HAN(num_meta_paths=len(metapath), in_size=feat_dim,
                                            hidden_size=feat_hid_dim, out_size=outsize,
                                            num_layers=num_layer, dropout=dropout)
        self.het_graph_encoder_leaner = HAN(num_meta_paths=len(metapath), in_size=feat_dim,
                                            hidden_size=feat_hid_dim, out_size=outsize,
                                            num_layers=num_layer, dropout=dropout)
        self.contrast = Contrast(feat_hid_dim, 0.6, 0.5)

    def forward(self, features_v1, features_v2, old_G, G2, G3, mp_emb, pos):
        feat_map = self.non_linear(self.feat_mapping(features_v2))
        new_G = []
        feat_graph_sema_final = []
        sema_graph_final = []
        for mp in self.metapath:
            old_G[mp] = F.normalize(old_G[mp], dim=1, p=2)
            new_G.append(torch.zeros_like(old_G[mp]))
            feat_graph_sema_final.append(torch.zeros_like(old_G[mp]))
            sema_graph_final.append(torch.zeros_like(old_G[mp]))
            feat_graph = self.feat_graph_gen[mp](feat_map)
            feat_graph_sema = feat_graph * G3[mp]
            feat_graph_sema = F.normalize(feat_graph_sema, dim=1, p=1)
            sema_graph = self.sema_graph_gen[mp](mp_emb[mp])
            feat_graph_sema_final[mp] = feat_graph_sema
            sema_graph_final[mp] = sema_graph
            new_G[mp] = self.overall_graph_gen[mp]([old_G[mp], feat_graph_sema, sema_graph])
            new_G[mp] = new_G[mp].t() + new_G[mp]
            new_G[mp] = F.normalize(new_G[mp], dim=1, p=1)
        logits_an, h_an = self.het_graph_encoder_anchor(G2, features_v1)
        logits_le, h_le = self.het_graph_encoder_leaner(new_G, features_v2)
        loss = self.contrast(h_an, h_le, pos)
        return G2, new_G, h_an, logits_le, h_le, loss


class MetricCalcLayer(nn.Module):
    def __init__(self, nhid):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, nhid))
        nn.init.xavier_uniform_(self.weight)
    def forward(self, h):
        return h * self.weight


class GraphGenerator(nn.Module):
    def __init__(self, dim, threshold=0.1, num_channel=2):
        super(GraphGenerator, self).__init__()
        self.threshold = threshold
        self.metric_layer = nn.ModuleList()
        self.num_channel = num_channel
        for i in range(num_channel):
            self.metric_layer.append(MetricCalcLayer(dim))
    def forward(self, h):
        s = torch.zeros((h.shape[0], h.shape[0]))
        for i in range(self.num_channel):
            weighted_left_h = self.metric_layer[i](h)
            weighted_right_h = self.metric_layer[i](h)
            s += cos_sim(weighted_left_h, weighted_right_h)
        s /= self.num_channel
        s = torch.where(s < self.threshold, torch.zeros_like(s), s)
        return s


class GraphChannelLayer(nn.Module):
    def __init__(self, num_channel):
        super(GraphChannelLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_channel, 1, 1))
        nn.init.constant_(self.weight, 0.1)

    def forward(self, adj_list):
        adj_list = torch.stack(adj_list)
        adj_list = F.normalize(adj_list, dim=1, p=2)
        return torch.sum(adj_list * F.softmax(self.weight, dim=0), dim=0)