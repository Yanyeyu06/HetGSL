import numpy as np
import scipy
import torch
import torch as th
import scipy.sparse as sp

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)

def load_DBLP_data(prefix=r'./Dataset/DBLP'):
    num_classes = 4
    features_0 = scipy.sparse.load_npz(prefix + '/a_feature.npz').toarray()
    label = np.zeros((features_0.shape[0], num_classes))
    idx_train = []
    idx_val = []
    idx_test = []
    f1 = open(prefix + '/label_train.dat', 'r', encoding='utf-8')
    for line in f1.readlines():
        a, b, c, d = line.strip('\n').split('\t')
        label[int(a)][int(c)] = 1
        idx_train.append(int(a))
    f2 = open(prefix + '/label_val.dat', 'r', encoding='utf-8')
    for line in f2.readlines():
        a, b, c, d = line.strip('\n').split('\t')
        label[int(a)][int(c)] = 1
        idx_val.append(int(a))
    f3 = open(prefix + '/label_test.dat', 'r', encoding='utf-8')
    for line in f3.readlines():
        a, b, c, d = line.strip('\n').split('\t')
        label[int(a)][int(c)] = 1
        idx_test.append(int(a))
    labels = []
    for i in range(features_0.shape[0]):
        for j in range(num_classes):
            if label[i][j] != 0:
                    labels.append(j)

    apcpa = scipy.sparse.load_npz(prefix + '/apcpa_n.npz').toarray()
    apcpa_only_one = (apcpa > 0) * 1
    apcpa = torch.from_numpy(apcpa).type(torch.FloatTensor)
    apcpa_only_one = torch.from_numpy(apcpa_only_one).type(torch.FloatTensor)

    apa = scipy.sparse.load_npz(prefix + '/apa_n.npz').toarray()
    apa_only_one = (apa > 0) * 1
    apa = torch.from_numpy(apa).type(torch.FloatTensor)
    apa_only_one = torch.from_numpy(apa_only_one).type(torch.FloatTensor)

    aptpa = scipy.sparse.load_npz(prefix + '/aptpa_n.npz').toarray()
    aptpa_only_one = (aptpa > 0) * 1
    aptpa = torch.from_numpy(aptpa).type(torch.FloatTensor)
    aptpa_only_one = torch.from_numpy(aptpa_only_one).type(torch.FloatTensor)

    G = [apcpa, apa, aptpa]
    G2 = [apcpa_only_one, apa_only_one, aptpa_only_one]
    G3 = [apcpa_only_one, apa_only_one, aptpa_only_one]
    features = torch.FloatTensor(features_0)
    labels = torch.LongTensor(labels)

    apcpa_emb = np.load(prefix + '/apcpa_emb.npy')
    apa_emb = np.load(prefix + '/apa_emb.npy')
    aptpa_emb = np.load(prefix + '/aptpa_emb.npy')

    apcpa_emb = torch.from_numpy(apcpa_emb).type(torch.FloatTensor)
    apa_emb = torch.from_numpy(apa_emb).type(torch.FloatTensor)
    aptpa_emb = torch.from_numpy(aptpa_emb).type(torch.FloatTensor)
    MP_emb = [apcpa_emb, apa_emb, aptpa_emb]

    pos = sp.load_npz(prefix + '/pos_dblp.npz')
    pos = sparse_mx_to_torch_sparse_tensor(pos)

    return G, G2, G3, features, labels, num_classes, idx_train, idx_val, idx_test, MP_emb, pos



