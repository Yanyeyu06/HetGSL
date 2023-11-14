import argparse
import torch
from model_HetGSL import HetGSL_HAN
from tools import evaluate_results_nc, EarlyStopping
from data import load_DBLP_data
import numpy as np
import random
import torch.nn.functional as F
from model import HAN

def set_random_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_feat_mask(features, mask_rate):
    feat_node = features.shape[1]
    mask = torch.zeros(features.shape)
    samples = np.random.choice(feat_node, size=int(feat_node * mask_rate), replace=False)
    mask[:, samples] = 1
    return mask, samples

def main(args):
    G, G2, G3, features, labels, num_classes, train_idx, val_idx, test_idx, MP_emb, pos = load_DBLP_data()
    mp_type = list(range(len(G)))
    features = features.to(args['device'])
    labels = labels.to(args['device'])
    G = [graph.to(args['device']) for graph in G]
    G2 = [graph.to(args['device']) for graph in G2]
    G3 = [graph.to(args['device']) for graph in G3]
    labels = labels.to(args['device'])
    MP_emb = [graph.to(args['device']) for graph in MP_emb]
    pos = pos.to(args['device'])

    model = HetGSL_HAN(feat_dim=features.shape[1], feat_hid_dim=args['hidden'],
                     mp_emb_dim=MP_emb[0].shape[1], sema_th=args['sema_th'],
                     metapath=mp_type, dropout=args['dropout'],
                     num_layer=args['num_layers'], outsize=num_classes).to(args['device'])

    early_stopping = EarlyStopping(patience=args['patience'], verbose=True, save_path='checkpoint/checkpoint_{}.pt'.format(args['dataset']))
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])


    for epoch in range(args['num_epochs']):
        mask_v1, _ = get_feat_mask(features, 0)
        features_v1 = features * (1 - mask_v1).to(args['device'])

        mask_v2, _ = get_feat_mask(features, 0)
        features_v2 = features * (1 - mask_v2).to(args['device'])

        model.train()
        G22, new_G, logits, h_an, h, train_l_cont = model(features_v1, features_v2, G, G2, G3, MP_emb, pos)
        loss = train_l_cont
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch: {:d}| Train Loss: {:.4f}'.format(epoch + 1, loss.item()))
        early_stopping(loss.data.item(), model)

        if epoch % 1 == 0:
            tau = args['tau']
            for mp in mp_type:
                G2[mp] = G22[mp] * tau + new_G[mp].detach() * (1 - tau)
                G2[mp] = F.normalize(G2[mp], dim=1, p=1)
        if early_stopping.early_stop:
            break


    print('\ntesting...')
    model.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(args['dataset'])))
    model.eval()
    _, new_G, _, _, h, loss = model(features, features, G, G2, G3, MP_emb, pos)
    print('Learner View (Self-supervised) ...')
    evaluate_results_nc(h[test_idx].detach().cpu().numpy(), labels[test_idx].cpu().numpy(), int(labels.max()) + 1)
    new_G = [F.normalize(graph.detach(), dim=1, p=2) for graph in new_G]
    net = HAN(num_meta_paths=len(mp_type),
            in_size=features.shape[1],
            hidden_size=args['hidden'],
            out_size=num_classes,
            num_layers=args['num_layers'],
            dropout=args['dropout']).to(args['device'])
    early_stopping_2 = EarlyStopping(patience=args['patience'], verbose=True,
                                   save_path='checkpoint/checkpoint2_{}.pt'.format(args['dataset']))
    optimizer_2 = torch.optim.Adam(net.parameters(), lr=0.002)
    for epoch in range(args['num_epochs']):
        net.train()
        logits, h = net(new_G, features)
        train_loss = loss_fcn(logits[train_idx], labels[train_idx])
        optimizer_2.zero_grad()
        train_loss.backward()
        optimizer_2.step()
        net.eval()
        logits, h = net(new_G, features)
        val_loss = loss_fcn(logits[val_idx], labels[val_idx])
        early_stopping_2(val_loss.data.item(), net)
        if early_stopping_2.early_stop:
            break
    net.load_state_dict(torch.load('checkpoint/checkpoint2_{}.pt'.format(args['dataset'])))
    net.eval()
    logits, h = net(new_G, features)
    print('Structure Test (Semi-Supervised)...')
    evaluate_results_nc(h[test_idx].detach().cpu().numpy(), labels[test_idx].cpu().numpy(), int(labels.max()) + 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HetGSL')
    parser.add_argument('--dataset', default='DBLP')
    parser.add_argument('--lr', default=0.005)
    parser.add_argument('--sema_th', default=[0.0, 0.0, 0.01])
    parser.add_argument('--num_layers', default=1)
    parser.add_argument('--hidden', default=64)
    parser.add_argument('--dropout', default=0.5)
    parser.add_argument('--num_epochs', default=60)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--tau', type=float, default=0.95)
    args = parser.parse_args().__dict__
    set_random_seed()
    print(args)
    main(args)

