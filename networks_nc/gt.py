import sys
import os
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
import torch.autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from copy import deepcopy
import torch
from deeprobust.graph import utils
from torch.nn import Linear
from utils.utils import *
from torch_sparse import SparseTensor


class TransformerConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = True,
        dropout: float = 0.,
        bias: bool = True,
        root_weight: bool = True
    ):
        super(TransformerConv,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout

        self.lin_key = Linear(in_channels, heads * out_channels)
        self.lin_query = Linear(in_channels, heads * out_channels)
        self.lin_value = Linear(in_channels, heads * out_channels)

        if concat:
            self.lin_skip = Linear(in_channels, heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels, out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(self, x, adj):
        H, C = self.heads, self.out_channels
        query = self.lin_query(x).view(-1, H, C)
        key = self.lin_key(x).view(-1, H, C)
        value = self.lin_value(x).view(-1, H, C)
        out = self.mypropagate(adj=adj, out_channels=self.out_channels, query=query, key=key, value=value)
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x)
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r

        return out


    def mypropagate(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, adj: torch.Tensor, out_channels: int, dropout=0,training=True):
        N, H, C = query.shape
        query = query.permute(1, 0, 2)  # [H, N, C]
        key = key.permute(1, 2, 0)  # [H, C, N]
        attn = torch.matmul(query,key) # [H, N, N]
        attn /= math.sqrt(out_channels)

        if isinstance(adj,SparseTensor):
            square_adj = SparseTensor(
                row=adj.coo()[0], 
                col=adj.coo()[1], 
                value=adj.storage.value(), 
                sparse_sizes=(N, N)
                )
            adj = square_adj.to_dense()
        elif adj.is_sparse:
            adj = adj.to_dense()
        out = attn.masked_fill((adj.unsqueeze(0)) == 0,float('-inf'))

        # prevent nan 
        max_vals = torch.max(out,dim=-1,keepdim=True)[0]
        all_inf_mask = (max_vals == float('-inf'))
        out = torch.where(all_inf_mask, torch.zeros_like(out), F.softmax(out,dim=-1))

        self._alpha = out
        out = F.dropout(out, p=dropout, training=training)
        value = value.permute(1, 0, 2) # [H, N, C]
        out = torch.matmul(out, value)  # [H, N, C]
        out = out.permute(1, 2, 0)  # [N, C, H]
        out = out.reshape(N, -1)
        
        return out


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class GT(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers,lr=0.01,weight_decay=5e-4,
                 dropout=0.5, beta=True, heads=1, device=None):
        """
        Params:
        - nfeat: The dimension of input features for each node.
        - nhid: The size of the hidden layers.
        - nclass: The dimension of the output features (often equal to the
            number of classes in a classification task).
        - nlayers: The number of layer blocks in the model.
        - dropout: The dropout rate for regularization. It is used to prevent
            overfitting, helping the learning process remains generalized.
        - beta: A boolean parameter indicating whether to use a gated residual
            connection (based on equations 5 and 6 from the GT paper). The
            gated residual connection (controlled by the beta parameter) helps
            preventing overfitting by allowing the model to balance between new
            and existing node features across layers.
        - heads: The number of heads in the multi-head attention mechanism.
        """
        super(GT, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.nlayers = nlayers
        conv_layers = [TransformerConv(nfeat, nhid//heads, heads=heads, beta=beta)]
        conv_layers += [TransformerConv(nhid, nhid//heads, heads=heads, beta=beta) for _ in range(nlayers - 2)]
        conv_layers.append(TransformerConv(nhid, nclass, beta=beta, concat=True))
        self.convs = torch.nn.ModuleList(conv_layers)

        norm_layers = [torch.nn.LayerNorm(nhid) for _ in range(nlayers - 1)]
        self.norms = torch.nn.ModuleList(norm_layers)
        self.dropout = dropout
        self.loss = F.nll_loss

    def initialize(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, x, adj, get_embedding=False):
        if type(adj) is not torch.Tensor:
            x, adj = utils.to_tensor(x, adj, device='cuda:0')
        for i in range(self.nlayers - 1):
            x = self.convs[i](x, adj)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p = self.dropout, training = self.training)
        x_out = x

        x = self.convs[-1](x, adj)
        if get_embedding:
            return x_out, F.log_softmax(x,dim=1)
        else:
            return F.log_softmax(x,dim=1)

    def forward_sampler(self, x, adjs):
        # for ix, layer in enumerate(self.layers):
        for ix, (adj, _, size) in enumerate(adjs):
            x = self.convs[ix](x, adj)
            if ix != len(adjs) - 1:
                x = self.norms[ix](x)
                x = F.relu(x)
                x = F.dropout(x, p = self.dropout, training = self.training)
        return F.log_softmax(x, dim=1)

    def fit_with_val(self, features, adj, labels, data, train_iters=200, initialize=True, verbose=False, normalize=True, patience=None, noval=False, idx_train=None, **kwargs):
        '''data: full data class'''
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        if 'feat_norm' in kwargs and kwargs['feat_norm']:
            from utils.utils import row_normalize_tensor
            features = row_normalize_tensor(features-features.min())

        self.adj_norm = adj_norm
        self.features = features

        if len(labels.shape) > 1:
            self.multi_label = True
            self.loss = torch.nn.BCELoss()
        else:
            self.multi_label = False
            self.loss = F.nll_loss

        labels = labels.float() if self.multi_label else labels
        self.labels = labels

        if noval:
            self._train_with_val(labels, data, train_iters, verbose, adj_val=True, idx_train=idx_train)
        else:
            self._train_with_val(labels, data, train_iters, verbose, idx_train=idx_train)

    def _train_with_val(self, labels, data, train_iters, verbose, adj_val=False, idx_train=None):
        if adj_val:
            feat_full, adj_full = data.feat_val, data.adj_val
        else:
            feat_full, adj_full = data.feat_full, data.adj_full
        feat_full, adj_full = utils.to_tensor(feat_full, adj_full, device=self.device)
        adj_full_norm = utils.normalize_adj_tensor(adj_full, sparse=True)
        labels_val = torch.LongTensor(data.labels_val).to(self.device)

        if verbose:
            print('=== training gt model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_acc_val = 0

        for i in range(train_iters):
            if i == train_iters // 2:
                lr = self.lr*0.1
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)

            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            if idx_train is not None:
                loss_train = self.loss(output[idx_train], labels[idx_train])
            else:
                loss_train = self.loss(output, labels)
            loss_train.backward()
            optimizer.step()

            if verbose and i % 100 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            with torch.no_grad():
                self.eval()
                output = self.forward(feat_full, adj_full_norm)

                if adj_val:
                    loss_val = F.nll_loss(output, labels_val)
                    acc_val = utils.accuracy(output, labels_val)
                else:
                    loss_val = F.nll_loss(output[data.idx_val], labels_val)
                    acc_val = utils.accuracy(output[data.idx_val], labels_val)

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    self.output = output
                    weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)


    def test(self, idx_test):
        """Evaluate GT performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.predict()
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()


    @torch.no_grad()
    def predict(self, features=None, adj=None):
        """By default, the inputs should be unnormalized adjacency
        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCN
        """

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)
