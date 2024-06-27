import math
import numpy as np
import scipy as sp
import torch
import copy
import sys

def have_nan(a):
    return torch.any(torch.isnan(a))

def have_negative(a):
    return torch.any(a<0)


class LiteNTK(torch.nn.Module):
    def __init__(self, num_layers, num_mlp_layers, scale, reg_lambda=1e-6):
        super(LiteNTK, self).__init__()
        """
        num_layers: number of layers in the neural networks (not including the input layer)
        num_mlp_layers: number of MLP layers
        scale: the scale used aggregate neighbors [uniform, degree]
        """
        self.num_layers = num_layers
        self.num_mlp_layers = num_mlp_layers
        self.scale = scale
        assert(scale in ['uniform', 'degree'])
        self.reg_lambda = reg_lambda

    def __adj(self, S, adj1, adj2):
        """
        go through one adj layer
        """
        tmp = adj1.mm(S)
        tmp = adj2.mm(tmp.transpose(0,1)).transpose(0,1)
        return tmp

    def __next_diag(self, S):
        """
        go through one normal layer, for diagonal element
        S: covariance of last layer
        """
        diag = torch.sqrt(S.diag())
        tmp = diag[:, None] * diag[None, :]
        S = S / tmp
        S = torch.clamp(S, -0.9999, 0.9999) # smooth the value so the derivative will not lead into NAN: https://discuss.pytorch.org/t/nan-gradient-for-torch-cos-torch-acos/9617
        DS = (math.pi - torch.acos(S)) / math.pi
        S = (S * (math.pi - torch.acos(S)) + torch.sqrt(1 - torch.pow(S, 2))) / math.pi
        S = S * tmp

        return S, DS, diag

    def __next(self, S):
        """
        go through one MLP layer
        """
        S = torch.clamp(S, -0.9999, 0.9999)
        DS = (math.pi - torch.acos(S)) / math.pi
        S = (S * (math.pi - torch.acos(S)) + torch.sqrt(1 - torch.pow(S, 2))) / math.pi
        return S, DS


    def diag(self, X, A):
        """
        compute the diagonal element of GNTK
        X: feature matrix
        A: adjacency matrix
        """

        if self.scale == 'uniform':
            scale_mat = 1.
        else:
            scale_mat = 1. / torch.outer(A.sum(dim=1), A.sum(dim=0))

        diag_list = []
        sigma = torch.mm(X, X.T) + 0.0001 # for smoothness, in case the diag of sigma has zero
        sigma = scale_mat * self.__adj(sigma, A, A)

        for mlp_layer in range(self.num_mlp_layers):
            sigma, dot_sigma, diag = self.__next_diag(sigma)
            diag_list.append(diag)

        return diag_list

    def forward(self, A_S, X_S, y_S, A_T, X_T):
        """
        N_S: # synthetic graphs
        n: # nodes
        d, c: # of features/classes
        A_S: (N_S, n, n)
        X_S: (N_S, n, d)
        y_S: (N_S, c)
        A_T: (N_T, n', n')
        X_T: (N_T, n', d)

        diag_T_list: (N_T, m, n'), m is the MLP layers
        diag_S: (N_S, m, n)
        """
        N, n = A_S.shape[0], A_S.shape[1]
        N_T, n_prime = A_T.shape[0], A_T.shape[1]
        device = A_S.get_device()

        assert A_S.shape == (N, n, n), "A_S shape wrong."
        assert A_T.shape == (N_T, n_prime, n_prime), "A_S shape wrong."

        A_S = 0.0001 * torch.eye(A_S.shape[1]).expand(A_S.shape[0],-1,-1).to(device) + A_S
        A_T = 0.0001 * torch.eye(A_T.shape[1]).expand(A_T.shape[0],-1,-1).to(device) + A_T

        diag_T_list = []
        for i in range(len(A_T)):
            diag = torch.cat(self.diag(X_T[i], A_T[i]))
            diag_T_list.append(diag)
        diag_T = torch.cat(diag_T_list).view(N_T, self.num_mlp_layers, -1)
        assert diag_T.shape == (N_T, self.num_mlp_layers, n_prime), "diag_T shape wrong."

        diag_S_list = []
        for i in range(A_S.shape[0]):
            diag = torch.cat(self.diag(X_S[i], A_S[i]))
            diag_S_list.append(diag)
        diag_S = torch.cat(diag_S_list).view(N, self.num_mlp_layers, -1)
        assert diag_S.shape == (N, self.num_mlp_layers, n), "diag_S shape wrong."

        """
        Computing K_SS
        """
        if self.scale == 'uniform':
            scale_mat = 1.
        else:
            scale_mat = 1. / (torch.einsum('Na,Mb->NMab', A_S.sum(dim=2), A_S.sum(dim=1)))
            assert scale_mat.shape == (N, N, n, n), "scale_mat shape wrong."

        sigma = torch.einsum('Nab,Mbc->NMac', X_S, X_S.permute(0,2,1)) + 0.0001
        sigma = torch.einsum('Nab,NMbc,Mcd->NMad', A_S, sigma, A_S.permute(0,2,1)) # batch random walk
        assert sigma.shape == (N, N, n, n), "sigma shape wrong."
        sigma = scale_mat * sigma
        ntk = torch.clone(sigma)

        for mlp_layer in range(self.num_mlp_layers):
            tmp = torch.einsum('Na,Mb->NMab', diag_S[:,mlp_layer,:], diag_S[:,mlp_layer,:]) + 0.000001
            assert tmp.shape == (N, N, n, n), "normalization matrix shape wrong."
            sigma = sigma / tmp
            sigma, dot_sigma = self.__next(sigma)
            ntk = ntk * dot_sigma + sigma
            sigma = sigma * tmp

        for layer in range(1, self.num_layers):
            ntk = torch.einsum('Nab,NMbc,Mcd->NMad', A_S, ntk, A_S.permute(0,2,1))
        assert ntk.shape == (N, N, n, n), "ntk shape wrong."
        ntk = scale_mat**(self.num_layers-1) * ntk
        K_SS = ntk.mean(dim=(2,3)) # (N, N)
        assert K_SS.shape == (N, N), "K_SS shape wrong."

        """
        Computing K_ST column by column
        """
        K_ST = torch.zeros((A_S.shape[0], A_T.shape[0])).to(device)
        if self.scale == 'uniform':
            scale_mat = 1.
        else:
            scale_mat = 1. / (torch.einsum('Na,Mb->NMab', A_S.sum(dim=2), A_T.sum(dim=1)))
            assert scale_mat.shape == (N, N_T, n, n_prime), "K_ST scale_mat shape wrong."

        sigma = torch.einsum('Nab,Mbc->NMac', X_S, X_T.permute(0,2,1)) + 0.0001
        sigma = torch.einsum('Nab,NMbc,Mcd->NMad', A_S, sigma, A_T.permute(0,2,1)) # batch random walk
        assert sigma.shape == (N, N_T, n, n_prime), "K_ST sigma shape wrong."
        sigma = scale_mat * sigma
        ntk = torch.clone(sigma)

        for mlp_layer in range(self.num_mlp_layers):
            tmp = torch.einsum('Na,Mb->NMab', diag_S[:,mlp_layer,:], diag_T[:,mlp_layer,:]) + 0.000001
            assert tmp.shape == (N, N_T, n, n_prime), "K_ST normalization matrix shape wrong."
            sigma = sigma / tmp
            sigma, dot_sigma = self.__next(sigma)
            ntk = ntk * dot_sigma + sigma
            sigma = sigma * tmp

        for layer in range(1, self.num_layers):
            ntk = torch.einsum('Nab,NMbc,Mcd->NMad', A_S, ntk, A_T.permute(0,2,1))
        assert ntk.shape == (N, N_T, n, n_prime), "K_ST ntk shape wrong."
        ntk = scale_mat**(self.num_layers-1) * ntk
        K_ST = ntk.mean(dim=(2,3)) # (N, N_T)
        assert K_ST.shape == (N, N_T), "K_ST shape wrong."

        """
        Prediction
        """
        KSS_reg = K_SS + self.reg_lambda * torch.trace(K_SS) / N * torch.eye(N).to(device)
        KSS_inverse_yS = torch.linalg.solve(KSS_reg, y_S)
        pred = K_ST.permute(1,0).mm(KSS_inverse_yS)

        return pred, K_SS