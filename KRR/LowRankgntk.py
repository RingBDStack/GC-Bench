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
        go through one MLP layer, for all elements
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
        sigma = torch.mm(X, X.T) + 0.0001 # for smoothness incase the diag of sigma has zero

        for mlp_layer in range(self.num_mlp_layers):
            sigma, dot_sigma, diag = self.__next_diag(sigma)
            diag_list.append(diag)

        return diag_list

    def forward(self, U_S, V_S, X_S, y_S, A_T, X_T):
        """
        U_S: (N_S, n, rank)
        V_S: (N_S, rank, n)
        X_S: (N_S, n, d)
        y_S: (N_S, c)
        A_T: (N_T, n', n')
        X_T: (N_T, n', d)

        diag_T_list: (N_T, m, n'), m is the MLP layers
        diag_S: (N_S, m, n)
        """
        N, n, rank = U_S.shape[0], U_S.shape[1], U_S.shape[2]
        N_T, n_prime = A_T.shape[0], A_T.shape[1]
        device = U_S.get_device()

        assert A_T.shape == (N_T, n_prime, n_prime), "A_T shape wrong."

        A_T = 0.0001 * torch.eye(A_T.shape[1]).expand(A_T.shape[0],-1,-1).to(device) + A_T

        diag_T_list = []
        for i in range(len(A_T)):
            diag = torch.cat(self.diag(X_T[i], A_T[i]))
            diag_T_list.append(diag)
        diag_T = torch.cat(diag_T_list).view(N_T, self.num_mlp_layers, -1)
        assert diag_T.shape == (N_T, self.num_mlp_layers, n_prime), "diag_T shape wrong."

        diag_S_list = []
        for i in range(U_S.shape[0]):
            diag = torch.cat(self.diag(X_S[i], U_S[i].mm(V_S[i])))
            diag_S_list.append(diag)
        diag_S = torch.cat(diag_S_list).view(N, self.num_mlp_layers, -1)
        assert diag_S.shape == (N, self.num_mlp_layers, n), "diag_S shape wrong."

        """
        Computing K_SS
        """
        if self.scale == 'uniform':
            scale_mat = 1.
        else:
            scale_mat = 1. / (torch.einsum('Na,Mb->NMab', U_S[i].mm(V_S[i]).sum(dim=2), U_S[i].mm(V_S[i]).sum(dim=1)))
            assert scale_mat.shape == (N, N, n, n), "scale_mat shape wrong."

        sigma = torch.einsum('Nab,Mbc->NMac', X_S, X_S.permute(0,2,1)) + 0.0001
        ntk = sigma.clone()

        for mlp_layer in range(self.num_mlp_layers):
            tmp = torch.einsum('Na,Mb->NMab', diag_S[:,mlp_layer,:], diag_S[:,mlp_layer,:]) + 0.000001
            assert tmp.shape == (N, N, n, n), "normalization matrix shape wrong."
            sigma = sigma / tmp
            sigma, dot_sigma = self.__next(sigma)
            ntk = ntk * dot_sigma + sigma
            sigma = sigma * tmp

        # ntk_clone = ntk.clone()
        VU = V_S.bmm(U_S)
        VU_clone = VU.clone()
        for _ in range(self.num_layers-2):
            VU = VU.bmm(VU_clone)
        assert VU.shape == (N, rank, rank)
        ntk = torch.einsum('Nab,NMbc,Mcd->NMad', V_S, ntk, V_S.permute(0,2,1))
        ntk = torch.einsum('Nab,NMbc,Mcd->NMad', VU, ntk, VU.permute(0,2,1))
        ntk = torch.einsum('Nab,NMbc,Mcd->NMad', U_S, ntk, U_S.permute(0,2,1))
        # ntk = ntk + 0.000001 * ntk_clone # self-loop

        assert ntk.shape == (N, N, n, n), "ntk shape wrong."
        ntk = scale_mat**(self.num_layers) * ntk
        K_SS = ntk.sum(dim=(2,3)) # (N, N)
        assert K_SS.shape == (N, N), "K_SS shape wrong."

        """
        Computing K_ST column by column
        """
        if self.scale == 'uniform':
            scale_mat = 1.
        else:
            scale_mat = 1. / (torch.einsum('Na,Mb->NMab', U_S[i].mm(V_S[i]).sum(dim=2), A_T.sum(dim=1)))
            assert scale_mat.shape == (N, N_T, n, n_prime), "K_ST scale_mat shape wrong."

        sigma = torch.einsum('Nab,Mbc->NMac', X_S, X_T.permute(0,2,1)) + 0.0001
        ntk = torch.clone(sigma)

        for mlp_layer in range(self.num_mlp_layers):
            tmp = torch.einsum('Na,Mb->NMab', diag_S[:,mlp_layer,:], diag_T[:,mlp_layer,:]) + 0.000001
            assert tmp.shape == (N, N_T, n, n_prime), "K_ST normalization matrix shape wrong."
            sigma = sigma / tmp
            sigma, dot_sigma = self.__next(sigma)
            ntk = ntk * dot_sigma + sigma
            sigma = sigma * tmp

        for layer in range(1, self.num_layers):
            ntk = torch.einsum('NMbc,Mcd->NMbd', ntk, A_T.permute(0,2,1))
        # ntk_clone = ntk.clone()
        VU = V_S.bmm(U_S)
        VU_clone = VU.clone()
        for _ in range(self.num_layers-2):
            VU = VU.bmm(VU_clone)
        assert VU.shape == (N, rank, rank)
        ntk = torch.einsum('Nab,NMbc->NMac', V_S, ntk)
        ntk = torch.einsum('Nab,NMbc->NMac', VU, ntk)
        ntk = torch.einsum('Nab,NMbc->NMac', U_S, ntk)
        # ntk = ntk + 0.000001 * ntk_clone # self-loop

        assert ntk.shape == (N, N_T, n, n_prime), "K_ST ntk shape wrong."
        ntk = scale_mat**(self.num_layers) * ntk
        K_ST = ntk.sum(dim=(2,3)) # (N, N_T)
        assert K_ST.shape == (N, N_T), "K_ST shape wrong."

        """
        Prediction
        """
        KSS_reg = K_SS + self.reg_lambda * torch.trace(K_SS) / N * torch.eye(N).to(device)
        KSS_inverse_yS = torch.linalg.solve(KSS_reg, y_S)
        pred = K_ST.permute(1,0).mm(KSS_inverse_yS)

        return pred, K_SS
