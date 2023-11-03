# -*- coding:utf-8 -*-
"""
Function: TransD model.
Input: Non.
Output: Non.
Author: Qing TANG
"""
import torch
import torch.nn as nn
import torch.nn.functional as f


class TransD(nn.Module):
    def __init__(self, entity_num, relation_num, entity_dim=128, relation_dim=128, margin=1, batch_size=128):
        super(TransD, self).__init__()
        self.ent_num = entity_num
        self.rel_num = relation_num
        self.ent_dim = entity_dim
        self.rel_dim = relation_dim
        self.margin = margin
        self.batch_size = batch_size

        # This is not creating ent_num embeddings, but creating a container that can hold ent_num embeddings.
        self.ent_embeddings = nn.Embedding(self.ent_num, self.ent_dim)
        self.rel_embeddings = nn.Embedding(self.rel_num, self.rel_dim)
        self.ent_projections = nn.Embedding(self.ent_num, self.ent_dim)
        self.rel_projections = nn.Embedding(self.rel_num, self.rel_dim)

        nn.init.xavier_normal_(self.ent_embeddings.weight.data)
        nn.init.xavier_normal_(self.rel_embeddings.weight.data)
        nn.init.xavier_normal_(self.ent_projections.weight.data)
        nn.init.xavier_normal_(self.rel_projections.weight.data)

    def embedding(self, _h, _r, _t):
        h = self.ent_embeddings(_h)
        r = self.rel_embeddings(_r)
        t = self.ent_embeddings(_t)

        h_p = self.ent_projections(_h)
        r_p = self.rel_projections(_r)
        t_p = self.ent_projections(_t)

        # Constructing mapping matrices
        r_p = r_p.reshape(self.batch_size, self.rel_dim, 1)
        h_p = h_p.reshape(self.batch_size, 1, self.ent_dim)
        t_p = t_p.reshape(self.batch_size, 1, self.ent_dim)

        eye = torch.eye(self.rel_dim, self.ent_dim)  # Constructing I^m*n

        M_rh = torch.bmm(r_p, h_p) + eye.repeat(self.batch_size, 1, 1)  # This is the peculiarity of pytorch tensor multiplication
        M_rt = torch.bmm(r_p, t_p) + eye.repeat(self.batch_size, 1, 1)

        # print("matrix shape", M_rh.shape)

        # print("recontructed ent", (h.reshape(self.batch_size, self.ent_dim, 1)).shape)

        # Project entities to hyperplane, vectors projection
        h = h.reshape(self.batch_size, self.ent_dim, 1)
        t = t.reshape(self.batch_size, self.ent_dim, 1)
        h_ = torch.bmm(M_rh, h)
        t_ = torch.bmm(M_rt, t)

        h_ = torch.squeeze(h_)
        t_ = torch.squeeze(t_)
        r = torch.squeeze(r)

        # h_ = f.normalize(h_, p=2, dim=1)
        # r = f.normalize(r, p=2, dim=1)
        # t_ = f.normalize(t_, p=2, dim=1)
        return h_, r, t_

    def score(self, h_, r, t_):
        h_ = f.normalize(h_, p=2, dim=1)
        r = f.normalize(r, p=2, dim=1)
        t_ = f.normalize(t_, p=2, dim=1)

        score = h_ + r - t_

        return torch.norm(score, p=2, dim=1).flatten()

    def hinge_loss(self, positive_loss, negative_loss):
        loss = positive_loss - negative_loss + self.margin
        return torch.maximum(loss, torch.tensor(0))

    def forward(self, X, Y):
        # X = golden_triple_batch, Y = negative_tripe_batch
        h = X[:, 0]
        r = X[:, 1]
        t = X[:, 2]

        h_0 = Y[:, 0]
        r_0 = Y[:, 1]
        t_0 = Y[:, 2]

        p_h_, p_r, p_t_ = self.embedding(h, r, t)
        n_h_, n_r, n_t_ = self.embedding(h_0, r_0, t_0)

        p_loss = self.score(p_h_, p_r, p_t_)
        n_loss = self.score(n_h_, n_r, n_t_)

        total_loss = self.hinge_loss(p_loss, n_loss)

        return total_loss
