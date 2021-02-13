# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:14:31 2019

@author: islam
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
# Collaborative Filtering
# use embedding to build a simple recommendation system
# Source:
# 1. Collaborative filtering, https://github.com/devforfu/pytorch_playground/blob/master/movielens.ipynb
# 2. https://github.com/yanneta/pytorch-tutorials/blob/master/collaborative-filtering-nn.ipynb
# training neural network based collaborative filtering
# neural network model (NCF)
class neuralCollabFilter(nn.Module):
    def __init__(self, num_users, num_likes, embed_size, num_hidden, output_size):
        super(neuralCollabFilter, self).__init__()
        self.user_emb = nn.Embedding(num_users, embed_size)
        self.like_emb = nn.Embedding(num_likes,embed_size)
        self.fc1 = nn.Linear(embed_size*2, num_hidden[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(num_hidden[0], num_hidden[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(num_hidden[1], num_hidden[2])
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(num_hidden[2], num_hidden[3])
        self.relu4 = nn.ReLU()
        self.outLayer = nn.Linear(num_hidden[3], output_size)
        self.out_act = nn.Sigmoid()
    def forward(self, u, v):
        U = self.user_emb(u)
        V = self.like_emb(v)
        out = torch.cat([U,V], dim=1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.outLayer(out)
        out = self.out_act(out)
        return out


class matrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, emb_size):
        super(matrixFactorization, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_emb.weight.data.uniform_(0,0.05)
        self.item_emb.weight.data.uniform_(0,0.05)
        self.user_bias.weight.data.uniform_(-0.01,0.01)
        self.item_bias.weight.data.uniform_(-0.01,0.01)
        self.out_act = nn.Sigmoid()
        
    def forward(self, u, v):
        U = self.user_emb(u)
        V = self.item_emb(v)
        b_u = self.user_bias(u).squeeze()
        b_v = self.item_bias(v).squeeze()
        out = self.out_act((U*V).sum(1) +  b_u  + b_v)
        return out
