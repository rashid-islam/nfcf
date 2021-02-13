# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 17:47:57 2019

@author: islam
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score,f1_score,recall_score
from sklearn.decomposition import PCA
import pickle

import matplotlib
import matplotlib.pyplot as plt

import heapq # for retrieval topK

from utilities import get_instances_with_random_neg_samples, get_test_instances_with_random_samples
from performance_and_fairness_measures import getHitRatio, getNDCG, differentialFairnessMultiClass, computeEDF, computeAbsoluteUnfairness

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from collaborative_models import neuralCollabFilter

#%%The function below ensures that we seed all random generators with the same value to get reproducible results
def set_random_seed(state=1):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)

RANDOM_STATE = 1
set_random_seed(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% compute bias direction & linear projection
def compute_gender_direction(data, S, user_vectors):
    genderEmbed = np.zeros((2,user_vectors.shape[1]))
    # S = 0 indicates male and S = 1 indicates female
    num_users_per_group = np.zeros((2,1))
    for i in range(len(data)):
        u = data['user_id'][i]
        if S['gender'][i] == 0:
            genderEmbed[0] +=  user_vectors[u]
            num_users_per_group[0] += 1.0
        else:
            genderEmbed[1] +=  user_vectors[u] 
            num_users_per_group[1] += 1.0
    
    genderEmbed = genderEmbed / num_users_per_group # average gender embedding
    return genderEmbed

def compute_bias_direction(gender_vectors):
    vBias= gender_vectors[1].reshape((1,-1))-gender_vectors[0].reshape((1,-1))
    vBias = vBias / np.linalg.norm(vBias,axis=1,keepdims=1)
    return vBias

def linear_projection(data,user_vectors,vBias):
    # linear projection: u - <u,v_b>v_b
    for i in range(len(data)):
        u = data['user_id'][i]
        user_vectors[u] = user_vectors[u] - (np.inner(user_vectors[u].reshape(1,-1),vBias)[0][0])*vBias
    return user_vectors
#%% load data
train_users= pd.read_csv("train-test/train_usersID.csv",names=['user_id'])
test_users = pd.read_csv("train-test/test_usersID.csv",names=['user_id'])

train_careers= pd.read_csv("train-test/train_concentrationsID.csv",names=['like_id'])
test_careers = pd.read_csv("train-test/test_concentrationsID.csv",names=['like_id'])

train_protected_attributes= pd.read_csv("train-test/train_protectedAttributes.csv")
test_protected_attributes = pd.read_csv("train-test/test_protectedAttributes.csv")

unique_careers= pd.read_csv("train-test/unique_careers.csv")
train_userPages = pd.read_csv("train-test/train_userPages.csv")

train_data = (pd.concat([train_users['user_id'],train_careers['like_id']],axis=1)).reset_index(drop=True)
test_data = (pd.concat([test_users['user_id'],test_careers['like_id']],axis=1)).reset_index(drop=True)

#%% set hyperparameters
emb_size = 128
hidden_layers = np.array([emb_size, 64, 32, 16])
output_size = 1
num_epochs = 10
learning_rate = 0.001
batch_size = 256 
num_negatives = 5

random_samples = 100
top_K = 25

# to load pre-train model correctly
num_uniqueUsers = len(train_userPages.user_id.unique())
num_uniqueLikes = len(train_userPages.like_id.unique())

# to fine tune career recommendation
num_uniqueCareers = len(train_data.like_id.unique())

#%% load pre-trained model
debiased_NCF = neuralCollabFilter(num_uniqueUsers, num_uniqueLikes, emb_size, hidden_layers,output_size).to(device)
debiased_NCF.load_state_dict(torch.load("trained-models/preTrained_NCF"))
debiased_NCF.to(device)
users_embed = debiased_NCF.user_emb.weight.data.cpu().detach().numpy()
users_embed = users_embed.astype('float')
np.savetxt('results/users_embed.txt',users_embed)

#%% compute bias direction on training users and debias user embeds using linear projection
gender_embed = compute_gender_direction(train_data, train_protected_attributes, users_embed)
np.savetxt('results/gender_embed.txt',gender_embed)

vBias = compute_bias_direction(gender_embed)
np.savetxt('results/vBias.txt',vBias)

# incorporate all users: debias train & test both
all_data = (pd.concat([train_data,test_data],axis=0)).reset_index(drop=True)
debias_users_embed = linear_projection(all_data,users_embed,vBias) # first debias training users
#debias_users_embed = linear_projection(test_data,debias_users_embed,vBias) # then debias test users
np.savetxt('results/debias_users_embed.txt',debias_users_embed)

