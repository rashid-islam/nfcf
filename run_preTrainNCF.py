# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:25:15 2019

@author: islam
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score,f1_score

from utilities import get_instances_with_random_neg_samples, get_test_instances_with_random_samples
from performance_and_fairness_measures import getHitRatio, getNDCG, differentialFairnessMultiClass, computeEDF, computeAbsoluteUnfairness

import math
import heapq # for retrieval topK

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

#%% pre-training NCF model with user-page pairs
def train_epochs(model,df_train, epochs, lr, batch_size,num_negatives, unsqueeze=False):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    model.train()
    for i in range(epochs):
        j = 0
        for batch_i in range(0,np.int64(np.floor(len(df_train)/batch_size))*batch_size,batch_size):
            data_batch = (df_train[batch_i:(batch_i+batch_size)]).reset_index(drop=True)
            #train_user_input, train_item_input, train_ratings = get_instances_with_neg_samples(data_batch, probabilities, num_negatives,device)
            train_user_input, train_item_input, train_ratings = get_instances_with_random_neg_samples(data_batch, num_uniqueLikes, num_negatives,device)
            if unsqueeze:
                train_ratings = train_ratings.unsqueeze(1)
            y_hat = model(train_user_input, train_item_input)
            loss = criterion(y_hat, train_ratings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('epoch: ', i, 'batch: ', j, 'out of: ',np.int64(np.floor(len(df_train)/batch_size)), 'average loss: ',loss.item())
            j = j+1


#%% model evaluation: hit rate and NDCG
def evaluate_model(model,df_val,top_K,random_samples, num_items):
    model.eval()
    avg_HR = np.zeros((len(df_val),top_K))
    avg_NDCG = np.zeros((len(df_val),top_K))
    
    for i in range(len(df_val)):
        test_user_input, test_item_input = get_test_instances_with_random_samples(df_val[i], random_samples,num_items,device)
        y_hat = model(test_user_input, test_item_input)
        y_hat = y_hat.cpu().detach().numpy().reshape((-1,))
        test_item_input = test_item_input.cpu().detach().numpy().reshape((-1,))
        map_item_score = {}
        for j in range(len(y_hat)):
            map_item_score[test_item_input[j]] = y_hat[j]
        for k in range(top_K):
            # Evaluate top rank list
            ranklist = heapq.nlargest(k, map_item_score, key=map_item_score.get)
            gtItem = test_item_input[0]
            avg_HR[i,k] = getHitRatio(ranklist, gtItem)
            avg_NDCG[i,k] = getNDCG(ranklist, gtItem)
    avg_HR = np.mean(avg_HR, axis = 0)
    avg_NDCG = np.mean(avg_NDCG, axis = 0)
    return avg_HR, avg_NDCG 
#%% load data
train_data = pd.read_csv("train-test/train_userPages.csv")
test_data = pd.read_csv("train-test/test_userPages.csv")


#probabilities = np.loadtxt('train-test/distributionLikes.txt') # frequency distribution of likes

#%% set hyperparameters
emb_size = 128
hidden_layers = np.array([emb_size, 64, 32, 16])
output_size = 1
num_epochs = 25
learning_rate = 0.001
batch_size = 2048 
num_negatives = 5

random_samples = 100
top_K = 10

num_uniqueUsers = len(train_data.user_id.unique())
num_uniqueLikes = len(train_data.like_id.unique())

#%% start training the NCF model

preTrained_NCF = neuralCollabFilter(num_uniqueUsers, num_uniqueLikes, emb_size, hidden_layers,output_size).to(device)

train_epochs(preTrained_NCF,train_data,num_epochs,learning_rate,batch_size,num_negatives,unsqueeze=True)

torch.save(preTrained_NCF.state_dict(), "trained-models/preTrained_NCF")

#%% evaluate the model

avg_HR_preTrain, avg_NDCG_preTrain = evaluate_model(preTrained_NCF,test_data.values,top_K,random_samples, num_uniqueLikes)

np.savetxt('results/avg_HR_preTrain.txt',avg_HR_preTrain)
np.savetxt('results/avg_NDCG_preTrain.txt',avg_NDCG_preTrain)

#sys.stdout.close()