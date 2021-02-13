# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 02:13:41 2019

@author: islam
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score,f1_score,recall_score
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

#%% fine-tuning pre-trained model with user-career pairs
def fine_tune_model(model,df_train, epochs, lr,batch_size,num_negatives,num_items, unsqueeze=False):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    model.train()
    for i in range(epochs):
        j = 0
        for batch_i in range(0,np.int64(np.floor(len(df_train)/batch_size))*batch_size,batch_size):
            data_batch = (df_train[batch_i:(batch_i+batch_size)]).reset_index(drop=True)
            train_user_input, train_item_input, train_ratings = get_instances_with_random_neg_samples(data_batch, num_items, num_negatives,device)
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
def evaluate_fine_tune(model,df_val,top_K,random_samples, num_items):
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
#%%
def fairness_measures(model,df_val,num_items,protectedAttributes):
    model.eval()
    user_input = torch.LongTensor(df_val['user_id'].values).to(device)
    item_input = torch.LongTensor(df_val['like_id'].values).to(device)
    y_hat = model(user_input, item_input)
    
    avg_epsilon = computeEDF(protectedAttributes,y_hat,num_items,item_input,device)
    U_abs = computeAbsoluteUnfairness(protectedAttributes,y_hat,num_items,item_input,device)
    
    avg_epsilon = avg_epsilon.cpu().detach().numpy().reshape((-1,)).item()
    print(f"average differential fairness: {avg_epsilon: .3f}")
    
    U_abs = U_abs.cpu().detach().numpy().reshape((-1,)).item()
    print(f"absolute unfairness: {U_abs: .3f}")

#%% load data
train_users= pd.read_csv("train-test/train_usersID.csv",names=['user_id'])
test_users = pd.read_csv("train-test/test_usersID.csv",names=['user_id'])

train_careers= pd.read_csv("train-test/train_concentrationsID.csv",names=['like_id'])
test_careers = pd.read_csv("train-test/test_concentrationsID.csv",names=['like_id'])

train_protected_attributes= pd.read_csv("train-test/train_protectedAttributes.csv")
test_protected_attributes = pd.read_csv("train-test/test_protectedAttributes.csv")

# =============================================================================
# train_labels= pd.read_csv("train-test/train_labels.csv",names=['labels'])
# test_labels = pd.read_csv("train-test/test_labels.csv",names=['labels'])
# 
# unique_concentrations = (pd.concat([train_careers['like_id'],train_labels['labels']],axis=1)).reset_index(drop=True)
# unique_concentrations = unique_concentrations.drop_duplicates(subset='like_id', keep='first')
# 
# unique_careers = unique_concentrations.sort_values(by=['like_id']).reset_index(drop=True)
# unique_careers.to_csv('train-test/unique_careers.csv',index=False)
# =============================================================================
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

random_samples = 15
top_K = 10

# to load pre-train model correctly
num_uniqueUsers = len(train_userPages.user_id.unique())
num_uniqueLikes = len(train_userPages.like_id.unique())

# to fine tune career recommendation
num_uniqueCareers = len(train_data.like_id.unique())

#%% load pre-trained model

fineTune_NCF = neuralCollabFilter(num_uniqueUsers, num_uniqueLikes, emb_size, hidden_layers,output_size).to(device)

fineTune_NCF.load_state_dict(torch.load("trained-models/preTrained_NCF"))

fineTune_NCF.to(device)

#%% fine-tune to career recommendation
# replace page items with career items
fineTune_NCF.like_emb = nn.Embedding(num_uniqueCareers,emb_size).to(device)
# freeze user embedding
fineTune_NCF.user_emb.weight.requires_grad=False

fine_tune_model(fineTune_NCF,train_data, num_epochs, learning_rate,batch_size,num_negatives,num_uniqueCareers, unsqueeze=True)

torch.save(fineTune_NCF.state_dict(), "trained-models/fineTune_NCF")
#%% evaluate the fine-tune model
import sys
sys.stdout=open("fine_tune_NCF_output.txt","w")


avg_HR_fineTune, avg_NDCG_fineTune = evaluate_fine_tune(fineTune_NCF,test_data.values,top_K,random_samples, num_uniqueCareers)

np.savetxt('results/avg_HR_fineTune_NCF.txt',avg_HR_fineTune)
np.savetxt('results/avg_NDCG_fineTune_NCF.txt',avg_NDCG_fineTune)

#%% fairness measurements
train_gender = train_protected_attributes['gender'].values
test_gender = test_protected_attributes['gender'].values

fairness_measures(fineTune_NCF,test_data,num_uniqueCareers,test_gender)
