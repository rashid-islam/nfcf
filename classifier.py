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
from performance_and_fairness_measures import getHitRatio, getNDCG, differentialFairnessMultiClass, computeEDF_clf, computeAbsoluteUnfairness_clf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from collaborative_models import neuralClassifier

#%%The function below ensures that we seed all random generators with the same value to get reproducible results
def set_random_seed(state=1):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)

RANDOM_STATE = 1
set_random_seed(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% loss function for differential fairness
def criterionHinge(epsilonClass, epsilonBase):
    zeroTerm = torch.tensor(0.0).to(device)
    return torch.max(zeroTerm, (epsilonClass-epsilonBase))

#%% fine-tuning pre-trained model with user-career pairs
def fair_fine_tune_model(model,df_train, epochs, lr,batch_size,num_negatives,num_items,protectedAttributes,lamda,epsilonBase,unsqueeze=False):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    model.train()
    
    all_user_input = torch.LongTensor(df_train['user_id'].values).to(device)
    all_item_input = torch.LongTensor(df_train['like_id'].values).to(device)
    
    for i in range(epochs):
        j = 0
        for batch_i in range(0,np.int64(np.floor(len(df_train)/batch_size))*batch_size,batch_size):
            data_batch = (df_train[batch_i:(batch_i+batch_size)]).reset_index(drop=True)
            train_user_input, train_item_input, train_ratings = get_instances_with_random_neg_samples(data_batch, num_items, num_negatives,device)
            if unsqueeze:
                train_ratings = train_ratings.unsqueeze(1)
            y_hat = model(train_user_input, train_item_input)
            loss1 = criterion(y_hat, train_ratings)
            
            predicted_probs = model(all_user_input, all_item_input)
            avg_epsilon = computeEDF(protectedAttributes,predicted_probs,num_items,all_item_input,device)
            loss2 = criterionHinge(avg_epsilon, epsilonBase)
            
            loss = loss1 + lamda*loss2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('epoch: ', i, 'batch: ', j, 'out of: ',np.int64(np.floor(len(df_train)/batch_size)), 'average loss: ',loss.item())
            j = j+1
#%% model evaluation
def test_fine_tune(model,df_val,num_negatives,num_items, unsqueeze=False):
    model.eval()
    test_user_input, test_item_input, test_ratings= get_instances_with_random_neg_samples(df_val, num_items, num_negatives,device)
    if unsqueeze:
        test_ratings = test_ratings.unsqueeze(1)
    y_hat = model(test_user_input, test_item_input)
    
    predicted_ratings = (((y_hat.cpu())>0.5).numpy()).reshape((-1,))
    y_hat = y_hat.cpu().detach().numpy().reshape((-1,))
    test_ratings = test_ratings.cpu().detach().numpy().reshape((-1,))
    
    Accuracy = sum(predicted_ratings == test_ratings)/len(test_ratings)
    
    print(f"accuracy: {Accuracy: .3f}")
    aucScore = roc_auc_score(test_ratings,y_hat)
    print(f"ROC AUC: {aucScore: .3f}") 
    f1_measure = f1_score(test_ratings,predicted_ratings)
    print(f"F1 score: {f1_measure: .2f}")
    recall_measure = recall_score(test_ratings,predicted_ratings)
    print(f"Recall score: {recall_measure: .2f}")

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

num_uniqueUsers = len(train_userPages.user_id.unique())
num_uniqueLikes = len(train_userPages.like_id.unique())

# to fine tune career recommendation
num_uniqueCareers = len(train_careers.like_id.unique())

# form binary features
usr_features = np.zeros((num_uniqueUsers,num_uniqueLikes))
for i in range(len(train_userPages)):
    usr_features[train_userPages['user_id'][i],train_userPages['like_id'][i]] = 1.0

train_features = np.zeros((len(train_users),num_uniqueLikes))
test_features = np.zeros((len(test_users),num_uniqueLikes))

for i in range(len(train_users)):
    train_features[i] = usr_features[train_users['user_id'][i]]
    
for i in range(len(test_users)):
    test_features[i] = usr_features[test_users['user_id'][i]]

from sklearn.decomposition import PCA,IncrementalPCA
pca = IncrementalPCA(n_components=50,batch_size=64)
pca.fit(train_features)

train_features = pca.transform(train_features)
test_features = pca.transform(test_features)



#%% set hyperparameters
hidden_layers = np.array([train_features.shape[1], 64, 32, 16])
output_size = num_uniqueCareers
num_epochs = 200
learning_rate = 0.001
batch_size = 256 

random_samples = 15
top_K = 10


train_gender = train_protected_attributes['gender'].values
test_gender = test_protected_attributes['gender'].values


#%% clf model

trainData = torch.from_numpy(train_features)
trainLabel = torch.from_numpy(train_careers.values)                        
testData = torch.from_numpy(test_features)

clf_model = neuralClassifier(train_features.shape[1], hidden_layers,output_size)
#%% training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(clf_model.parameters(), lr=learning_rate, weight_decay=1e-6)

clf_model.train()
trainData = Variable(trainData.float())
trainLabel = Variable(trainLabel.squeeze(1).long())
for i in range(num_epochs):
    for batch_i in range(0,np.int64(np.floor(len(trainData)/batch_size))*batch_size,batch_size):
        train_batch = trainData[batch_i:batch_i+batch_size,:]
        label_batch = trainLabel[batch_i:batch_i+batch_size]
        y_hat = clf_model(train_batch)
        loss = criterion(y_hat, label_batch)    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch: ', i, 'average loss: ',loss.item())

#%% evaluate
testData = Variable(testData.float())
with torch.no_grad():
    avg_HR = np.zeros((len(test_features),top_K))
    avg_NDCG = np.zeros((len(test_features),top_K))
    
    for i in range(len(test_features)):
        y_hat = clf_model(testData[i])
#        _, predicted = torch.max(y_hat.data, 0)
        for ki in range(top_K):
            # Evaluate top rank list
            idx = torch.topk(y_hat.data, k=ki, dim=0)[1]
            ranklist = idx.tolist()
            gtItem = test_careers['like_id'][i]
            avg_HR[i,ki] = getHitRatio(ranklist, gtItem)
            avg_NDCG[i,ki] = getNDCG(ranklist, gtItem)
            
    avg_HR = np.mean(avg_HR, axis = 0)
    avg_NDCG = np.mean(avg_NDCG, axis = 0)


np.savetxt('results/avg_HR_CLF.txt',avg_HR)
np.savetxt('results/avg_NDCG_CLF.txt',avg_NDCG)

#%% evaluate fairness
import sys
sys.stdout=open("dnnClf_output.txt","w")

with torch.no_grad():    
    y_hat = clf_model(testData)

device = torch.device("cpu")

y_hat = torch.nn.functional.softmax(y_hat,dim=1)
item_input = test_careers['like_id'].values
avg_epsilon = computeEDF_clf(test_gender,y_hat,num_uniqueCareers,item_input,device)

U_abs = computeAbsoluteUnfairness_clf(test_gender,y_hat,num_uniqueCareers,item_input,device)

avg_epsilon = avg_epsilon.numpy().reshape((-1,)).item()
print(f"average differential fairness: {avg_epsilon: .3f}")

U_abs = U_abs.numpy().reshape((-1,)).item()
print(f"absolute unfairness: {U_abs: .3f}")
