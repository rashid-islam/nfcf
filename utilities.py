# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:28:38 2019

@author: islam
"""
import pandas as pd
import numpy as np
from numpy.random import choice
import math

import torch
import torch.nn as nn
#%% Data encoding
# here is a handy function modified from fast.ai
def proc_col(col, train_col=None):
    """Encodes a pandas column with continous ids. 
    """
    if train_col is not None:
        uniq = train_col.unique()
    else:
        uniq = col.unique()
    name2idx = {o:i for i,o in enumerate(uniq)}
    return name2idx, np.array([name2idx.get(x, -1) for x in col]), len(uniq)

def encode_data(df,col_names_list, train=None):
    """ Encodes rating data with continous user and movie ids. 
    If train is provided, encodes df with the same encoding as train.
    """
    df = df.copy()
    for col_name in col_names_list:
        train_col = None
        if train is not None:
            train_col = train[col_name]
        _,col,_ = proc_col(df[col_name], train_col)
        df[col_name] = col
        df = df[df[col_name] >= 0]
    return df
#%% negative samples
def get_instances_with_neg_samples(train, probabilities, num_negatives,device):
    user_input = np.zeros((len(train)+len(train)*num_negatives))
    item_input = np.zeros((len(train)+len(train)*num_negatives))
    labels = np.zeros((len(train)+len(train)*num_negatives))
    
    neg_samples = choice(len(probabilities), size=(2*len(train)*num_negatives,), p=probabilities) # multiply by 2 to make sure, we dont run out of negative samples
    neg_counter = 0
    i = 0
    for n in range(len(train)):
        # positive instance
        user_input[i] = train['user_id'][n]
        item_input[i] = train['like_id'][n]
        labels[i] = 1
        i += 1
        # negative instances
        checkList = list(train['like_id'][train['user_id']==train['user_id'][n]])
        for t in range(num_negatives):
            j = neg_samples[neg_counter]
            while j in checkList:
                neg_counter += 1
                j = neg_samples[neg_counter]
            user_input[i] = train['user_id'][n]
            item_input[i] = j
            labels[i] = 0
            i += 1
            neg_counter += 1            
    return torch.LongTensor(user_input).to(device), torch.LongTensor(item_input).to(device), torch.FloatTensor(labels).to(device)

#%% negative samples
def get_instances_with_random_neg_samples(train,num_items, num_negatives,device):
    user_input = np.zeros((len(train)+len(train)*num_negatives))
    item_input = np.zeros((len(train)+len(train)*num_negatives))
    labels = np.zeros((len(train)+len(train)*num_negatives))
    
    neg_samples = choice(num_items, size=(10*len(train)*num_negatives,)) # multiply by 2 to make sure, we dont run out of negative samples
    neg_counter = 0
    i = 0
    for n in range(len(train)):
        # positive instance
        user_input[i] = train['user_id'][n]
        item_input[i] = train['like_id'][n]
        labels[i] = 1
        i += 1
        # negative instances
        checkList = list(train['like_id'][train['user_id']==train['user_id'][n]])
        for t in range(num_negatives):
            j = neg_samples[neg_counter]
            while j in checkList:
                neg_counter += 1
                j = neg_samples[neg_counter]
            user_input[i] = train['user_id'][n]
            item_input[i] = j
            labels[i] = 0
            i += 1
            neg_counter += 1           
    return torch.LongTensor(user_input).to(device), torch.LongTensor(item_input).to(device), torch.FloatTensor(labels).to(device)

#%% negative samples
def get_test_instances_with_random_samples(data, random_samples,num_items,device):
    user_input = np.zeros((random_samples+1))
    item_input = np.zeros((random_samples+1))
    
    # positive instance
    user_input[0] = data[0]
    item_input[0] = data[1]
    i = 1
    # negative instances
    checkList = data[1]
    for t in range(random_samples):
        j = np.random.randint(num_items)
        while j == checkList:
            j = np.random.randint(num_items)
        user_input[i] = data[0]
        item_input[i] = j
        i += 1
    return torch.LongTensor(user_input).to(device), torch.LongTensor(item_input).to(device)
