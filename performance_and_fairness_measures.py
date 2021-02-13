# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 15:34:04 2019

@author: islam
"""

import pandas as pd
import numpy as np
from numpy.random import choice
import math

import torch
import torch.nn as nn

#%% performance measures: hit rate and NDCG
def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0

#%% Fairness metrics
#differential fairness: \epsilon
def differentialFairnessMultiClass(probabilitiesOfPositive,numClasses,device):
    # input: probabilitiesOfPositive = positive p(y|S) from ML algorithm
    # output: epsilon = differential fairness measure
    epsilonPerClass = torch.zeros(len(probabilitiesOfPositive),dtype=torch.float).to(device)
    for c in  range(len(probabilitiesOfPositive)):
        epsilon = torch.tensor(0.0).to(device) # initialization of DF
        for i in  range(len(probabilitiesOfPositive[c])):
            for j in range(len(probabilitiesOfPositive[c])):
                if i == j:
                    continue
                else:
                    epsilon = torch.max(epsilon,torch.abs(torch.log(probabilitiesOfPositive[c,i])-torch.log(probabilitiesOfPositive[c,j]))) # ratio of probabilities of positive outcome
#                    epsilon = torch.max(epsilon,torch.abs((torch.log(1-probabilitiesOfPositive[c,i]))-(torch.log(1-probabilitiesOfPositive[c,j])))) # ratio of probabilities of negative outcome
        epsilonPerClass[c] = epsilon # overall DF of the algorithm 
    avg_epsilon = torch.mean(epsilonPerClass)
    return avg_epsilon

# smoothed empirical differential fairness measurement
def computeEDF(protectedAttributes,predictions,numClasses,item_input,device):
    # compute counts and probabilities
    S = np.unique(protectedAttributes) # number of gender: male = 0; female = 1
    countsClassOne = torch.zeros((numClasses,len(S)),dtype=torch.float).to(device) #each entry corresponds to an intersection, arrays sized by largest number of values 
    countsTotal = torch.zeros((numClasses,len(S)),dtype=torch.float).to(device)
    
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter/numClasses
    
    for i in range(len(predictions)):
        countsTotal[item_input[i],protectedAttributes[i]] = countsTotal[item_input[i],protectedAttributes[i]] + 1.0
        countsClassOne[item_input[i],protectedAttributes[i]] = countsClassOne[item_input[i],protectedAttributes[i]] + predictions[i]
    
    #probabilitiesClassOne = countsClassOne/countsTotal
    probabilitiesForDFSmoothed = (countsClassOne + dirichletAlpha) /(countsTotal + concentrationParameter)
    avg_epsilon = differentialFairnessMultiClass(probabilitiesForDFSmoothed,numClasses,device)           
    return avg_epsilon

def computeAbsoluteUnfairness(protectedAttributes,predictions,numClasses,item_input,device):
    # compute counts and probabilities
    S = np.unique(protectedAttributes) # number of gender: male = 0; female = 1
    scorePerGroupPerItem = torch.zeros((numClasses,len(S)),dtype=torch.float).to(device) #each entry corresponds to an intersection, arrays sized by largest number of values 
    scorePerGroup = torch.zeros(len(S),dtype=torch.float).to(device)
    countPerItem = torch.zeros((numClasses,len(S)),dtype=torch.float).to(device)
    
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter/numClasses
    
    for i in range(len(predictions)):
        scorePerGroupPerItem[item_input[i],protectedAttributes[i]] = scorePerGroupPerItem[item_input[i],protectedAttributes[i]] + predictions[i]
        countPerItem[item_input[i],protectedAttributes[i]] = countPerItem[item_input[i],protectedAttributes[i]] + 1.0
        scorePerGroup[protectedAttributes[i]] = scorePerGroup[protectedAttributes[i]] + predictions[i]
    #probabilitiesClassOne = countsClassOne/countsTotal
    avgScorePerGroupPerItem = (scorePerGroupPerItem + dirichletAlpha) /(countPerItem + concentrationParameter)
    avg_score = scorePerGroup/torch.sum(countPerItem,axis=0)  #torch.mean(avgScorePerGroupPerItem,axis=0)  
    difference = torch.abs(avgScorePerGroupPerItem - avg_score)
    U_abs = torch.mean(torch.abs(difference[:,0]-difference[:,1]))       
    return U_abs

# smoothed empirical differential fairness measurement
def computeEDF_clf(protectedAttributes,predictions,numClasses,item_input,device):
    # compute counts and probabilities
    S = np.unique(protectedAttributes) # number of gender: male = 0; female = 1
    countsClassOne = torch.zeros((numClasses,len(S)),dtype=torch.float).to(device) #each entry corresponds to an intersection, arrays sized by largest number of values 
    countsTotal = torch.zeros((numClasses,len(S)),dtype=torch.float).to(device)
    
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter/numClasses
    
    for i in range(len(predictions)):
        countsTotal[item_input[i],protectedAttributes[i]] = countsTotal[item_input[i],protectedAttributes[i]] + 1.0
        countsClassOne[item_input[i],protectedAttributes[i]] = countsClassOne[item_input[i],protectedAttributes[i]] + predictions[i,item_input[i]]
    
    #probabilitiesClassOne = countsClassOne/countsTotal
    probabilitiesForDFSmoothed = (countsClassOne + dirichletAlpha) /(countsTotal + concentrationParameter)
    avg_epsilon = differentialFairnessMultiClass(probabilitiesForDFSmoothed,numClasses,device)           
    return avg_epsilon

def computeAbsoluteUnfairness_clf(protectedAttributes,predictions,numClasses,item_input,device):
    # compute counts and probabilities
    S = np.unique(protectedAttributes) # number of gender: male = 0; female = 1
    scorePerGroupPerItem = torch.zeros((numClasses,len(S)),dtype=torch.float).to(device) #each entry corresponds to an intersection, arrays sized by largest number of values 
    scorePerGroup = torch.zeros(len(S),dtype=torch.float).to(device)
    countPerItem = torch.zeros((numClasses,len(S)),dtype=torch.float).to(device)
    
    concentrationParameter = 1.0
    dirichletAlpha = concentrationParameter/numClasses
    
    for i in range(len(predictions)):
        scorePerGroupPerItem[item_input[i],protectedAttributes[i]] = scorePerGroupPerItem[item_input[i],protectedAttributes[i]] + predictions[i,item_input[i]]
        countPerItem[item_input[i],protectedAttributes[i]] = countPerItem[item_input[i],protectedAttributes[i]] + 1.0
        scorePerGroup[protectedAttributes[i]] = scorePerGroup[protectedAttributes[i]] + 1.0
    #probabilitiesClassOne = countsClassOne/countsTotal
    avgScorePerGroupPerItem = (scorePerGroupPerItem + dirichletAlpha) /(countPerItem + concentrationParameter)
    avg_score = scorePerGroup/torch.sum(countPerItem,axis=0)  #torch.mean(avgScorePerGroupPerItem,axis=0)  
    difference = torch.abs(avgScorePerGroupPerItem - avg_score)
    U_abs = torch.mean(torch.abs(difference[:,0]-difference[:,1]))       
    return U_abs