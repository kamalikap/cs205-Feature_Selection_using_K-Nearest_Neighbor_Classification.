#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 13:17:00 2018

@author: kamalika
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy import stats
import time
import math
import matplotlib.pyplot as plt

#Normalizing data by zscore normalization
def Normalize_zscore(instances):
    dataset = stats.zscore(instances)
    return dataset

#Normalizing data using min_max normalization
def Normalize_minmax(instances):    
    df = pd.DataFrame(instances)
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df)
    normalized = pd.DataFrame(np_scaled)
    return normalized
    
#plotting the normalized datasets
def plot(instances,normalized,normalized1):
    plt.figure(figsize=(8,6))
    plt.scatter(instances[0], instances[1], s=60, c='red', marker='^',label="dataset")
    plt.scatter(normalized[0],normalized[1], s=60,marker='o',label="normalized by z-score")
    plt.scatter(normalized1[0],normalized1[1], s=60,c= 'green',marker='x',label="normalized by min_max")
    plt.title('Normalizing the BIGtestdata22.txt')
    plt.xlabel('first instance')
    plt.ylabel('second instance')
    plt.legend()
    plt.show()
    return

#Performing Leave One Out Cross Validation on the normalized data
def LeaveOneOut_CV(current_features,normalized,my_feature, parameter):
    feature_set=list(current_features)
    if parameter==1:#selecting parameter to perform the desired algorithm
        feature_set.append(my_feature)
    if parameter ==2:
        feature_set.remove(my_feature)
    correct = 0
    nearest_neighbor_distance = float('inf')
    result=0
    for i in range(0,len(normalized)):
        one_out=i
        nearest_neighbor_distance = float('inf')
        for h in range(0,len(normalized)):
            if not np.array_equal(h,i):
                sum = 0
                for j in range(0, len(feature_set)):#Calculating Euclidean Distance to determine the nearest neighbors
                    sum=sum+pow((normalized[h][feature_set[j]] - normalized[one_out][feature_set[j]]),2)
                    distance=math.sqrt(sum)
                if distance < nearest_neighbor_distance:
                        nearest_neighbor_distance = distance
                        result = h
        if (normalized[result][0]==normalized[i][0]):
            correct += 1
            accuracy = (correct / (len(normalized)-1))#Calculating accuracy
    print("Testing features: ",feature_set, " with accuracy of ", accuracy*100, "%")
    return accuracy

#Selecting best features using forward selection
def ForwardSelection(normalized,num_features):
    print("." * 100)
    current_set_of_features = []
    global_acc = 0.0
    best_feature_set=[]
    print("." * 100)
    start_time = time.time()
    for i in range(1, num_features+1):
        print("\n On level %d of the search tree" % (i),"contains", current_set_of_features)
        feature_to_add = 0
        local_acc=0.0
        for j in range(1, num_features+1):
            if j not in current_set_of_features:
                accuracy = LeaveOneOut_CV(current_set_of_features,normalized,j,1)
                if accuracy > local_acc:
                    local_acc = accuracy
                    feature_to_add = j
        current_set_of_features.append(feature_to_add)
        print("\n On level %d of the search tree," % (i),"feature %d was added to the current set" % (feature_to_add))
        print("\n With ", len(current_set_of_features), " features, the accuracy is: ", local_acc * 100, "%")
        if local_acc >= global_acc: # check for decrease in accuracy
            global_acc= local_acc
            best_feature_set = list(current_set_of_features)
    end_time = time.time()
    print("." * 100)
    print("Best set of features to use: ", best_feature_set,"with accuracy", global_acc * 100)
    print("Time Elapsed",end_time - start_time,"seconds")
    return

#Selecting best features after backward elimination
def BackwardElimination(normalized,num_features):
    print("." * 100)
    global_acc = 0
    best_feature_set=[]
    current_set_of_features = [i for i in range(1, num_features+1)]
    print("." * 100)
    start_time = time.time()
    for i in range(1, num_features):
        print("\n On level %d of the search tree" % (i),"contains", current_set_of_features)
        feature_to_remove = 0
        local_acc = 0.0
        for j in range(1,num_features):
            if (j in current_set_of_features):
                accuracy = LeaveOneOut_CV(current_set_of_features,normalized,j,2)
                if accuracy > local_acc:
                    local_acc = accuracy
                    feature_to_remove = j
        if feature_to_remove in current_set_of_features: 
            current_set_of_features.remove(feature_to_remove) # removes feature selected by inner for loop
            print("\n On level ", i, " feature ", feature_to_remove, " was removed from the current set")
            print("\n With ", len(current_set_of_features), " features, the accuracy is: ", local_acc * 100, "%")
        if local_acc >= global_acc: # check for decrease in accuracy
            global_acc = local_acc
            best_feature_set= list(current_set_of_features)
    end_time = time.time()
    print("." * 100)
    print("Best set of features to use:", best_feature_set,"with accuracy", global_acc * 100, "%")
    print("Time Elapsed",end_time - start_time,"seconds")


#This search algorithm works for both forward and backward trimming of data when 
#the accuracy of level decreases from it's previous level then it stops the search and exit the loop. 
def SearchAlgorithm(normalized,num_features):
    print("." * 100)
    global_acc = 0
    best_feature_set=[]
    print("." * 100)
    feature_to_remove = 0
    feature_to_add = 0
    
    curr1 = []
    curr = [i for i in range(1, num_features+1)]
    prun=input("""2
               Enter the type of trimming needs to be done: 
                \n 1. Trimming of features using Forward selection . 
                \n 2. Trimming of features using Backward elimination \n
                """)
    if (prun=="1"):
         parameter=1
         feature_to_prun=feature_to_add
         current_set_of_features=curr1
         features=len(normalized[0])
    if (prun=="2"):
         parameter=2
         feature_to_prun=feature_to_remove
         current_set_of_features=curr
         features=num_features
    start_time = time.time()
    for i in range(1, features):
        print("\n On level %d of the search tree" % (i),"contains", current_set_of_features)
        local_acc = 0.0
        for j in range(1,features):
            if (prun=="1"):
                if (j not in current_set_of_features):
                    accuracy = LeaveOneOut_CV(current_set_of_features,normalized,j,parameter)
                    if accuracy > local_acc:
                        local_acc = accuracy
                        feature_to_prun = j
            if (prun=="2"):
                if (j in current_set_of_features):
                    accuracy = LeaveOneOut_CV(current_set_of_features,normalized,j,parameter)
                    if accuracy > local_acc:
                        local_acc = accuracy
                        feature_to_prun = j
        if local_acc <= global_acc:
            if j == num_features: # checks if addition of any feature results in increase in accuracy, if not then it breaks the loop
                break
        if local_acc > global_acc: # check for decrease in accuracy
            if prun=="1":
                current_set_of_features.append(feature_to_prun)# adds feature selected by inner for loop
            if prun=="2":
                current_set_of_features.remove(feature_to_prun) # remove feature selected by inner for loop
            global_acc = local_acc
            best_feature_set= list(current_set_of_features)
    end_time = time.time()
    print("." * 100)
    print("Best set of features to use:", best_feature_set,"with accuracy", local_acc * 100, "%")
    print("Time Elapsed",end_time - start_time,"seconds")
         
def main():
    print("Welcome to CS 205 Feature Selection Algorithm")
    file_name = input("Enter the name of the file to test: ")
    instances=np.loadtxt(file_name)
    num_instances=len(instances)
    
    print("\n *** Normalizing data... ***")
    normalized = Normalize_zscore(instances)
    normalized1 = Normalize_minmax(instances)
    print (normalized)
    print("\n The data is normalized.")
    num_features= len(normalized[0])-1
    print ("\nThis dataset has "+ str(num_features)+ " features (without class attribute), with "+str(num_instances)+ " instances")
    plot(instances,normalized,normalized1)
    
    algo=input("Type the algorithm you want to run:\n \n 1.FS-Forward Selection\n 2.BE-Backward Elimination\n 3.MS- My search algorithm \n \n \n")
    print ("Performing "+ str(algo)+ " ..........")

    if (algo == "1"):
        ForwardSelection(normalized,num_features)
    elif (algo == "2"):
        BackwardElimination(normalized,num_features)
    else:
        SearchAlgorithm(normalized,num_features)
    
if __name__ == '__main__':
     main()
