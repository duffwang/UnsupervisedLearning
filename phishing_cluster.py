#!/usr/bin/python3
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from cluster_func import kmeans
from cluster_func import em







data = pd.read_csv('phishing_clean.csv')

X = data.iloc[:,:-2]
y = data.iloc[:,-2]

#Splitting data into training and testing and keeping testing data aside
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

#Converting into numpy arrays
X_train = X_train.as_matrix()
X_test = X_test.as_matrix()
y_train = y_train.as_matrix()
y_test = y_test.as_matrix()


#Preprocessing the data between 0 and 1
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

means_init = np.array([X[y == i].mean(axis=0) for i in range(3)])

##############################################################################################################################
#For Expected Maximization
em(X_train, X_test, y_train, y_test,component_list = [3,4,5,6,7,8,9,10,11], num_class = 7, file_no = "phish")

#############################################################################################################################
#For KMeans
kmeans(X_train, X_test, y_train, y_test,  component_list = [3,4,5,6,7,8,9,10,11], num_class = 7, file_no = "phish")















