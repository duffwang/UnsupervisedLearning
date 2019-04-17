#!/usr/bin/python3
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import scipy
from sklearn import random_projection
from cluster_func import em
from cluster_func import kmeans





data = pd.read_csv('winequality-data.csv')

X = data.iloc[:,:-2]
y = data.iloc[:,-2]
y = y > 6

#Splitting data into training and testing and keeping testing data aside
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

#Converting into numpy arrays
X_train = X_train.as_matrix()
X_test = X_test.as_matrix()
y_train = y_train.as_matrix()
y_test = y_test.as_matrix()

# Dimensionality reduction ICA
# kurtosis calculation

print("Starting ICA")
print("Dimensionality reduction")


def _calculate(X, ica_, n_components):
    components = ica_.components_
    ica_.components_ = components[:n_components]

    transformed = ica_.transform(X)
    ica_.components_ = components

    kurtosis = scipy.stats.kurtosis(transformed)

    return sorted(kurtosis, reverse=True)


decisiontree = DecisionTreeClassifier(criterion='gini', max_depth=15, min_samples_split=5)
ica = FastICA()

pipe = Pipeline(steps=[('ica', ica), ('decisionTree', decisiontree)])

# Plot the ICA spectrum
ica.fit(X)

fig, ax = plt.subplots()
ax.bar(list(range(1, 12)), _calculate(X, ica, 11), linewidth=2, color='blue')
plt.axis('tight')
plt.xlabel('n_components')
ax.set_ylabel('kurtosis')

# Checking the accuracy for taking all combination of components
n_components = range(1, 12)
# Parameters of pipelines can be set using ‘__’ separated parameter names:
gridSearch = GridSearchCV(pipe, dict(ica__n_components=n_components), cv=3)
gridSearch.fit(X, y)
results = gridSearch.cv_results_
ax1 = ax.twinx()

# Plotting the accuracies and best component
ax1.plot(results['mean_test_score'], linewidth=2, color='red')
ax1.set_ylabel('Mean Cross Validation Accuracy')
ax1.axvline(gridSearch.best_estimator_.named_steps['ica'].n_components, linestyle=':', label='n_components chosen',
            linewidth=2)

plt.legend(prop=dict(size=11))
plt.title(
    'Accuracy/kurtosis for ICA (best n_components=  %d)' % gridSearch.best_estimator_.named_steps['ica'].n_components)
plt.savefig("wine_ica_1")
#plt.show()

# Reducing the dimensions with optimal number of components
ica_new = FastICA(n_components=gridSearch.best_estimator_.named_steps['ica'].n_components)
ica_new.fit(X_train)
X_train_transformed = ica_new.transform(X_train)
X_test_transformed = ica_new.transform(X_test)

###############################################################################################################################
# Reconstruction Error

print("Calculating Reconstruction Error")

reconstruction_error = []
for comp in n_components:

    ica = FastICA(n_components=comp)
    X_transformed = ica.fit_transform(X_train)
    X_projected = ica.inverse_transform(X_transformed)
    reconstruction_error.append(((X_train - X_projected) ** 2).mean())

    if (comp == gridSearch.best_estimator_.named_steps['ica'].n_components):
        chosen_error = ((X_train - X_projected) ** 2).mean()

fig2, ax2 = plt.subplots()
ax2.plot(n_components, reconstruction_error, linewidth=2)
ax2.axvline(gridSearch.best_estimator_.named_steps['ica'].n_components, linestyle=':', label='n_components chosen',
            linewidth=2)
plt.axis('tight')
plt.xlabel('Number of components')
plt.ylabel('Reconstruction Error')
plt.title('Reconstruction error for n_components chosen %f ' % chosen_error)
plt.savefig("wine_ica_2")
#plt.show()

################################################################################################################################
# Dimensionally reduce the full dataset
# Reducing the dimensions with optimal number of components
ica_new = FastICA(n_components=gridSearch.best_estimator_.named_steps['ica'].n_components)
ica_new.fit(X)
X_transformed_f = ica_new.transform(X)

# Clustering after dimensionality reduction
print("Clustering ICA")

# clustering experiments
print("Expected Maximization")
component_list, array_aic, array_bic, array_homo_1, array_comp_1, array_sil_1, array_avg_log = em(X_train_transformed,
                                                                                                  X_test_transformed,
                                                                                                  y_train, y_test,

                                                                                                  component_list=[3, 4,
                                                                                                                  5, 6,
                                                                                                                  7, 8,
                                                                                                                  9, 10,
                                                                                                                  11],
                                                                                                  num_class=7, toshow=0)

print("KMeans")
component_list, array_homo_2, array_comp_2, array_sil_2, array_var = kmeans(X_train_transformed, X_test_transformed,
                                                                            y_train, y_test,
                                                                            component_list=[3, 4, 5, 6, 7, 8, 9, 10,
                                                                                            11], num_class=7, toshow=0)

# Writing data to file
component_list = np.array(component_list).reshape(-1, 1)
array_aic = np.array(array_aic).reshape(-1, 1)
array_bic = np.array(array_bic).reshape(-1, 1)
array_homo_1 = np.array(array_homo_1).reshape(-1, 1)
array_comp_1 = np.array(array_comp_1).reshape(-1, 1)
array_sil_1 = np.array(array_sil_1).reshape(-1, 1)
array_avg_log = np.array(array_avg_log).reshape(-1, 1)
array_homo_2 = np.array(array_homo_2).reshape(-1, 1)
array_comp_2 = np.array(array_comp_2).reshape(-1, 1)
array_sil_2 = np.array(array_sil_2).reshape(-1, 1)
array_var = np.array(array_var).reshape(-1, 1)

reconstruction_error = np.array(reconstruction_error).reshape(-1, 1)

data_em_ica_wine = np.concatenate(
    (component_list, array_aic, array_bic, array_homo_1, array_comp_1, array_sil_1, array_avg_log), axis=1)

data_km_ica_wine = np.concatenate((component_list, array_homo_2, array_sil_2, array_var), axis=1)

reconstruction_error_ica_wine = np.concatenate((np.arange(1, 12).reshape(-1, 1), reconstruction_error), axis=1)

file = './data/data_em_ica_wine.csv'
with open(file, 'w', newline='') as output:
    writer = csv.writer(output, delimiter=',')
    writer.writerows(data_em_ica_wine)

file = './data/data_km_ica_wine.csv'
with open(file, 'w', newline='') as output:
    writer = csv.writer(output, delimiter=',')
    writer.writerows(data_km_ica_wine)

file = './data/reconstruction_error_ica_wine.csv'
with open(file, 'w', newline='') as output:
    writer = csv.writer(output, delimiter=',')
    writer.writerows(reconstruction_error_ica_wine)