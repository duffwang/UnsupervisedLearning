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
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import scipy
from sklearn import random_projection
from cluster_func import em
from cluster_func import kmeans
from sklearn.neural_network import MLPClassifier



data = pd.read_csv('winequality-data.csv')

X = data.iloc[:,:-2]
y = data.iloc[:,-2]
y = y > 6

#Splitting data into training and testing and keeping testing data aside
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

n_classes = 7
##########################################################################################################
#PCA

print('PCA....')
time_pca = []
n_components_pca = range(1,12)
cv_score_pca = []
for comp_pca in n_components_pca:

    #Reducing the dimensions with optimal number of components
    pca_new = PCA(n_components = comp_pca)
    pca_new.fit(X_train)
    X_transformed_pca = pca_new.transform(X)
    nodes_hidden_layer = int((comp_pca + n_classes)/2)
    #neural network learner
    t1 = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(nodes_hidden_layer,),max_iter=500)

    cv_score_pca.append(np.mean(cross_val_score(mlp, X_transformed_pca, y, cv = 3)))

    t2 = time.time()

    time_pca.append((t2 - t1))


print('Adding cluster label and checking accuracy')

#Adding a cluster label as a feature

cv_score_em_pca = []
cv_score_km_pca = []
clf_em = GaussianMixture(n_components=n_classes,covariance_type='spherical', max_iter= 500, init_params= 'kmeans')
clf_km = KMeans(n_clusters= n_classes, init='k-means++')


for comp_pca in n_components_pca:

    nodes_hidden_layer = int((comp_pca + n_classes)/2)
    #neural network learner
    mlp = MLPClassifier(hidden_layer_sizes=(nodes_hidden_layer,),max_iter=500)

    #Reducing the dimensions with optimal number of components
    pca_new = PCA(n_components = comp_pca)
    pca_new.fit(X_train)
    X_transformed_pca = pca_new.transform(X)

    clf_em.fit(X_transformed_pca)
    cluster_em = clf_em.predict(X_transformed_pca)
    cluster_em = np.array(cluster_em).reshape(-1,1)

    X_transformed_em_pca = np.concatenate((X_transformed_pca, cluster_em), axis=1)


    cv_score_em_pca.append(np.mean(cross_val_score(mlp, X_transformed_em_pca, y, cv = 3)))


    clf_km.fit(X_transformed_pca)
    cluster_km = clf_km.predict(X_transformed_pca)
    cluster_km = np.array(cluster_km).reshape(-1,1)

    X_transformed_km_pca = np.concatenate((X_transformed_pca, cluster_km), axis=1)


    cv_score_km_pca.append(np.mean(cross_val_score(mlp, X_transformed_km_pca, y, cv = 3)))



#Plotting

fig1, ax1 = plt.subplots()
ax1.plot(n_components_pca, cv_score_pca, linewidth =2)
ax1.plot(n_components_pca, cv_score_em_pca, linewidth = 2)
ax1.plot(n_components_pca, cv_score_km_pca, linewidth = 2)
plt.legend(['without cluster label', 'with EM label', 'with KMeans label'])
plt.xlabel("Number of components")
plt.ylabel("Three fold Cross Validation score")
plt.title("Neural network accuracy with dimensionally reduced dataset using PCA")
plt.savefig("nn_plotwine/phish_nn_1")
#plt.show()


##########################################################################################################
#ICA

print('ICA...')
n_components_ica = range(1,12)
cv_score_ica = []
time_ica = []
for comp_ica in n_components_ica:


    #Reducing the dimensions with optimal number of components
    ica_new = FastICA(n_components = comp_ica)
    ica_new.fit(X_train)
    X_transformed_ica = ica_new.transform(X)
    nodes_hidden_layer = int((comp_ica + n_classes)/2)
    #neural network learner
    t1 = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(nodes_hidden_layer,),max_iter=500)

    cv_score_ica.append(np.mean(cross_val_score(mlp, X_transformed_ica, y, cv = 3)))

    t2 = time.time()
    time_ica.append((t2 - t1))


#Adding a cluster label as a feature

print('Adding Cluster label and checking accuracy')
cv_score_em_ica = []
cv_score_km_ica = []
clf_em = GaussianMixture(n_components=n_classes, covariance_type='spherical', max_iter= 500, init_params= 'kmeans')
clf_km = KMeans(n_clusters= n_classes, init='k-means++')


for comp_ica in n_components_ica:

    nodes_hidden_layer = int((comp_ica + n_classes)/2)
    #neural network learner
    mlp = MLPClassifier(hidden_layer_sizes=(nodes_hidden_layer,),max_iter=500)

    #Reducing the dimensions with optimal number of components
    ica_new = FastICA(n_components = comp_ica)
    ica_new.fit(X_train)
    X_transformed_ica = ica_new.transform(X)

    clf_em.fit(X_transformed_ica)
    cluster_em = clf_em.predict(X_transformed_ica)
    cluster_em = np.array(cluster_em).reshape(-1,1)

    X_transformed_em_ica = np.concatenate((X_transformed_ica, cluster_em), axis=1)


    cv_score_em_ica.append(np.mean(cross_val_score(mlp, X_transformed_em_ica, y, cv = 3)))


    clf_km.fit(X_transformed_ica)
    cluster_km = clf_km.predict(X_transformed_ica)
    cluster_km = np.array(cluster_km).reshape(-1,1)

    X_transformed_km_ica = np.concatenate((X_transformed_ica, cluster_km), axis=1)


    cv_score_km_ica.append(np.mean(cross_val_score(mlp, X_transformed_km_ica, y, cv = 3)))



#Reducing the dimensions with optimal number of components
fig2, ax2 = plt.subplots()
ax2.plot(n_components_ica, cv_score_ica, linewidth = 2)
ax2.plot(n_components_ica, cv_score_em_ica, linewidth = 2)
ax2.plot(n_components_ica, cv_score_km_ica, linewidth = 2)
plt.legend(['without cluster label', 'with EM label', 'with Kmeans label'])
plt.xlabel("Number of components")
plt.ylabel("Three fold Cross Validation score")
plt.title("Neural network accuracy with dimensionally reduced dataset using ICA")
plt.savefig("nn_plotwine/phish_nn_2")
#plt.show()


# ##########################################################################################################
#RP


print('RP...')
n_components_rp = range(1,12)
cv_score_rp = []

time_rp = []
for comp_rp in n_components_rp:

    #Reducing the dimensions with optimal number of components
    rp_new = random_projection.GaussianRandomProjection(n_components = comp_rp)
    rp_new.fit(X_train)
    X_transformed_rp = rp_new.transform(X)
    nodes_hidden_layer = int((comp_rp + n_classes)/2)
    #neural network learner
    t1 = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(nodes_hidden_layer,),max_iter=500)

    cv_score_rp.append(np.mean(cross_val_score(mlp, X_transformed_rp, y, cv = 3)))

    t2 = time.time()

    time_rp.append((t2 - t1))


#Adding a cluster label as a feature
print('Adding cluster label and checking accuracy')
cv_score_em_rp = []
cv_score_km_rp = []
clf_em = GaussianMixture(n_components=n_classes, covariance_type='spherical', max_iter= 500, init_params= 'kmeans')
clf_km = KMeans(n_clusters= n_classes, init='k-means++')


for comp_rp in n_components_rp:

    nodes_hidden_layer = int((comp_rp + n_classes)/2)
    #neural network learner
    mlp = MLPClassifier(hidden_layer_sizes=(nodes_hidden_layer,),max_iter=500)

    #Reducing the dimensions with optimal number of components
    rp_new = random_projection.GaussianRandomProjection(n_components = comp_rp)
    rp_new.fit(X_train)
    X_transformed_rp = rp_new.transform(X)

    clf_em.fit(X_transformed_rp)
    cluster_em = clf_em.predict(X_transformed_rp)
    cluster_em = np.array(cluster_em).reshape(-1,1)

    X_transformed_em_rp = np.concatenate((X_transformed_rp, cluster_em), axis=1)


    cv_score_em_rp.append(np.mean(cross_val_score(mlp, X_transformed_em_rp, y, cv = 3)))


    clf_km.fit(X_transformed_rp)
    cluster_km = clf_km.predict(X_transformed_rp)
    cluster_km = np.array(cluster_km).reshape(-1,1)

    X_transformed_km_rp = np.concatenate((X_transformed_rp, cluster_km), axis=1)


    cv_score_km_rp.append(np.mean(cross_val_score(mlp, X_transformed_km_rp, y, cv = 3)))


fig3, ax3 = plt.subplots()
ax3.plot(n_components_rp, cv_score_rp, linewidth= 2)
ax3.plot(n_components_rp, cv_score_em_rp, linewidth =2)
ax3.plot(n_components_rp, cv_score_km_rp, linewidth = 2)
plt.legend(['without cluster label', 'with EM label', 'with Kmeans label'])
plt.xlabel("Number of components")
plt.ylabel("Three fold Cross Validation score")
plt.title("Neural network accuracy with dimensionally reduced dataset using RP")
plt.savefig("nn_plotwine/phish_nn_3")
#plt.show()

# ##########################################################################################################
#fa

print('FA...')
n_components_fa = range(1,12)
cv_score_fa = []

time_fa = []
for comp_fa in n_components_fa:

    #Reducing the dimensions with optimal number of components
    fa_new = FactorAnalysis(n_components = comp_fa, max_iter = 100)
    fa_new.fit(X_train)
    X_transformed_fa = fa_new.transform(X)
    nodes_hidden_layer = int((comp_fa + n_classes)/2)
    #neural network learner
    t1 = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(nodes_hidden_layer,),max_iter=500)

    cv_score_fa.append(np.mean(cross_val_score(mlp, X_transformed_fa, y, cv = 3)))

    t2 = time.time()

    time_fa.append((t2 - t1))



#Adding a cluster label as a feature

print('Adding cluster label and checking accuracy.')
cv_score_em_fa = []
cv_score_km_fa = []
clf_em = GaussianMixture(n_components=n_classes, covariance_type='spherical', max_iter= 500, init_params= 'kmeans')
clf_km = KMeans(n_clusters= n_classes, init='k-means++')


for comp_fa in n_components_fa:

    nodes_hidden_layer = int((comp_fa + n_classes)/2)
    #neural network learner
    mlp = MLPClassifier(hidden_layer_sizes=(nodes_hidden_layer,),max_iter=500)

    #Reducing the dimensions with optimal number of components
    fa_new = FactorAnalysis(n_components = comp_fa, max_iter = 100)
    fa_new.fit(X_train)
    X_transformed_fa = fa_new.transform(X)

    clf_em.fit(X_transformed_fa)
    cluster_em = clf_em.predict(X_transformed_fa)
    cluster_em = np.array(cluster_em).reshape(-1,1)

    X_transformed_em_fa = np.concatenate((X_transformed_fa, cluster_em), axis=1)


    cv_score_em_fa.append(np.mean(cross_val_score(mlp, X_transformed_em_fa, y, cv = 3)))


    clf_km.fit(X_transformed_fa)
    cluster_km = clf_km.predict(X_transformed_fa)
    cluster_km = np.array(cluster_km).reshape(-1,1)

    X_transformed_km_fa = np.concatenate((X_transformed_fa, cluster_km), axis=1)


    cv_score_km_fa.append(np.mean(cross_val_score(mlp, X_transformed_km_fa, y, cv = 3)))


fig4, ax4 = plt.subplots()
ax4.plot(n_components_fa, cv_score_fa, linewidth= 2)
ax4.plot(n_components_fa, cv_score_em_fa, linewidth =2)
ax4.plot(n_components_fa, cv_score_km_fa, linewidth =2)
plt.legend(['without cluster label', 'with EM label', 'with Kmeans label'])
plt.xlabel("Number of components")
plt.ylabel("Three fold Cross Validation score")
plt.title("Neural network accuracy with dimensionally reduced dataset using FA")
plt.savefig("nn_plotwine/phish_nn_4")
#plt.show()


#############################################################################################################
#Plotting neural network time
#pca

print('plotting time graph')
fig5, ax5 = plt.subplots()
plt.plot(n_components_pca, time_pca, linewidth =2)
plt.plot(n_components_ica, time_ica, linewidth=2)
plt.plot(n_components_rp, time_rp, linewidth=2)
plt.plot(n_components_fa, time_fa, linewidth=2)
plt.legend(['PCA', 'ICA', 'RP', 'FA'])
plt.xlabel("Number of components")
plt.ylabel("Total training time for 3 fold CV")
plt.title("Neural network computation time after dimensionality reduction")
plt.savefig("nn_plotwine/phish_nn_5")
#plt.show()

