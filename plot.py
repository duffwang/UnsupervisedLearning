import csv
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
from os import path


##Reconstruction Error

reconstruction_error_fa_phish = pd.read_csv("./data/reconstruction_error_fa_phish.csv", header= None)
reconstruction_error_ica_phish = pd.read_csv("./data/reconstruction_error_ica_phish.csv", header = None)
reconstruction_error_pca_phish = pd.read_csv("./data/reconstruction_error_pca_phish.csv", header = None)
reconstruction_error_rp_phish = pd.read_csv("./data/reconstruction_error_rp_phish.csv", header = None)


reconstruction_error_fa_wine = pd.read_csv("./data/reconstruction_error_fa_wine.csv", header= None)
reconstruction_error_ica_wine = pd.read_csv("./data/reconstruction_error_ica_wine.csv", header = None)
reconstruction_error_pca_wine = pd.read_csv("./data/reconstruction_error_pca_wine.csv", header = None)
reconstruction_error_rp_wine = pd.read_csv("./data/reconstruction_error_rp_wine.csv", header = None)

new_rec_error_rp_phish = [x / (10) for x in list(reconstruction_error_rp_phish.loc[:,1].values)]


fig1, ax1 = plt.subplots()
ax1.plot(reconstruction_error_pca_phish.loc[:,0].values, reconstruction_error_pca_phish.loc[:,1].values, linewidth = 2)
ax1.plot(reconstruction_error_ica_phish.loc[:,0].values, reconstruction_error_ica_phish.loc[:,1].values, linewidth = 2)
ax1.plot(reconstruction_error_rp_phish.loc[:,0].values, new_rec_error_rp_phish, linewidth = 2)
ax1.plot(reconstruction_error_fa_phish.loc[:,0].values, reconstruction_error_fa_phish.loc[:,1].values, linewidth = 2)

plt.legend(['PCA', 'ICA', 'RP error / 10', 'FA' ])
plt.xlabel('Number of components')
plt.ylabel('Reconstruction Error')
plt.title('Reconstruction Error for Phishing dataset')
plt.savefig("compare_plot/phish_reconstruct")

new_rec_error_rp_wine = [x / (1e3) for x in list(reconstruction_error_rp_wine.loc[:,1].values)]
fig2, ax2 = plt.subplots()
ax2.plot(reconstruction_error_pca_wine.loc[:,0].values, reconstruction_error_pca_wine.loc[:,1].values, linewidth = 2)
ax2.plot(reconstruction_error_ica_wine.loc[:,0].values, reconstruction_error_ica_wine.loc[:,1].values, linewidth = 2)
ax2.plot(reconstruction_error_rp_wine.loc[:,0].values, new_rec_error_rp_wine, linewidth = 2)
ax2.plot(reconstruction_error_fa_wine.loc[:,0].values, reconstruction_error_fa_wine.loc[:,1].values, linewidth = 2)

plt.legend(['PCA', 'ICA', 'RP Error /1e3', 'FA'])
plt.xlabel('Number of components')
plt.ylabel('Reconstruction Error')
plt.title('Reconstruction Error for wine dataset')
plt.savefig("compare_plot/wine_reconstruct")
plt.show()

##Performance evaluation scores
#Phishing dataset

data_em_pca_phish = pd.read_csv("./data/data_em_pca_phish.csv", header = None)
data_em_ica_phish = pd.read_csv("./data/data_em_ica_phish.csv", header = None)
data_em_rp_phish = pd.read_csv("./data/data_em_rp_phish.csv", header = None)
data_em_fa_phish = pd.read_csv("./data/data_em_fa_phish.csv", header = None)

data_km_pca_phish = pd.read_csv("./data/data_km_pca_phish.csv", header = None)
data_km_ica_phish = pd.read_csv("./data/data_km_ica_phish.csv", header = None)
data_km_rp_phish = pd.read_csv("./data/data_km_rp_phish.csv", header = None)
data_km_fa_phish = pd.read_csv("./data/data_km_fa_phish.csv", header = None)

# fig3,ax3 = plt.subplots()
# ax3.plot(data_em_pca_phish.loc[:,0].values, data_em_pca_phish.loc[:,1].values, linewidth = 2)
# ax3.plot(data_em_ica_phish.loc[:,0].values, data_em_ica_phish.loc[:,1].values, linewidth = 2)
# ax3.plot(data_em_rp_phish.loc[:,0].values, data_em_rp_phish.loc[:,1].values, linewidth = 2)
# ax3.plot(data_em_fa_phish.loc[:,0].values, data_em_fa_phish.loc[:,1].values, linewidth = 2)
# plt.legend(['PCA', 'ICA', 'RP', 'FA'])
# plt.xlabel('Number of components')
# plt.ylabel('AIC')
# plt.title('AIC curve for expected maximization for Phishing dataset')

# fig4,ax4 = plt.subplots()
# ax4.plot(data_em_pca_phish.loc[:,0].values, data_em_pca_phish.loc[:,2].values, linewidth = 2)
# ax4.plot(data_em_ica_phish.loc[:,0].values, data_em_ica_phish.loc[:,2].values, linewidth = 2)
# ax4.plot(data_em_rp_phish.loc[:,0].values, data_em_rp_phish.loc[:,2].values, linewidth = 2)
# ax4.plot(data_em_fa_phish.loc[:,0].values, data_em_fa_phish.loc[:,2].values, linewidth = 2)
# plt.legend(['PCA', 'ICA', 'RP', 'FA'])
# plt.xlabel('Number of components')
# plt.ylabel('BIC')
# plt.title('BIC curve for expected maximization for Phishing dataset')


fig5,ax5 = plt.subplots()
ax5.plot(data_em_pca_phish.loc[:,0].values, data_em_pca_phish.loc[:,3].values, linewidth = 2)
ax5.plot(data_em_ica_phish.loc[:,0].values, data_em_ica_phish.loc[:,3].values, linewidth = 2)
ax5.plot(data_em_rp_phish.loc[:,0].values, data_em_rp_phish.loc[:,3].values, linewidth = 2)
ax5.plot(data_em_fa_phish.loc[:,0].values, data_em_fa_phish.loc[:,3].values, linewidth = 2)
plt.legend(['PCA', 'ICA', 'RP', 'FA'])
plt.xlabel('Number of components')
plt.ylabel('Homogenity Score')
plt.title('Homogenity Score for expected maximization for Phishing dataset')
plt.savefig("compare_plot/phish_compare_1")

fig6,ax6 = plt.subplots()
ax6.plot(data_em_pca_phish.loc[:,0].values, data_em_pca_phish.loc[:,4].values, linewidth = 2)
ax6.plot(data_em_ica_phish.loc[:,0].values, data_em_ica_phish.loc[:,4].values, linewidth = 2)
ax6.plot(data_em_rp_phish.loc[:,0].values, data_em_rp_phish.loc[:,4].values, linewidth = 2)
ax6.plot(data_em_fa_phish.loc[:,0].values, data_em_fa_phish.loc[:,4].values, linewidth = 2)
plt.legend(['PCA', 'ICA', 'RP', 'FA'])
plt.xlabel('Number of components')
plt.ylabel('Completeness Score')
plt.title('Completeness Score for expected maximization for Phishing dataset')
plt.savefig("compare_plot/phish_compare_2")

fig7,ax7 = plt.subplots()
ax7.plot(data_em_pca_phish.loc[:,0].values, data_em_pca_phish.loc[:,5].values, linewidth = 2)
ax7.plot(data_em_ica_phish.loc[:,0].values, data_em_ica_phish.loc[:,5].values, linewidth = 2)
ax7.plot(data_em_rp_phish.loc[:,0].values, data_em_rp_phish.loc[:,5].values, linewidth = 2)
ax7.plot(data_em_fa_phish.loc[:,0].values, data_em_fa_phish.loc[:,5].values, linewidth = 2)
plt.legend(['PCA', 'ICA', 'RP', 'FA'])
plt.xlabel('Number of components')
plt.ylabel('Silhoutette Score')
plt.title('Silhoutette Score for expected maximization for Phishing dataset')
plt.savefig("compare_plot/phish_compare_3")

fig8,ax8 = plt.subplots()
ax8.plot(data_em_pca_phish.loc[:,0].values, data_em_pca_phish.loc[:,6].values, linewidth = 2)
ax8.plot(data_em_ica_phish.loc[:,0].values, data_em_ica_phish.loc[:,6].values, linewidth = 2)
ax8.plot(data_em_rp_phish.loc[:,0].values, data_em_rp_phish.loc[:,6].values, linewidth = 2)
ax8.plot(data_em_fa_phish.loc[:,0].values, data_em_fa_phish.loc[:,6].values, linewidth = 2)
plt.legend(['PCA', 'ICA', 'RP', 'FA'])
plt.xlabel('Number of components')
plt.ylabel('Average Log likelihood')
plt.title('Per sample average log likelihood for EM for Phishing dataset')
plt.savefig("compare_plot/phish_compare_4")

fig9,ax9 = plt.subplots()
ax9.plot(data_km_pca_phish.loc[:,0].values, data_km_pca_phish.loc[:,1].values, linewidth = 2)
ax9.plot(data_km_ica_phish.loc[:,0].values, data_km_ica_phish.loc[:,1].values, linewidth = 2)
ax9.plot(data_km_rp_phish.loc[:,0].values, data_km_rp_phish.loc[:,1].values, linewidth = 2)
ax9.plot(data_km_fa_phish.loc[:,0].values, data_km_fa_phish.loc[:,1].values, linewidth = 2)
plt.legend(['PCA', 'ICA', 'RP', 'FA'])
plt.xlabel('Number of components')
plt.ylabel('Homogenity Score')
plt.title('Homogenity Score for kmeans for Phishing dataset')
plt.savefig("compare_plot/phish_compare_5")


fig10,ax10 = plt.subplots()
ax10.plot(data_km_pca_phish.loc[:,0].values, data_km_pca_phish.loc[:,2].values, linewidth = 2)
ax10.plot(data_km_ica_phish.loc[:,0].values, data_km_ica_phish.loc[:,2].values, linewidth = 2)
ax10.plot(data_km_rp_phish.loc[:,0].values, data_km_rp_phish.loc[:,2].values, linewidth = 2)
ax10.plot(data_km_fa_phish.loc[:,0].values, data_km_fa_phish.loc[:,2].values, linewidth = 2)
plt.legend(['PCA', 'ICA', 'RP', 'FA'])
plt.xlabel('Number of components')
plt.ylabel('Silhoutette Score')
plt.title('Silhoutette Score for kmeans for Phishing dataset')
plt.savefig("compare_plot/phish_compare_6")

pca_var = [x/(1e9) for x in data_km_pca_phish.loc[:,3].values]

fa_var = [x/(1e4) for x in data_km_fa_phish.loc[:,3].values]

rp_var = [x/(1e9) for x in data_km_rp_phish.loc[:,3].values]
print(pca_var)
print(fa_var)
print(data_km_ica_phish.loc[:,3].values)
print(rp_var)
fig11,ax11 = plt.subplots()
ax11.plot(data_km_pca_phish.loc[:,0].values, pca_var, linewidth = 2)
ax11.plot(data_km_ica_phish.loc[:,0].values, data_km_ica_phish.loc[:,3].values, linewidth = 2)
ax11.plot(data_km_rp_phish.loc[:,0].values, rp_var, linewidth = 2)
ax11.plot(data_km_fa_phish.loc[:,0].values, fa_var, linewidth = 2)
plt.legend(['PCA var / 1e9', 'ICA', 'RP var /1e9', 'FA noise var /1e4'])
plt.xlabel('Number of components')
plt.ylabel('Variance')
plt.title('Variance explained by each cluster for kmeans for Phishing dataset')
plt.savefig("compare_plot/phish_compare_7")

# plt.show()

##Performance evaluation scores
#wine dataset

data_em_pca_wine = pd.read_csv("./data/data_em_pca_wine.csv", header = None)
data_em_ica_wine = pd.read_csv("./data/data_em_ica_wine.csv", header = None)
data_em_rp_wine = pd.read_csv("./data/data_em_rp_wine.csv", header = None)
data_em_fa_wine = pd.read_csv("./data/data_em_fa_wine.csv", header = None)

data_km_pca_wine = pd.read_csv("./data/data_km_pca_wine.csv", header = None)
data_km_ica_wine = pd.read_csv("./data/data_km_ica_wine.csv", header = None)
data_km_rp_wine = pd.read_csv("./data/data_km_rp_wine.csv", header = None)
data_km_fa_wine = pd.read_csv("./data/data_km_fa_wine.csv", header = None)

# fig12,ax12 = plt.subplots()
# ax12.plot(data_em_pca_wine.loc[:,0].values, data_em_pca_wine.loc[:,1].values, linewidth = 2)
# ax12.plot(data_em_ica_wine.loc[:,0].values, data_em_ica_wine.loc[:,1].values, linewidth = 2)
# ax12.plot(data_em_rp_wine.loc[:,0].values, data_em_rp_wine.loc[:,1].values, linewidth = 2)
# ax12.plot(data_em_fa_wine.loc[:,0].values, data_em_fa_wine.loc[:,1].values, linewidth = 2)
# plt.legend(['PCA', 'ICA', 'RP', 'FA'])
# plt.xlabel('Number of components')
# plt.ylabel('AIC')
# plt.title('AIC curve for expected maximization for wineless drive diagnosis dataset')

# fig13,ax13 = plt.subplots()
# ax13.plot(data_em_pca_wine.loc[:,0].values, data_em_pca_wine.loc[:,2].values, linewidth = 2)
# ax13.plot(data_em_ica_wine.loc[:,0].values, data_em_ica_wine.loc[:,2].values, linewidth = 2)
# ax13.plot(data_em_rp_wine.loc[:,0].values, data_em_rp_wine.loc[:,2].values, linewidth = 2)
# ax13.plot(data_em_fa_wine.loc[:,0].values, data_em_fa_wine.loc[:,2].values, linewidth = 2)
# plt.legend(['PCA', 'ICA', 'RP', 'FA'])
# plt.xlabel('Number of components')
# plt.ylabel('BIC')
# plt.title('BIC curve for expected maximization for wineless drive diagnosis dataset')


fig14,ax14 = plt.subplots()
ax14.plot(data_em_pca_wine.loc[:,0].values, data_em_pca_wine.loc[:,3].values, linewidth = 2)
ax14.plot(data_em_ica_wine.loc[:,0].values, data_em_ica_wine.loc[:,3].values, linewidth = 2)
ax14.plot(data_em_rp_wine.loc[:,0].values, data_em_rp_wine.loc[:,3].values, linewidth = 2)
ax14.plot(data_em_fa_wine.loc[:,0].values, data_em_fa_wine.loc[:,3].values, linewidth = 2)
plt.legend(['PCA', 'ICA', 'RP', 'FA'])
plt.xlabel('Number of components')
plt.ylabel('Homogenity Score')
plt.title('Homogenity Score for expected maximization for wine dataset')
plt.savefig("compare_plot/wine_compare_1")

fig15,ax15 = plt.subplots()
ax15.plot(data_em_pca_wine.loc[:,0].values, data_em_pca_wine.loc[:,4].values, linewidth = 2)
ax15.plot(data_em_ica_wine.loc[:,0].values, data_em_ica_wine.loc[:,4].values, linewidth = 2)
ax15.plot(data_em_rp_wine.loc[:,0].values, data_em_rp_wine.loc[:,4].values, linewidth = 2)
ax15.plot(data_em_fa_wine.loc[:,0].values, data_em_fa_wine.loc[:,4].values, linewidth = 2)
plt.legend(['PCA', 'ICA', 'RP', 'FA'])
plt.xlabel('Number of components')
plt.ylabel('Completeness Score')
plt.title('Completeness Score for expected maximization for wine dataset')
plt.savefig("compare_plot/wine_compare_2")

fig16,ax16 = plt.subplots()
ax16.plot(data_em_pca_wine.loc[:,0].values, data_em_pca_wine.loc[:,5].values, linewidth = 2)
ax16.plot(data_em_ica_wine.loc[:,0].values, data_em_ica_wine.loc[:,5].values, linewidth = 2)
ax16.plot(data_em_rp_wine.loc[:,0].values, data_em_rp_wine.loc[:,5].values, linewidth = 2)
ax16.plot(data_em_fa_wine.loc[:,0].values, data_em_fa_wine.loc[:,5].values, linewidth = 2)
plt.legend(['PCA', 'ICA', 'RP', 'FA'])
plt.xlabel('Number of components')
plt.ylabel('Silhoutette Score')
plt.title('Silhoutette Score for expected maximization for wine dataset')
plt.savefig("compare_plot/wine_compare_3")

fig17,ax17 = plt.subplots()
ax17.plot(data_em_pca_wine.loc[:,0].values, data_em_pca_wine.loc[:,6].values, linewidth = 2)
ax17.plot(data_em_ica_wine.loc[:,0].values, data_em_ica_wine.loc[:,6].values, linewidth = 2)
ax17.plot(data_em_rp_wine.loc[:,0].values, data_em_rp_wine.loc[:,6].values, linewidth = 2)
ax17.plot(data_em_fa_wine.loc[:,0].values, data_em_fa_wine.loc[:,6].values, linewidth = 2)
plt.legend(['PCA', 'ICA', 'RP', 'FA'])
plt.xlabel('Number of components')
plt.ylabel('Average Log likelihood')
plt.title('Per sample avg log likelihood for EM for wine data')
plt.savefig("compare_plot/wine_compare_4")

fig18,ax18 = plt.subplots()
ax18.plot(data_km_pca_wine.loc[:,0].values, data_km_pca_wine.loc[:,1].values, linewidth = 2)
ax18.plot(data_km_ica_wine.loc[:,0].values, data_km_ica_wine.loc[:,1].values, linewidth = 2)
ax18.plot(data_km_rp_wine.loc[:,0].values, data_km_rp_wine.loc[:,1].values, linewidth = 2)
ax18.plot(data_km_fa_wine.loc[:,0].values, data_km_fa_wine.loc[:,1].values, linewidth = 2)
plt.legend(['PCA', 'ICA', 'RP', 'FA'])
plt.xlabel('Number of components')
plt.ylabel('Homogenity Score')
plt.title('Homogenity Score for kmeans for wine dataset')
plt.savefig("compare_plot/wine_compare_5")


fig19,ax19 = plt.subplots()
ax19.plot(data_km_pca_wine.loc[:,0].values, data_km_pca_wine.loc[:,2].values, linewidth = 2)
ax19.plot(data_km_ica_wine.loc[:,0].values, data_km_ica_wine.loc[:,2].values, linewidth = 2)
ax19.plot(data_km_rp_wine.loc[:,0].values, data_km_rp_wine.loc[:,2].values, linewidth = 2)
ax19.plot(data_km_fa_wine.loc[:,0].values, data_km_fa_wine.loc[:,2].values, linewidth = 2)
plt.legend(['PCA', 'ICA', 'RP', 'FA'])
plt.xlabel('Number of components')
plt.ylabel('Silhoutette Score')
plt.title('Silhoutette Score for kmeans for wine dataset')
plt.savefig("compare_plot/wine_compare_6")

fig20,ax20 = plt.subplots()
ax20.plot(data_km_pca_wine.loc[:,0].values, data_km_pca_wine.loc[:,3].values, linewidth = 2)
ax20.plot(data_km_ica_wine.loc[:,0].values, data_km_ica_wine.loc[:,3].values, linewidth = 2)
ax20.plot(data_km_rp_wine.loc[:,0].values, data_km_rp_wine.loc[:,3].values, linewidth = 2)
ax20.plot(data_km_fa_wine.loc[:,0].values, data_km_fa_wine.loc[:,3].values, linewidth = 2)
plt.legend(['PCA', 'ICA', 'RP', 'FA'])
plt.xlabel('Number of components')
plt.ylabel('Variance explained')
plt.title('Variance explained by each cluster for kmeans for wineless drive dataset')
plt.savefig("compare_plot/wine_compare_7")

plt.show()