#<!--------------------------------------------------------------------------->
#<!-- File       : As301_pca_viz_compare                                    -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2018042401 $"

########################################################################
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys

import As301_classifier

########################################################################

dataset  = np.load("./outputs/dataset.npy")
df       = pd.read_csv("./outputs/hog_dataset.csv", index_col=0)
indices  = df.index.values
features = dataset[:,:-1]
labels   = dataset[:,-1]

positive_set = np.array([row for row in dataset if (row[-1] ==  1 )])
pos_size = positive_set.shape[0]
negative_set = np.array([row for row in dataset if (row[-1] == -1) ])
neg_size = negative_set.shape[0]

########################################################################

# Compare 2D PCA of the HOG features and the PCA applied on the 
# raw image data (grayscale).

########################################################################
pca = PCA(n_components=2)

fig_imgs = plt.figure("PCA HOG Comparison", figsize=(12,4))

colors = ['navy', 'darkorange']
lw = 2

sub = fig_imgs.add_subplot(1,2,1)

img_data = []

for fpath in list(df["files"]):
    img = cv2.imread(fpath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_data.append(gray.flatten())

img_data = np.array(img_data)


scaler = StandardScaler()
img_data = scaler.fit_transform(img_data)

img_pca = pca.fit(img_data).transform(img_data)
print('explained variance ratio (first two components): {0}'
    .format(str(pca.explained_variance_ratio_)))

plt.scatter(img_pca[:pos_size,0], img_pca[:pos_size,1], 
            color=colors[0], alpha=.8, lw=lw, label='car')
plt.scatter(img_pca[pos_size:,0], img_pca[pos_size:,1], 
            color=colors[1], alpha=.8, lw=lw, label='non-car')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of Car/Non-Car dataset (raw image data)')

# 1st principal component explains 40% of the total variance
plt.xlabel('1st principal component')
plt.xlim((-140,170))

# 2nd principal component explains 10% of the total variance
plt.ylabel('2nd principal component')
plt.ylim((-65,65))

plt.plot()

full_set = np.vstack((positive_set,negative_set))
features_pca = pca.fit(full_set).transform(full_set) 
data_pca = np.vstack((features_pca.T, indices, labels)).T

# Percentage of variance explained for each components
print('explained variance ratio (first two components): {0}'
    .format(str(pca.explained_variance_ratio_)))
var_expl_1st, var_expl_2nd = pca.explained_variance_ratio_

sub = fig_imgs.add_subplot(1,2,2)

plt.scatter(features_pca[:pos_size,0], features_pca[:pos_size,1], 
            color=colors[0], alpha=.8, lw=lw, label='car')
plt.scatter(features_pca[pos_size:,0], features_pca[pos_size:,1], 
            color=colors[1], alpha=.8, lw=lw, label='non-car')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of Car/Non-Car dataset (HOG)')

# 1st principal component explains 26% of the total variance
plt.xlabel('1st principal component')
plt.xlim((-4,4))

# 2nd principal component explains 8% of the total variance
plt.ylabel('2nd principal component')
plt.ylim((-4,4))

plt.plot()

plt.show()

########################################################################



