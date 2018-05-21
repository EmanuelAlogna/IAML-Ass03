#<!--------------------------------------------------------------------------->
#<!-- File       : As301_pca_visualization                                  -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2018042401 $"

########################################################################
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
import sys

import As301_classifier

########################################################################

dataset  = np.load("./outputs/dataset.npy")
df       = pd.read_csv("./outputs/hog_dataset.csv", index_col=0)
indices  = df.index.values
features = dataset[:,:-1]
labels   = dataset[:,-1]

########################################################################

# We are going to apply PCA to reduce the dimensions of our HOG feature
# to see if there is a obvious separation between the two classes 
# car and non-car

########################################################################
pca = PCA(n_components=2)

fig_imgs = plt.figure("PCA Analysis", figsize=(12,4))

colors = ['navy', 'darkorange']
lw = 2

positive_set = np.array([row for row in dataset if (row[-1] ==  1 )])
pos_size = positive_set.shape[0]
negative_set = np.array([row for row in dataset if (row[-1] == -1) ])
neg_size = negative_set.shape[0]

features_pca = pca.fit(np.vstack((positive_set,negative_set))).transform(np.vstack((positive_set,negative_set))) 
print(features_pca.shape)
print(labels.shape)
data_pca = np.vstack((features_pca.T, indices, labels)).T
print(data_pca[:5])

# Percentage of variance explained for each components
print('explained variance ratio (first two components): {0}'
    .format(str(pca.explained_variance_ratio_)))
var_expl_1st, var_expl_2nd = pca.explained_variance_ratio_

sub = fig_imgs.add_subplot(1,3,1)

plt.scatter(features_pca[:pos_size,0], features_pca[:pos_size,1], 
            color=colors[0], alpha=.8, lw=lw, label='car')
plt.scatter(features_pca[pos_size:,0], features_pca[pos_size:,1], 
            color=colors[1], alpha=.8, lw=lw, label='non-car')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of Car/Non-Car dataset')

# 1st principal component explains 26% of the total variance
plt.xlabel('1st principal component')
plt.xlim((-4,4))

# 2nd principal component explains 8% of the total variance
plt.ylabel('2nd principal component')
plt.ylim((-4,4))

plt.plot()


# Analyze the corner cases ('non-car' label but value of 1st component less than 0.5)
special_cases = np.array([row for row in data_pca if (row[-1] == -1 and row[0] < 0.5)])
img2_neg_set  = np.array([row for row in data_pca if (row[-1] == -1 and row[0] >= 0.5)])
img2_pos_set  = np.array([row for row in data_pca if (row[-1] == 1)])

sub = fig_imgs.add_subplot(1,3,2)

plt.scatter(img2_pos_set[:,0], img2_pos_set[:,1], 
            color=colors[0], alpha=.8, lw=lw, label='car')
plt.scatter(img2_neg_set[:,0], img2_neg_set[:,1], 
            color=colors[1], alpha=.8, lw=lw, label='non-car')
plt.scatter(special_cases[:,0], special_cases[:,1], 
            color="red", alpha=.8, lw=lw, label='special cases of non-car')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('Highlighting instances of Non-Cars\nthat are very similar to Cars', fontsize=8)
plt.xlabel(r"$1^{st}$ principal component")
plt.xlim((-4,4))
plt.ylabel(r"$2^{nd}$ principal component")
plt.ylim((-4,4))

plt.plot()

# analyze the special non-car indices 
special_indices = special_cases[:,-2].astype(np.int)
special_filenames = df.iloc[list(special_indices)]["files"].values

res = cv2.imread(special_filenames[0])

for fname in list(special_filenames)[1:]:
    img = cv2.imread(fname)
    res = np.hstack((res,img))

# create a 5x5 collage
collage = np.zeros((5*64,5*64,3), dtype=int)
collage[0:64,0:5*64,:]    = res[:,0:5*64,:].copy()
collage[64:128,0:5*64,:]  = res[:,5*64:10*64,:].copy()
collage[128:192,0:5*64,:] = res[:,10*64:15*64,:].copy()
collage[192:256,0:5*64,:] = res[:,15*64:20*64,:].copy()
collage[256:320,0:5*64,:] = res[:,20*64:25*64,:].copy()
collage = collage.astype(np.uint8)

sub = fig_imgs.add_subplot(1,3,3)
sub.set_title("Special Case Images")
sub.imshow(collage)
sub.axis("off")

plt.plot()


plt.show()

########################################################################



