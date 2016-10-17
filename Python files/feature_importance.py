# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 15:03:23 2016

@author: anam
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

# Build a classification task using 3 informative features

mydata = pd.read_csv("D:/thesis stuff/Final paper implementation/MeaningfulCitationsDataset/testing.csv")
y = mydata["Class"]  #provided your csv has header row, and the label column is named "Label"

#select all but the last column as data
X = mydata.ix[:,:-1]
X=X.iloc[:,0:13]


#X, y = make_classification(n_samples=1000,
#                           n_features=10,
#                           n_informative=3,
#                           n_redundant=0,
#                           n_repeated=0,
#                           n_classes=2,
#                           random_state=0,
#                           shuffle=False)

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=50,
                              random_state=0)

forest.fit(X, y)

model = SelectFromModel(forest, prefit=True)
X_new = model.transform(X)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="g", align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()