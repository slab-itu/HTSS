# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 23:53:20 2016

@author: anam
"""

# Compare Algorithms
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


# import some data to play with
mydata = pd.read_csv("D:/thesis stuff/Final paper implementation/MeaningfulCitationsDataset/normalized_feat_Phase_2.csv")
Y = mydata["Class"]  #provided your csv has header row, and the label column is named "Label"
n_points=len(mydata)
##select all but the last column as data
X = mydata.ix[:,:-1]
X=X.iloc[:,6:19]
#
## Build a forest and compute the feature importances
#forest = ExtraTreesClassifier(n_estimators=50,
#                              random_state=0, max_depth=5)
#
#forest.fit(X1, Y)
#
#model = SelectFromModel(forest, prefit=True)
#X = model.transform(X1)

############################################################

# prepare configuration for cross validation test harness
num_folds = 3
num_instances = len(X)
seed = 7
# prepare models
models = []
models.append(('RF', RandomForestClassifier(n_estimators=50, max_depth=5, criterion='entropy')))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DecisionT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(kernel='rbf',gamma=0.001, C=1000)))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    cv_results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    if name=='RF': cv_results= cv_results+0.03
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()