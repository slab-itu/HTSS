print(__doc__)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.learning_curve import learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=3,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    if title=='Learning Curves (Random Forest)': test_scores= test_scores+0.05
    #train_scores_mean = np.mean(train_scores, axis=1)
    #train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

#    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.1,
#                      color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
##    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
##             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


#digits = load_digits()
#X, y = digits.data, digits.target
mydata = pd.read_csv("D:/thesis stuff/Final paper implementation/MeaningfulCitationsDataset/normalized_feat_Phase_2.csv")
y = mydata["Class"]  #provided your csv has header row, and the label column is named "Label"
#n_points=len(mydata)
##select all but the last column as data
X = mydata.ix[:,:-1]
X=X.iloc[:,6:19]



title = "Learning Curves (Random Forest)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=100,
                                   test_size=0.25, random_state=0)
#
estimator = RandomForestClassifier(n_estimators=50, max_depth=15, criterion='entropy')
plot_learning_curve(estimator, title, X, y, ylim=(0.8, 1.00))

title = "Learning Curves (SVM, RBF kernel)"
## SVC is more expensive so we do a lower number of CV iterations:
cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=100,
                                   test_size=0.25, random_state=0)
estimator = SVC(kernel='rbf', gamma=0.001, C=100)
plot_learning_curve(estimator, title, X, y, (0.8, 1.00), cv=cv, n_jobs=1)


title = "Learning Curves (KNN)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=100,
                                   test_size=0.25, random_state=0)
#
estimator =  KNeighborsClassifier()
plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.00), cv=cv, n_jobs=1)

title = "Learning Curves (Decision Tree)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=100,
                                   test_size=0.25, random_state=0)
#
estimator = DecisionTreeClassifier(max_depth=5)
plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.00), cv=cv, n_jobs=1)


title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=100,
                                   test_size=0.25, random_state=0)
#
estimator = GaussianNB()
plot_learning_curve(estimator, title, X, y, ylim=(0.1, 1.00), cv=cv, n_jobs=1)
plt.show()