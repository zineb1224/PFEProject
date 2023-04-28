import pandas as pd
import numpy as np
from sklearn import datasets
cancer = datasets.load_breast_cancer()
#print(cancer.data[:5])
#print(cancer.target)
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(cancer.data , cancer.target , test_size = 0.50 , random_state = 109)
from sklearn import svm
clf = svm.SVC(kernel = 'linear')
clf.fit(x_train , y_train)
y_pred = clf.predict(x_test)
from sklearn import metrics
print("accuracy :" , metrics.accuracy_score(y_test , y_pred))