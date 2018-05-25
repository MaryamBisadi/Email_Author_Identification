
"""
    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

features_train, features_test, labels_train, labels_test = preprocess()

clf = SVC(C=10000, kernel="rbf")

t0 = time()
clf.fit(features_train, labels_train)
print("training time: ", round(time()-t0, 3))

predict = clf.predict(features_test)
print("predict to be Chris", sum(predict))
print("predict to be Sara", len(predict)-sum(predict))


acc = accuracy_score(predict, labels_test)
print("accuracy: ", acc)