
"""
    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
from email_preprocess import preprocess
from sklearn.metrics import accuracy_score

features_train, features_test, labels_train, labels_test = preprocess()

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

t0 = time()
clf.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s")

predict = clf.predict(features_test)

acc = accuracy_score(predict, labels_test)
print("accuracy: ", acc)