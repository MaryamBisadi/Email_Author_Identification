
"""
    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
from email_preprocess import preprocess
from sklearn import tree
from sklearn.metrics import accuracy_score

features_train, features_test, labels_train, labels_test = preprocess()

clf = tree.DecisionTreeClassifier(min_samples_split=2)

print("feature no:", len(features_train[0]))

t0 = time()
clf = clf.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3))
predict = clf.predict(features_test)

acc = accuracy_score(predict, labels_test)
print("accuracy: ", acc)
