
import pickle
import cPickle
import numpy

from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif


def preprocess(emails_file="data/email_data.pkl", authors_file="data/email_authors.pkl"):

    authors_file_handler = open(authors_file, "r")
    authors = pickle.load(authors_file_handler)
    authors_file_handler.close()

    emails_file_handler = open(emails_file, "r")
    emails_data = cPickle.load(emails_file_handler)
    emails_file_handler.close()

    features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(emails_data, authors, test_size=0.1, random_state=42)

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed = vectorizer.transform(features_test)

    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(features_train_transformed, labels_train)
    features_train_transformed = selector.transform(features_train_transformed).toarray()
    features_test_transformed = selector.transform(features_test_transformed).toarray()

    return features_train_transformed, features_test_transformed, labels_train, labels_test