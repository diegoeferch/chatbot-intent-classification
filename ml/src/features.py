import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(ngram_range=(1, 2), norm=None)

def oh_labels(y: pd.Series):
    return pd.get_dummies(y, dtype=int)


def tf_idf_vectorization(x_train, x_test):
    train_features = vectorizer.fit_transform(x_train).toarray()
    test_features = vectorizer.transform(x_test).toarray()
    return train_features, test_features
