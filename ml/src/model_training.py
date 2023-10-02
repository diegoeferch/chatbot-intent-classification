import mlflow
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from evaluation import get_performance
from features import oh_labels, tf_idf_vectorization
from text_preprocessing import normalize_corpus

mlflow.set_tracking_uri('http://localhost:5000')


def train_logistic_regression(x_train, y_train, x_test, y_test):
    x_train = normalize_corpus(x_train['message'])
    x_test = normalize_corpus(x_test['message'])

    x_train, x_test = tf_idf_vectorization(x_train, x_test)
    with mlflow.start_run():
        print('Starting training...')
        lr_model = linear_model.LogisticRegression(max_iter=1000, multi_class='auto', solver='lbfgs')
        lr_model.fit(x_train, y_train)
        predicted_qualities = lr_model.predict(x_test)

        print('Assessing performance...')
        accuracy, precision, recall, f1_score = get_performance(
            predicted_qualities, y_test
        )

        print('Saving metrics to MLFlow...')
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('f1_score', f1_score)
        mlflow.sklearn.log_model(lr_model, 'model')

