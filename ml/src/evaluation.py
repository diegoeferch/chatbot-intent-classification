import sklearn
import numpy as np
import pandas as pd
from typing import Union, List, Optional, Tuple


def get_performance(
    predictions: Union[List, np.ndarray],
    y_test: Union[List, np.ndarray],
    labels: Optional[Union[List, np.ndarray]] = [1, 0],
) -> Tuple[float, float, float, float]:

    accuracy = sklearn.metrics.accuracy_score(y_test, predictions)
    precision = sklearn.metrics.precision_score(y_test, predictions, average='micro')
    recall = sklearn.metrics.recall_score(y_test, predictions, average='micro')
    f1_score = sklearn.metrics.f1_score(y_test, predictions, average='micro')

    report = sklearn.metrics.classification_report(y_test, predictions)
    cm = sklearn.metrics.confusion_matrix(y_test, predictions)

    cm_as_dataframe = pd.DataFrame(data=cm)

    print("Model Performance metrics:")
    print("-" * 30)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)
    print("\nModel Classification report:")
    print("-" * 30)
    print(report)
    print("\nPrediction Confusion Matrix:")
    print("-" * 30)
    print(cm_as_dataframe)

    return accuracy, precision, recall, f1_score
