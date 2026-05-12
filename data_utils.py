import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_iris():
    iris = load_iris()
    x = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return x_scaled, y, feature_names, target_names


def class_distribution(y):
    labels, counts = np.unique(y, return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts)}
