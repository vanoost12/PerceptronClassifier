import os
import sys
import argparse
import time
import itertools
import numpy as np
import pandas as pd


class PerceptronClassifier:
    def __init__(self):
        """
        Constructor for the PerceptronClassifier.
        """
        self.possible_labels = None
        self.weight_vectors = {}
        self.data_point_size = 0
        self.number_of_data_points = 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        This method trains a multiclass perceptron classifier on a given training set X with label set y.
        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
        Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
        Array datatype is guaranteed to be np.uint8.
        """
        self.data_point_size = len(X[0,:])
        self.number_of_data_points = len(X[:,0])
        self.possible_labels = list(set(y))
        for label in self.possible_labels:
            self.weight_vectors[label] = np.zeros(self.data_point_size)
        flag = True
        while flag:
            flag = False
            for index in range(self.number_of_data_points):
                xt = X[index, :]
                max_label = self.possible_labels[0]
                max_inner_dot_val = 0
                for label in self.possible_labels:
                    cur_val = xt @ self.weight_vectors[label]
                    if cur_val > max_inner_dot_val:
                        max_inner_dot_val = cur_val
                        max_label = label
                if y[index] != max_label:
                    flag = True
                    self.weight_vectors[y[index]] = self.weight_vectors[y[index]] + xt
                    self.weight_vectors[max_label] = self.weight_vectors[max_label] - xt


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call PerceptronClassifier.fit before calling this method.
        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
        Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """
        prediction = []
        number_of_samples = len(X[:, 0])
        for index in range(number_of_samples):
            xt = X[index, :]
            max_label = self.possible_labels[0]
            max_inner_dot_val = 0
            for label in self.possible_labels:
                cur_val = xt @ self.weight_vectors[label]
                if cur_val > max_inner_dot_val:
                    max_inner_dot_val = cur_val
                    max_label = label
            prediction.append(max_label)
        return np.array(prediction)
        ### Example code - don't use this:
        # return np.random.randint(low=0, high=2, size=len(X), dtype=np.uint8)


if __name__ == "__main__":

    print("*" * 20)
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    args = parser.parse_args()

    print("Processed input arguments:")
    print(f"csv = {args.csv}")

    print("Initiating PerceptronClassifier")
    model = PerceptronClassifier()
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)
    print("Fitting...")
    is_separable = model.fit(X, y)
    print("Done")
    y_pred = model.predict(X)
    print("Done")
    accuracy = np.sum(y_pred == y.ravel()) / y.shape[0]
    print(f"Train accuracy: {accuracy * 100 :.2f}%")
    print("*" * 20)
