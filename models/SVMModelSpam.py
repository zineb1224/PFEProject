from sklearn import svm
import pandas as pd


def import_dataSpam(file_path):
    data = pd.read_csv(file_path)
    # effectuer les opérations de prétraitement nécessaires
    return data


class SVMModelSpam:
    def __init__(self, kernel='linear', c=1.0):
        self.model = svm.SVC(kernel=kernel, C=c)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def decision_function(self, X):
        if self.model is None:
            raise Exception("Model has not been trained yet!")
        return self.model.decision_function(X)
