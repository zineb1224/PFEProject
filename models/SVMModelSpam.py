from sklearn import svm
import pandas as pd


def import_data(file_path):
    data = pd.read_csv(file_path)
    # effectuer les opérations de prétraitement nécessaires
    return data


class SVMModelSpam:
    def __init__(self, kernel='linear'):
        self.model = svm.SVC(kernel=kernel)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)
