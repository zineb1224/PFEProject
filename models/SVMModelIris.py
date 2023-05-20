from sklearn.svm import SVC
from sklearn.datasets import load_iris


def import_dataIris():
    data = load_iris()
    return data


class SVMModelIris:
    def __init__(self, kernel='linear', C=1.0, gamma=0):
        self.model = SVC(kernel=kernel, C=C, gamma=gamma)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def decision_function(self, X):
        if self.model is None:
            raise Exception("Model has not been trained yet!")
        return self.model.decision_function(X)

    def support_vectors_(self):
        return self.model.support_vectors_
