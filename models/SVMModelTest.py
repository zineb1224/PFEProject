from sklearn import svm

class SVM_Model:
    def __init__(self):
        self.model = svm.SVC()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)