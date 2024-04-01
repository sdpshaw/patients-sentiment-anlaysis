from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier


# models class having all the models
class Models:
    def __init__(self, X_train, y_train, weights=None):
        self.X_train = X_train
        self.y_train = y_train
        self.weights_dict = weights

    def linear_svc(self):
        """Linear Support Vector Classification"""
        clf = LinearSVC(penalty='l2', C=1.0, class_weight=self.weights_dict)
        clf.fit(self.X_train, self.y_train)
        return clf

    def log_reg(self):
        """Logistic Regression model"""
        clf = LogisticRegression(penalty='l2', C=1.0, class_weight=self.weights_dict)
        clf.fit(self.X_train, self.y_train)
        return clf

    def naive_bayes(self):
        """Naive Bayes"""
        clf = MultinomialNB(alpha=0.5, class_prior=None, fit_prior=True)
        clf.fit(self.X_train, self.y_train)
        return clf

    def random_forest(self):
        """Random Forest"""
        clf = RandomForestClassifier(n_estimators=100, criterion='entropy',
                                     bootstrap=False, class_weight=self.weights_dict)
        clf.fit(self.X_train, self.y_train)
        return clf

    def knn(self):
        """K-Nearest Neighbors"""
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(self.X_train, self.y_train)
        return clf
