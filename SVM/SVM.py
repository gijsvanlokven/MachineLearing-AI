import sklearn
from sklearn import datasets
from sklearn import svm

cancer = datasets.load_breast_cancer()

print(cancer.feature_names)
print(cancer.target_names)

X = cancer.data
Y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)