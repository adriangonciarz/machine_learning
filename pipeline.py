from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.5)