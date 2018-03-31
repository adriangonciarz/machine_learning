import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
test_idx = [0, 50, 100]

train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print(test_target)
print(clf.predict(test_data))

from sklearn.externals.six import StringIO
import pydotplus

dotdata = StringIO()
tree.export_graphviz(
    clf,
    out_file=dotdata,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True, rounded=True,
    impurity=False)
graph = pydotplus.graph_from_dot_data(dotdata.getvalue())
graph.write_pdf("target.pdf")
