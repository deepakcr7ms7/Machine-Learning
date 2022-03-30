from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split
iris = load_iris()
decision_tree = DecisionTreeClassifier(splitter="best",max_depth=3)
X, Y = iris.data, iris.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)
decision_tree = decision_tree.fit(X, Y)
r = export_text(decision_tree, feature_names=iris['feature_names'])
print(r)