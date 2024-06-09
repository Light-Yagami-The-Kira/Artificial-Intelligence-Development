from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
# print(iris.DESCR)
features = iris.data
labels = iris.target

# print(features[0], labels[0])

clf = KNeighborsClassifier()
clf.fit(features, labels)

print(clf.predict([[11,31,111,1]]))