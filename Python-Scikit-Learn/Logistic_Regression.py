# g (Ey) = a + bx1 + cx2 -----> Link Function

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()

# print(iris.DESCR)

X = iris['data'][:, 3:]
Y = (iris["target"] == 2).astype(np.int32)

clf = LogisticRegression()
clf.fit(X,Y)

example = clf.predict([[1.6]])
# print(example)

# USING MATPLOTLIB TO VISUALISE

X_new = np.linspace(0,3,1000).reshape(-1,1)
# print(X_new)
Y_prob = clf.predict_proba(X_new)



plt.plot(X_new, Y_prob[:, 1], "g-", label="virginica")



plt.show()