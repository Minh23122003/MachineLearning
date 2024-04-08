import mglearn as mglearn
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

iris = datasets.load_iris()

X_iris = iris.data
y_iris = iris.target

X = X_iris[:, :2]
y = y_iris

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

scaler = preprocessing.StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

colors = ['red', 'greenyellow', 'blue']

for i in range(len(colors)):
    Xs = X_train[:, 0][y_train == i]
    ys = X_train[:, 1][y_train == i]
    plt.scatter(Xs, ys, c=colors[i])
plt.legend(iris.target_names)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

# iris_dataframe = pd.DataFrame(X_train, columns=iris.feature_names)
# grr = pd.scatter_maxtrix(iris_dataframe, c=y_train, figsize=(15, 15),
#                          marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

print(iris.feature_names)