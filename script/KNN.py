from script import load_data
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

df = load_data.load_data()
df.fillna(0, inplace=True)

X = []
Y = []

Y = df['target'].tolist()
df = df.drop(['target'],axis=1)
X = df.values.tolist()

x_train = []
y_train = []
x_test = []
y_test = []

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)


classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(x_train, y_train)

print(classifier.score(x_test, y_test))

plot_confusion_matrix(classifier, X, Y, normalize='true')
plt.show(block=True)