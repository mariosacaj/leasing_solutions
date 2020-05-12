from script import load_data
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = load_data.load_data()
df.fillna(0, inplace=True)
df = shuffle(df).reset_index(drop=True)
#df = df.truncate(before=0, after=10000)

Y = df['target'].tolist()
df = df.drop(['target'],axis=1)
X = df.values.tolist()

X, Y = shuffle(X,Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.9)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)


clf = SVC(kernel='linear')
clf.fit(x_train,y_train)
print("FITTED")
y_pred = clf.predict(x_test)
print("PREDICTED")
print(accuracy_score(y_test,y_pred))
