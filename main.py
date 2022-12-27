import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import seaborn as sns
from matplotlib import interactive
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("cpdata.csv", index_col=False)
X2 = dataset.iloc[:50, [0, 1, 2, 3]]
X = dataset.iloc[:, [0, 1, 2, 3]].values
Y = dataset.iloc[:, [4]].values

for column in dataset.columns:
    interactive(True)
    plt.show()
    sns.displot(dataset[column])

interactive(False)

sns.boxplot(dataset)
sns.pairplot(dataset)
plt.show()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

svm = svm.SVC()
svm.fit(x_train, np.ravel(y_train))
print("Support Vector Machine: ", svm.score(x_test, y_test))

cm = confusion_matrix(y_test, svm.predict(x_test))
sns.heatmap(cm, annot=True)

cr = classification_report(y_test, svm.predict(x_test))
print(cr)
plt.show()

knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(x_train, np.ravel(y_train))
print("Knn: ", knn.score(x_test, y_test))
cm = confusion_matrix(y_test, knn.predict(x_test))
sns.heatmap(cm, annot=True)

cr = classification_report(y_test, knn.predict(x_test))
print(cr)
plt.show()

rfc = RandomForestClassifier(max_depth=125, random_state=0)
rfc.fit(x_train, np.ravel(y_train))
print("Random Forest Classifier: ", rfc.score(x_test, y_test))

cm = confusion_matrix(y_test, rfc.predict(x_test))
sns.heatmap(cm, annot=True)
cr = classification_report(y_test, rfc.predict(x_test))
print(cr)
plt.show()
