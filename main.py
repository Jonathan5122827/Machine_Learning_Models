import inline as inline
import matplotlib
import matplotlib.pyplot as plt
import numpy as mp
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge, Lasso , RidgeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

from matplotlib import interactive
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import GaussianNB, MultinomialNB ,ComplementNB , BernoulliNB ,CategoricalNB
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
#from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

#from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

dataset = pd.read_csv("dataset01_eurusd4h.csv", index_col=False)


X = dataset.iloc[:, [2,8,14,22,21,25,38,44]].values
#dont leave 70 in, keep incrementing 
print(X)
close = dataset.iloc[:, [90]].values

greenclose = np.full_like(close, 2)
nextgreenclose = np.full_like(close, 999)


#print(dataset)
#print(close)
#print(greenclose)
index = 0
for i in close:
    #print(i, " > ",close[index-1] )
    if(close[index-1]  != None):
        if(i > close[index-1]):
            greenclose[index] = 1
        else:
            greenclose[index] = 0

#    print(greenclose[index])

    index = index + 1

index2 = 0
for i in close:
    if(close[index2+1]  != None ):
        if(close[index2+1] > i):
            nextgreenclose[index2] = 1
        else:
            nextgreenclose[index2] = 0
    print(close[index2],i,close[index2+1], nextgreenclose[index2], close[index2+1]-close[index2], close[index2]*1.00025)

    if index2 < 4477:
        index2 = index2 + 1
#    print(greenclose[index])

print(nextgreenclose[4478])
nextgreenclose[4478] = 1
print(nextgreenclose[4478])
print(nextgreenclose[4477])

#X = np.array(X).reshape(-1,1)
#greenclose = np.array(greenclose).reshape(-1,1)
x_train, x_test, y_train, y_test = train_test_split(X, nextgreenclose, test_size=0.03, random_state=0)

sc = StandardScaler()

x_train_s = sc.fit_transform(x_train)
x_test_s = sc.fit_transform(x_test)

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#y_train_s = sc.fit_transform(y_train)
#x_train_s = sc.transform(x_train)

#scaler = StandardScaler().fit(x_train)
#train_sc = scaler.transform(x_train)

#test_sc = scaler.transform(x_test)

LogReg = LogisticRegression(C=1, max_iter=100,solver='saga')

LogReg.fit(x_train_s,np.ravel(y_train))

#pred = LogReg.predict(x_test)

#test_sc = test_sc.reshape(1, -1)
#y_test = y_test.reshape(1, -1)

print("Logistic Regression:", LogReg.score(x_test_s,y_test))
#############################################
nb = GaussianNB()
nb.fit(x_train,np.ravel(y_train))
print("Gaussian Naive Bayes:",nb.score(x_test,y_test))
#################################
#lr = LogisticRegression(random_state=0)
#lr.fit(x_train,np.ravel(y_train))
#print(lr.score(x_test,y_test))
#print(greenclose)
##############################
svm = svm.SVC()
svm.fit(x_train,np.ravel(y_train))
print("Support Vector Machine: ", svm.score(x_test,y_test))
#print(svm.score(X,np.ravel(greenclose)))

###############################
dt = tree.DecisionTreeClassifier()
dt.fit(x_train,np.ravel(y_train))
print("Decision Tree Classifier:",dt.score(x_test,y_test))
###################
#for x in range(1,1000):
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

knn.fit(x_train, np.ravel(y_train))

#Ypred = knn.predict(x)

#knn.score(x_test,y_test)

print("Knn: ",knn.score(x_test,y_test))
#for x in range(1,1000):
rfc = RandomForestClassifier(max_depth=1, random_state=0)
rfc.fit(x_train, np.ravel(y_train))
print("Random Forest Classifier: ",rfc.score(x_test,y_test))

#for x in range(1,1000):
bag = BaggingClassifier(base_estimator=LogisticRegression(), n_estimators=5, random_state=0)
bag.fit(x_train,np.ravel(y_train))
print("Bagging Classifier with Logistic Regression base:",bag.score(x_test,y_test))
#print(pred)
print(nextgreenclose)
cr = classification_report(y_test,bag.predict(x_test))
print(cr)
sns.displot(nextgreenclose)

xgb = XGBClassifier(learning_rate = 1, max_depth = 20, n_estimators = 500)

Y_train = LabelEncoder().fit_transform(np.ravel(y_train))
xgb.fit(x_train,np.ravel(y_train))

print("XGBoost with no GridSearchCV", accuracy_score(y_test,xgb.predict(x_test)))

param_grid = {'learning_rate': [1,0.5,0.1,0.01,0.001],
              'max_depth': [3,5,10,20],
              'n_estimators':[10,20,50,100,200]}
param_grid2 = {
              'C': [1,2,2.2],
              'class_weight': ['balanced', None],
              'max_iter': [100,1000,10000],
              'solver':['saga', 'newton-cg', 'sag', 'liblinear', 'lbfgs']}

param_grid3 = {
              'var_smoothing': [1e-9,.9e-9,1e-8,1.1e-9]}

param_grid4 = {
              'alpha': [0.0,.5,1.0,2],
              'fit_prior': [True, False]}


grid = GridSearchCV(LogisticRegression(), param_grid2, refit = True, verbose = 3,n_jobs=-1)
grid.fit(x_train, np.ravel(y_train))

print("Best parameters: ", grid.best_params_)
print("XGBoost with GridSearchCV",accuracy_score(y_test,grid.predict(x_test)))

cr = classification_report(y_test,grid.predict(x_test))
print(cr)
plt.show()
