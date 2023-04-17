import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('./hospital_deaths_train.csv')

target = 'In-hospital_death'
X = df.drop([target,'recordid'],axis=1)
y = df[target]

q_sam, q_fet = X.shape
for i in range(q_fet):
  avg = X.iloc[:,i].mean(skipna=True)
  X.iloc[:,i] = X.iloc[:,i].fillna(avg)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)



clf1 = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
clf1.fit(X_train,y_train)

print(clf1.score(X_test,y_test))
print(clf1.score(X_train,y_train))
