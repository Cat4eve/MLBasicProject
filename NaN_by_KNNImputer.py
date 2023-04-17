import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import pandas as pd
df = pd.read_csv('./hospital_deaths_train.csv')

target = 'In-hospital_death'
X = df.drop([target,'recordid'],axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
imputer = KNNImputer(n_neighbors=5)
imputed_train_data = imputer.fit_transform(X_train, y_train)
clf2 = KNeighborsClassifier(n_neighbors=5)
clf2.fit(imputed_train_data, y_train)
imputed_test_data = imputer.fit_transform(X_test,y_test)

##just to check working
clf2.predict(imputed_test_data)

print(clf2.score(imputed_test_data,y_test))
print(clf2.score(imputed_train_data,y_train))
