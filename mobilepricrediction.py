from os import pipe
import pandas as pd
import numpy as np


dataset=pd.read_csv('train.csv')


X=dataset.drop('price_range',axis=1)
y=dataset['price_range']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)

knn.score(X_test,y_test)

error_rate = []
for i in range(1,20):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# Price prediction of Test.csv Using KNN for Prediction
### Import test.csv
# data_test=pd.read_csv('test.csv')
# data_test.head()
# data_test=data_test.drop('id',axis=1)
# data_test.head()

# # Model
# predicted_price=knn.predict(data_test)
# # Predicted Price Range
# print(predicted_price)
# # Adding Predicted price to test.csv
# data_test['price_range']=predicted_price
# print(data_test)

arr=np.array([[1600,1,12,1,12,1,512,1,100,4,15,1566,1453,512,8,9,30,1,1,1]])
print(knn.predict(arr))

import pickle

pickle.dump(knn, open('knn.pkl', 'wb'))

