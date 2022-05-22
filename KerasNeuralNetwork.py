import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_test = pd.read_csv('churn-bigml-20.csv')
df_train=pd.read_csv('churn-bigml-80.csv')
df_final=df_train.append(df_test)
pd.get_dummies(df_final,columns=['International plan'],drop_first=True).head()
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
df_final['State']=labelencoder.fit_transform(df_final['State'])
df_final=pd.get_dummies(df_final,columns=['International plan','Voice mail plan','Churn'],drop_first=True)
df_final['State']=labelencoder.fit_transform(df_final['State'])
df_final.drop('Account length', axis=1, inplace=True)
X = df_final.drop('Churn_True', axis=1)

y = df_final['Churn_True']
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

import tensorflow as tf
from keras.models import Sequential
import pandas as pd
from keras.layers import Dense

model = Sequential()
#model.add(Dense(8, activation='relu', input_shape=(16,)))
model.add(tf.keras.layers.Dense(256, input_shape=(X_train.shape[1],), activation='sigmoid'))
model.add(Dense(8, activation='relu'))
#model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
optimizer='sgd',
metrics=['accuracy'])
model.fit(X_train, y_train,epochs=8, batch_size=1, verbose=1)
y_pred = model.predict(X_test)
score = model.evaluate(X_test, y_test,verbose=1)
print(score)
y_pred = model.predict(X_test)
score = model.evaluate(X_test, y_test,verbose=1)
print(score)
