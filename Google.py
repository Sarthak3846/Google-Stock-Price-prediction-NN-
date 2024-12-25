import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense,Dropout,LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 

df = pd.read_csv("C:\\Users\\Sarthak Tyagi\\Downloads\\GOOG.csv")
df = df.drop('Price', axis='columns')
df = df.drop(df.index[0]).reset_index(drop=True)

X = df.drop('Close', axis='columns')
y = df['Close']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test = scaler.transform(y_test.values.reshape(-1, 1)).flatten()

model = Sequential()

model.add(Dense(units=128,activation='relu',input_dim=X_train.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(units=64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='linear'))

model.compile(
    optimizer='adam',
    loss='mean_squared_error')

model.fit(X_train,y_train,epochs=10,batch_size=32,validation_data=(X_test, y_test))

predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.figure(figsize=(10, 5))
plt.plot(actual_prices, color='blue', label='Actual Prices')
plt.plot(predicted_prices, color='red', label='Predicted Prices')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

