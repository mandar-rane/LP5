import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

df = pd.read_csv('HousingData.csv')

imputer = SimpleImputer(strategy="mean")
df = imputer.fit_transform(df)
df = pd.DataFrame(df)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state=42)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(5, activation='relu', input_shape = X_train[0].shape))
model.add(tf.keras.layers.Dense(1))
model.summary()

model.compile(optimizer='sgd', loss = 'mse')

model.fit(X_train, y_train, epochs=100, batch_size=32)

loss = model.evaluate(X_test, y_test, verbose=0)
print("Mean Squared Error on Test Data:", loss)


y_test[41]

model.predict(np.reshape(X_test[41], [1, 13]))