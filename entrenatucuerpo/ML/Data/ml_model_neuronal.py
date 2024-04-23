
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.losses import mean_squared_error

from joblib import dump

PATH = "Physical Movements Calories counting/"

exercise = pd.read_csv(PATH+"exercise.csv")
calories= pd.read_csv(PATH+"calories.csv")

exercise.head()
calories.head()

# join df
#calories_df = pd.concat([exercise.drop('User_ID', axis=1), calories['Calories']], axis=1)
exercise_df = exercise.drop('User_ID', axis=1)
calories_df = calories.drop('User_ID', axis=1)

print(exercise_df['Gender'][0])
#calories_df = calories_df.drop('Calories', axis=1)

data =exercise_df
print(data.head())

# Preprocesar datos
# Codificar variables categóricas
encoder = OneHotEncoder(categories=[['male', 'female']])
encoded_gender = encoder.fit_transform(data[['Gender']]).toarray()
print('encoded_gender 1', encoded_gender)

# Escalar características numéricas
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']])
print('sc1 ',scaled_features)

# Combinar características codificadas y escaladas
X = np.concatenate((encoded_gender, scaled_features),axis=1)
print('X ',X[0])

# Crear conjunto de etiquetas
y = calories_df
#y = scaler.fit_transform(calories_df[['Calories']])

# Dividir datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(y)
print ("qqq", X_train.data.shape)
# Definir modelo de red neuronal
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compilar modelo
model.compile(optimizer='adam', loss=mean_squared_error)
print(model.summary())

# Entrenar modelo
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluar modelo
loss = model.evaluate(X_test, y_test)
print('loss', loss)

y_pred = model.predict(X_test)
print('y_pred ',y_pred[0])
print(X_test[0])

# Guardar el modelo entrenado
model.save("modelo_entrenado_Cals.h5")

Gender = 'male'
Age = 45.0
Height = 178.0
Weight = 90.0
Duration = 100.0
Heart_Rate = 165.0
Body_Temp = 40.0

# Crear un diccionario con las variables
data_dict = {
    'Gender': [Gender],  # Codificación one-hot para male: 1
    'Age': [Age],
    'Height': [Height],
    'Weight': [Weight],
    'Duration': [Duration],
    'Heart_Rate': [Heart_Rate],
    'Body_Temp': [Body_Temp]
}

# Convertir el diccionario en un DataFrame de Pandas
data = pd.DataFrame(data_dict)
data.head()
print(data.head())

# Preprocesar datos
# Codificar variables categóricas
encoder = OneHotEncoder(categories=[['male', 'female']])
#encoded_gender = encoder.fit_transform([[Gender]]).toarray()
encoded_gender = encoder.fit_transform(data[['Gender']]).toarray()
print('encoded_gender',encoded_gender)

# Escalar características numéricas
scaler = StandardScaler()
# Escalar características numéricas
#scaled_features = scaler.fit_transform([[Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp]])
scaled_features = scaler.fit_transform(data[['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']])
print('sc ', scaled_features)

# Combinar características codificadas y escaladas
features = np.hstack((encoded_gender, scaled_features))
print('features ',features)

# Hacer predicciones
y_pred = model.predict(features)  # Predicciones
print(y_pred)
