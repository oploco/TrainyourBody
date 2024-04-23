from joblib import load
import numpy as np
import pandas as pd
import os

class MyModel:
    def predict(self, Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp):
        # Cargar el modelo entrenado
        model = load('./ML/Data/modelo_entrenado_Calories.joblib')

        feature_names = ['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
        features = np.array([[Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp]])
        features_df = pd.DataFrame(features, columns=feature_names)

        prediction = model.predict(features_df).reshape(1, -1)
        print('ppp',prediction[0][0])
        return prediction[0][0]