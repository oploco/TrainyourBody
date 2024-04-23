import numpy as np
import pandas as pd

from joblib import load

# function
def pred_calories(Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp):
    # Cargar el modelo entrenado
    model = load('modelo_entrenado_Calories.joblib')

    feature_names = ['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
    features = np.array([[Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp]])
    features_df = pd.DataFrame(features, columns=feature_names)

    prediction = model.predict(features_df).reshape(1,-1)
    return prediction[0]

def pred_duration(Gender,Age,Height,Weight,Heart_Rate,Body_Temp,Calories):
    # Cargar el modelo entrenado
    model = load('modelo_entrenado_Duration.joblib')

    feature_names = ['Gender', 'Age', 'Height', 'Weight', 'Heart_Rate', 'Body_Temp', 'Calories']
    features = np.array([[Gender,Age,Height,Weight,Heart_Rate,Body_Temp,Calories]])
    features_df = pd.DataFrame(features, columns=feature_names)

    prediction = model.predict(features_df).reshape(1,-1)
    return prediction[0]

if __name__ == "__main__":
    # data for predict calories
    # 'male': 0, 'female': 1
    Gender = 0
    Age = 45
    Height = 178.0
    Weight = 90.0
    Duration = 45.0
    Heart_Rate = 160.0
    Body_Temp = 40.8
    # BMI = (Weight*10000)/(Height**2) out

    Calories = 500.0

    # predict Calories
    result = pred_calories(Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp)
    print("calories predicted",result)

    # predict Duration
    result = pred_duration(Gender, Age, Height, Weight, Heart_Rate, Body_Temp, Calories)
    print("Duration predicted", result)

    # predict Duration
    #result = pred_cals(Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp)
    #print("Duration predicted", result)


