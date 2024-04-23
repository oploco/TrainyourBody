
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn import metrics
from joblib import dump

PATH = "Physical Movements Calories counting/"

def modelar(vFeature):
    exercise = pd.read_csv(PATH+"exercise.csv")
    calories= pd.read_csv(PATH+"calories.csv")

    exercise.head()
    calories.head()

    # join df
    calories_df = pd.concat([exercise.drop('User_ID',axis=1),calories['Calories']],axis=1)
    calories_df

    # gender to numeric values
    gender_map = {'male': 0, 'female': 1}
    calories_df['Gender'] = calories_df['Gender'].map(gender_map)

    # separo para features
    X = calories_df.drop(vFeature,axis=1)
    y = calories_df[vFeature]

    print(X)
    print(y)

    # Split groups
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    models = {
        'lr':LinearRegression(),
        "rfr":RandomForestRegressor(),
        'dtr':DecisionTreeRegressor(),
        'xg':XGBRFRegressor()
    }

    for name, mod in models.items():
        mod.fit(X_train,y_train)
        ypred = mod.predict(X_test)
        print ("ypred " , ypred)
        mae = metrics.mean_absolute_error(y_test, ypred)
        print('mae ', mae  )
        print(f"{name}  mse: {(mean_squared_error(y_test,ypred)):.2f} r2 score: {(r2_score(y_test,ypred)):.2f}")
        print()

    # aplico el RandomForestRegressor por mse
    model = LinearRegression()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    # Guardar el modelo entrenado
    nombre_archivo = f"modelo_entrenado_{vFeature}.joblib"
    dump(model, nombre_archivo)

    print("ypred ", ypred)



if __name__ == "__main__":
    modelar("Calories")
    modelar("Duration")