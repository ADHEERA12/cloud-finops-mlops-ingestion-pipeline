import pandas as pd

df = pd.read_csv("data/processed/feature_engineered.csv")

print(df.head())
X = df.drop("cloud_cost", axis=1)
y = df["cloud_cost"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)
predictions = model.predict(X_test)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

mae = mean_absolute_error(y_test, predictions)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)
import joblib

joblib.dump(
    model,
    "models/cloud_cost_model.pkl"
)

print("Model saved successfully!")