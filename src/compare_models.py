import pandas as pd

df = pd.read_csv("data/processed/feature_engineered.csv")

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
from sklearn.metrics import mean_absolute_error, r2_score

lr = LinearRegression()

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("Linear Regression")
print("MAE:",
      mean_absolute_error(y_test, lr_pred))
print("R2:",
      r2_score(y_test, lr_pred))

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\nRandom Forest")

print(
    "MAE:",
    mean_absolute_error(y_test, rf_pred)
)

print(
    "R2:",
    r2_score(y_test, rf_pred)
)
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print(feature_importance)

import joblib

joblib.dump(
    rf,
    "models/best_model.pkl"
)
print("Best model saved successfully!")