# import pandas as pd

# df = pd.read_csv("data/processed/feature_engineered.csv")

# print(df.head())
# print(X.columns.tolist())
# X = df.drop("cloud_cost", axis=1)
# y = df["cloud_cost"]
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42
# )
# from sklearn.linear_model import LinearRegression

# model = LinearRegression()

# model.fit(X_train, y_train)
# predictions = model.predict(X_test)
# from sklearn.metrics import (
#     mean_absolute_error,
#     mean_squared_error,
#     r2_score
# )

# mae = mean_absolute_error(y_test, predictions)

# rmse = mean_squared_error(
#     y_test,
#     predictions
# ) ** 0.5

# r2 = r2_score(y_test, predictions)

# print("MAE:", mae)
# print("RMSE:", rmse)
# print("R2:", r2)
# import joblib

# joblib.dump(
#     model,
#     "models/cloud_cost_model.pkl"
# )

# print("Model saved successfully!")
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# --------------------------------------------------
# Load Feature Engineered Dataset
# --------------------------------------------------

df = pd.read_csv(
    "data/processed/feature_engineered.csv"
)

print("Dataset Loaded Successfully\n")

print("First 5 Rows:")
print(df.head())

# --------------------------------------------------
# Features and Target
# --------------------------------------------------

X = df.drop("cloud_cost", axis=1)
y = df["cloud_cost"]

print("\nFeatures Used For Training:")
print(X.columns.tolist())

# --------------------------------------------------
# Train Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Rows:", len(X_train))
print("Testing Rows:", len(X_test))

# --------------------------------------------------
# Train Model
# --------------------------------------------------

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

print("\nModel Training Completed")

# --------------------------------------------------
# Predictions
# --------------------------------------------------

predictions = model.predict(X_test)

# --------------------------------------------------
# Evaluation
# --------------------------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = (
    mean_squared_error(
        y_test,
        predictions
    ) ** 0.5
)

r2 = r2_score(
    y_test,
    predictions
)

print("\nModel Performance")

print("MAE :", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2  :", round(r2, 4))

# --------------------------------------------------
# Save Model
# --------------------------------------------------

joblib.dump(
    model,
    "models/cloud_cost_model.pkl"
)

print("\nModel Saved Successfully")
print(
    "Location: models/cloud_cost_model.pkl"
)