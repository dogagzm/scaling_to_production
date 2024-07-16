# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import shap

# Load the data
file_path = 'c:/Users/Furka/PycharmProjects/scaling_to_production/05_src/data/fires/forestfires.csv'
fires_dt = pd.read_csv(file_path)

# Initial data inspection
print(fires_dt.head())
print(fires_dt.dtypes)

# Data cleaning
# Clean up string values like 'X' and 'Y' and convert to float
# Rename columns like 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain'
fires_dt.columns = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area']

# Rename 'X' and 'Y' columns to 'coord_x' and 'coord_y'
fires_dt.rename(columns={'X': 'coord_x', 'Y': 'coord_y'}, inplace=True)

# Cleaning operations
# Convert columns to float data type
for column in ['coord_x', 'coord_y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area']:
    fires_dt[column] = pd.to_numeric(fires_dt[column], errors='coerce')

# Final data state
print(fires_dt.head())
print(fires_dt.dtypes)

# Separate features and target variable
X = fires_dt.drop('area', axis=1)
y = fires_dt['area']

# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Pipeline A
pipeline_a = Pipeline(steps=[
    ('preprocessing', ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), ['coord_x', 'coord_y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['month', 'day'])
        ]
    )),
    ('regressor', KNeighborsRegressor())
])

# Define parameter grid for Pipeline A
param_grid_a = {
    'regressor__n_neighbors': [3, 5, 7, 9]
}

# Perform hyperparameter tuning with GridSearchCV
grid_search_a = GridSearchCV(pipeline_a, param_grid_a, cv=5, n_jobs=-1, verbose=2)
grid_search_a.fit(X_train, y_train)

# Evaluate the performance of the best model
best_pipeline_a = grid_search_a.best_estimator_
y_pred_train_a = best_pipeline_a.predict(X_train)
y_pred_test_a = best_pipeline_a.predict(X_test)
print("Pipeline A - RMSE:", np.sqrt(mean_squared_error(y_train, y_pred_train_a)))
print("Pipeline A - R^2:", r2_score(y_train, y_pred_train_a))
print("Best pipeline A:", best_pipeline_a)

# Define Pipeline B
pipeline_b = Pipeline(steps=[
    ('preprocessing', ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer()),  # Changed from median to mean
                ('scaler', StandardScaler())
            ]), ['coord_x', 'coord_y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['month', 'day'])
        ]
    )),
    ('regressor', KNeighborsRegressor())
])

# Define parameter grid for Pipeline B
param_grid_b = {
    'regressor__n_neighbors': [5, 7, 9, 11]
}

# Perform hyperparameter tuning with GridSearchCV
grid_search_b = GridSearchCV(pipeline_b, param_grid_b, cv=5, n_jobs=-1, verbose=2)
grid_search_b.fit(X_train, y_train)

# Evaluate the performance of the best model
best_pipeline_b = grid_search_b.best_estimator_
y_pred_train_b = best_pipeline_b.predict(X_train)
y_pred_test_b = best_pipeline_b.predict(X_test)
print("Pipeline B - RMSE:", np.sqrt(mean_squared_error(y_train, y_pred_train_b)))
print("Pipeline B - R^2:", r2_score(y_train, y_pred_train_b))
print("Best pipeline B:", best_pipeline_b)

# SHAP analysis
# Use the unprocessed version of X_train
X_train_transformed = best_pipeline_b.named_steps['preprocessing'].transform(X_train)

# Create a SHAP explainer
explainer = shap.KernelExplainer(best_pipeline_b.named_steps['regressor'].predict, X_train_transformed)

# Compute SHAP values
shap_values = explainer.shap_values(X_train_transformed)

# SHAP summary plot
shap.summary_plot(shap_values, X_train_transformed, feature_names=X_train.columns)

# Create a force plot for a sample data point
shap.force_plot(explainer.expected_value, shap_values[0], X_train_transformed[0], feature_names=X_train.columns)

# Initialize SHAP JavaScript visualization support
shap.initjs()
