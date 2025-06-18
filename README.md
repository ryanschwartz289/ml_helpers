# ML Utility Functions 📊🤖

A comprehensive Python module for streamlining machine learning workflows with commonly used utilities, plotting functions, modeling pipelines, and time series handling. Perfect for fast prototyping and experimentation.

---

## 🔧 Features

### 🧰 General Utilities
- `ml_imports()`: Prints commonly used ML import statements.

### 🧪 Model Evaluation & Iteration
- `iterate_knn(...)`: KNN evaluation loop over a range of `k` values.
- `iterate_dt(...)`: Decision Tree evaluation loop over various depths.
- `iterate_rfe(...)`: RFE feature selection with scoring.
- `model_fitter(...)`: Trains and evaluates classification models.
- `regression_model_fitter(...)`: Evaluates regression models.
- `text_model_fitter(...)`: Evaluates models for text vector inputs.
- `grid_searcher(...)`: Grid search with hyperparameter tuning.

### 📈 Plotting Functions
- `plot_algorithm(...)`: Plots validation accuracy across hyperparameters.
- `plot_dt(...)`: Visualizes decision trees using `graphviz`.
- `model_barplot(...)`: Horizontal barplot of model accuracies.
- `model_heatmap(...)`: Multiple confusion matrix heatmaps.

### 📅 Time Series Support
- `make_date_features(...)`: Adds time-based features from a datetime column or index.
- `create_lags(...)`: Adds lagged values for time series.
- `xgb_time_series_split(...)`: XGBoost evaluation with time series splits and optional grid search.
- `time_series_split_predict(...)`: Generic time series CV evaluation for any model.
- `visualize_time_series_split(...)`: Visualizes train/test splits across folds.
- `create_future_df(...)`: Generates a DataFrame for forecasting with lag and time features.

---

## 📦 Dependencies

Ensure the following libraries are installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost graphviz tqdm
```
## 🧪 Example Usage
```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from ml_helpers import (
    model_fitter, make_date_features, scaled_split
)

# Load and preprocess data
df = pd.read_csv("data.csv")
df = make_date_features(df, date="timestamp", index=False)

# Prepare features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split and scale the data
X_train, X_test, y_train, y_test = scaled_split(X, y)

# Train and evaluate model
model = LogisticRegression()
model_fitter(X_train, X_test, y_train, y_test, model)
