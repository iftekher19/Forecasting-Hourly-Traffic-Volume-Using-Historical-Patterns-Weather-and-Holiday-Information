# Traffic Volume Forecasting Using Machine Learning

## Project Overview
This project focuses on forecasting hourly traffic volume using historical traffic patterns, weather conditions, and holiday information.

The project uses the Metro Interstate Traffic Volume dataset and explores time-series forecasting techniques and machine learning approaches.

## Technologies Used
- Python 3.11
- Pandas and NumPy
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Matplotlib

## Methodology
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Missing-value treatment
- Time-based and lag feature engineering
- Chronological train-validation-test splitting
- Baseline forecasting
- XGBoost model training and evaluation

## Results

| Metric | XGBoost |
|---|---:|
| MAE | 130.46 |
| RMSE | 202.48 |
| R² | 0.9896 |
| MASE | 0.2272 |

XGBoost achieved approximately 47.1% lower MAE than the strongest baseline model.

## Dataset
Metro Interstate Traffic Volume dataset:
https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume

## Project Status
Phase 4 evaluation completed. LSTM and SARIMAX final test evaluations remain incomplete.

## Author
Iftekher Ahmed

CSE437 Academic Research Project
