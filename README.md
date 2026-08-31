# 🌍 Environmental Monitoring & Air Quality Intelligence

A Data Science and Machine Learning project for analyzing air pollution patterns and predicting the Air Quality Index (AQI) using pollutant measurements from multiple Indian cities.

## 📌 Project Overview

Air pollution is an important environmental and public health concern. This project analyzes historical air-quality data from 26 Indian cities and develops a machine learning system to predict AQI based on atmospheric pollutant concentrations.

The project combines:

- Exploratory Data Analysis
- Data Cleaning
- Missing Value Treatment
- Statistical Analysis
- Machine Learning
- Hyperparameter Tuning
- Feature Importance Analysis
- Interactive Data Visualization
- Streamlit Dashboard

## 📊 Dataset

The dataset contains **29,531 records** covering the period:

**2015 – 2020**

It includes air-quality measurements from **26 Indian cities**.

### Pollutant Features

- PM2.5
- PM10
- NO
- NO2
- NOx
- NH3
- CO
- SO2
- O3
- Benzene
- Toluene
- Xylene

### Target Variable

**AQI — Air Quality Index**

The dataset also contains:

**AQI_Bucket — Air Quality Category**

## 🔎 Exploratory Data Analysis

The project performs several EDA analyses:

- Dataset structure and statistics
- Missing-value analysis
- AQI distribution
- AQI category distribution
- City-wise average AQI
- Pollutant correlation analysis
- Feature importance
- Actual vs predicted AQI
- Prediction error analysis
- Historical AQI trends
- Pollutant trends

## 🧹 Data Preprocessing

Rows with missing AQI values were removed because AQI is the target variable for the prediction task.

Pollutant missing values were handled using **median imputation**.

This resulted in a clean modeling dataset containing:

**24,850 records and 12 pollutant features.**

## 🤖 Machine Learning

Several regression models were evaluated:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 31.17 | 59.44 | 0.8070 |
| Ridge Regression | 31.17 | 59.44 | 0.8070 |
| Gradient Boosting | 23.59 | 43.56 | 0.8964 |
| Random Forest | 20.76 | 40.68 | 0.9096 |
| **Tuned Random Forest** | **20.72** | **40.09** | **0.9122** |

### 🏆 Final Model

The **Tuned Random Forest Regressor** was selected as the final model.

Performance:

- **R²:** 0.9122
- **MAE:** 20.72
- **RMSE:** 40.09

The model explains approximately **91.22% of the variation in AQI** on the test dataset.

## 🔍 Feature Importance

The most important features identified by the Random Forest model were:

| Feature | Importance |
|---|---:|
| PM2.5 | 30.42% |
| CO | 25.84% |
| PM10 | 10.27% |
| SO2 | 6.55% |
| NO | 6.34% |
| NO2 | 5.80% |
| NOx | 5.24% |

PM2.5 and CO were the two most influential features in the trained model.

## 🌐 Streamlit Application

The project includes an interactive Streamlit dashboard.

### Dashboard Features

- 🌫️ AQI prediction
- 📊 Pollutant input visualization
- 🏙️ City selection
- 📈 Historical AQI trends
- 🌫️ Pollutant trend analysis
- 🏆 Model performance metrics
- 💡 Environmental insights
- 🌈 Color-coded AQI categories

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels
- Joblib
- Streamlit
- Jupyter Notebook

## 📁 Project Structure

```text
Environmental_Monitoring_Air_Quality/
│
├── app/
│   └── app.py
│
├── data/
│   └── air_quality_city_day.csv
│
├── models/
│   ├── final_aqi_model.pkl
│   └── aqi_feature_columns.pkl
│
├── notebooks/
│   └── 01_Air_Quality_EDA.ipynb
│
├── outputs/
│
├── README.md
└── requirements.txt