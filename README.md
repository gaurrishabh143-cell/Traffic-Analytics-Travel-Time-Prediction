# Traffic Analytics and Travel Time Prediction

## Project Overview

Traffic congestion is a major problem in urban areas. This project analyzes traffic data and uses Machine Learning to predict travel time based on different traffic-related factors.

The project performs traffic data analysis, creates visualizations, and develops a Machine Learning model for travel time prediction.

## Objectives

- Analyze traffic volume patterns.
- Analyze average vehicle speed.
- Study travel time variations.
- Understand the relationship between traffic volume and travel time.
- Understand the relationship between speed and travel time.
- Build a Machine Learning model to predict travel time.
- Evaluate the performance of the prediction model.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- GitHub

## Dataset

The dataset contains traffic-related information such as:

- Date
- Hour
- Day
- Traffic Volume
- Average Speed
- Weather
- Distance
- Travel Time

## Project Structure

```text
Traffic-Analytics-Travel-Time-Prediction/
│
├── dataset/
│   └── traffic_data.csv
│
├── graphs/
│   ├── traffic_by_hour.png
│   ├── travel_time_by_hour.png
│   ├── traffic_vs_travel_time.png
│   ├── speed_vs_travel_time.png
│   └── actual_vs_predicted.png
│
├── notebooks/
│
├── src/
│   ├── traffic_analysis.py
│   └── travel_time_prediction.py
│
├── README.md
└── requirements.txt
## Model Performance

The Random Forest Regression model achieved the following results on the test dataset:

- Mean Absolute Error (MAE): 1.08 minutes
- Mean Squared Error (MSE): 1.56
- R² Score: 0.98

The model was also used to predict travel time for a sample traffic condition. The predicted travel time for the example input was 44.46 minutes.