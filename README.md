# Smart_Room

## Overview

This repository contains the Python code for an intelligent system that aims at optimizing the power usage in a building's various rooms. It takes into account the energy consumption patterns, room vacancy rates, and other specific room centric details to provide meaningful insights and suggestions to improve energy usage efficiency. The system also provides a feedback mechanism to adjust settings of various devices (like lights and heating systems) in a room to optimize power usage.

The data analysis phase includes handling missing data, scaling numerical values, encoding categorical values, and visualizing outliers, which together aid the subsequent data modelling. A variety of regression models, from basic ones like linear regression to more sophisticated types like gradient boosting regressors, are employed for prediction. Other advanced methods such as SVM, KNN, AdaBoost, and XGBoost are covered as well.

The temperature control is predicted using the OpenWeatherMap API and a comparison is made between room temperature and outside temperature based on vacancy. 

In addition to that we have used advanced image detection method to track the waste in an office area and simultaneously send alerts to users in a fun way to pick this waste up to reduce the after effects of improper waste management along with educating them about recyclable or non recyclable waste

## Contents

- `Garbage_Dataset.md`: Information about the garbage dataset.
- `Garbage_Detection.py`: Code for waste detection.
- `LICENSE`: The project's license information.
- `README.md`: This file, providing an overview of the project.
- `devices_power.csv`: Dataset containing device power consumption data.
- `smartroom.py`: Python script for Smart Room Energy Optimization.
- `weather.py`: Python script for weather data retrieval .

## Prerequisites

This project requires Python along with the following Python libraries installed:

- Pandas
- Numpy
- Matplotlib
- datetime
- sklearn
- XGBoost

You can install these libraries using pip:

`pip install pandas numpy matplotlib datetime sklearn xgboost`

In case you're using a Jupyter notebook, make sure it's installed. If not, install it using pip:

`pip install jupyter`


## Project Content / Use of Data Science for 
The scripts in this repository provide functionality for:

1. Handling missing data, scaling numerical values, encoding categorical values.
2. Visualizing outliers and understanding distributions.
3. Performing time series analysis on power usage.
4. Calculating daily averages of power usage and daily occupancy rates.
5. Daily Energy Efficiency Ratio and peak power usage calculation.
6. Data modeling and prediction on power usage using various regression models.
7. Evaluating model performance using Mean Squared Error (MSE).
8. Adjusting various device settings in a room to optimize power usage.
9. The scripts are commented and divided into sections that provide different functionalities.

The final part of the script optimizes the settings for power usage in different rooms using various machine learning models. With this system, you can obtain recommendations to adjust the lighting and heating systems in the room to optimize energy usage.

## Output

## Conclusion
This project is merely a baseline. As more data is accumulated, we can refine our model to improve our prediction accuracy and make the Smart Room Energy Optimization system even more effective. Continuous model training and implementation of other AI practices can make this project more sustainable and helpful.

