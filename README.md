# Smart_Room
Second position GeeksForGeeks Ecotech Hackathon
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


## Project Content / Use of Data Science for Electricity control
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

## Output

![image](https://github.com/Nabanita29/Smart_Room/assets/107246882/f7400565-806a-4408-8d2e-2dab667e5653)     
![image](https://github.com/Nabanita29/Smart_Room/assets/107246882/55291d6d-2a8f-4885-b491-0fe9330ab273)       
![image](https://github.com/Nabanita29/Smart_Room/assets/107246882/3d0ddf84-1037-43d1-96f1-e91251c7c54c)               
![image](https://github.com/Nabanita29/Smart_Room/assets/107246882/fda7e058-80f7-4a0e-902f-ebf2d42cfb14)      
![image](https://github.com/Nabanita29/Smart_Room/assets/107246882/085103a2-a004-49d7-962f-b1c2ca626c89)


## Use of Data Science for waste detection

The project includes a machine learning component for waste detection. It leverages pre-trained models and image classification techniques to identify different types of waste items. Here's a brief overview of this component:

### Machine Learning Model

We use a pre-trained VGG16 model, fine-tuned for waste detection, to classify waste items into various categories. The model is trained on a dataset containing images of waste items.

### Usage

To use the waste detection model, you can follow these steps:

1. Prepare an image containing the waste item you want to classify.

2. Load the image using the `waste_prediction` function, providing the file path to the image.

3. The model will process the image and provide a prediction for the type of waste detected.

### Example


https://github.com/Nabanita29/Smart_Room/assets/107246882/848d8926-c6d5-4e62-9726-b195ea383772

## Conclusion

The Smart Room Energy Optimization project serves as a foundational framework for optimizing power usage within various rooms of a building. While it offers valuable insights and suggestions for improving energy efficiency based on existing data, it's important to note that this project is just the beginning of what can be achieved in the realm of smart energy management.

As we gather more data and insights, there is immense potential to refine our models and algorithms, ultimately leading to more accurate predictions and even greater energy optimization. Continuous model training, the integration of cutting-edge AI practices, and ongoing research can further enhance the capabilities of the Smart Room Energy Optimization system.

In addition to its energy optimization features, the project includes a waste detection component that uses machine learning to identify different types of waste items. This innovative addition opens doors to efficient waste management strategies within smart rooms.


## License

This project is open-source and is released under the [MIT License](LICENSE.md). You are free to use, modify, and distribute this software within the terms and conditions specified in the license.



