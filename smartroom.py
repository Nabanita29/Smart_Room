import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Read the dataset
dataset = pd.read_csv('/content/devices_power.csv', encoding='latin1')

# Convert 'Time Stamp' to datetime
dataset['Time Stamp'] = pd.to_datetime(dataset['Time Stamp'], format='%d-%m-%Y %H:%M')

# Handle missing values (if any)
# For example, filling missing values with zeros for 'Power used' and False for 'Vacancy'
dataset['Power used'].fillna(0, inplace=True)
dataset['Vacancy'].fillna(False, inplace=True)

# Check data types
print(dataset.dtypes)

# Separate numerical and categorical features
numerical_features = ['Power used', 'Vacancy']
categorical_features = ['Meter Type', 'Building', 'Room']

# Scale numerical features
scaler = MinMaxScaler()
scaled_numerical = scaler.fit_transform(dataset[numerical_features])

# One-hot encode categorical features
encoder = OneHotEncoder(sparse=False, drop='first')  # drop='first' removes the first category to avoid multicollinearity
encoded_categorical = encoder.fit_transform(dataset[categorical_features])

# Concatenate scaled numerical and encoded categorical features
X = np.hstack((scaled_numerical, encoded_categorical))

# Checking for outliers (boxplot for specific columns)
plt.figure(figsize=(8, 6))
plt.boxplot([dataset['Power used'], dataset['Vacancy']])
plt.xticks([1, 2], ['Power used', 'Vacancy'])
plt.title('Boxplot for Outliers')
plt.show()

# Summary statistics
print(dataset.describe())

# Time series analysis
plt.figure(figsize=(12, 6))
plt.plot(dataset['Time Stamp'], dataset['Power used'], label='Power Used', marker='o')
plt.xlabel('Time')
plt.ylabel('Power Used')
plt.title('Power Usage Over Time')
plt.legend()
plt.show()

# Histogram for Power used
plt.figure(figsize=(8, 6))
plt.hist(dataset['Power used'], bins=20)
plt.xlabel('Power Used')
plt.ylabel('Frequency')
plt.title('Histogram of Power Used')
plt.show()

# Correlation matrix
correlation_matrix = dataset[['Power used', 'Vacancy']].corr()
print(correlation_matrix)

# Average Power Usage per day
average_power_per_day = dataset.groupby(dataset['Time Stamp'].dt.date)['Power used'].mean()

# Occupancy Rate
occupancy_rate = (dataset['Vacancy'].sum() / len(dataset)) * 100

# Energy Efficiency Ratio
energy_efficiency_ratio = dataset['Power used'].sum() / dataset['Vacancy'].sum()

# Peak Power Demand
peak_power_demand = dataset['Power used'].max()

# Energy Cost Savings (if applicable)
# You would need energy cost data to calculate this metric
energy_cost_savings = ...

print("Average Power Usage per day:", average_power_per_day)
print("Occupancy Rate:", occupancy_rate, "%")
print("Energy Efficiency Ratio:", energy_efficiency_ratio)
print("Peak Power Demand:", peak_power_demand)
print("Energy Cost Savings:", energy_cost_savings)

# Extract time-based features
dataset['Hour'] = dataset['Time Stamp'].dt.hour
dataset['DayOfWeek'] = dataset['Time Stamp'].dt.dayofweek
dataset['Month'] = dataset['Time Stamp'].dt.month

# Create binary features based on keywords in Description
dataset['Has_Computer_Systems'] = dataset['Description'].str.contains('computer systems', case=False)
dataset['Has_Lights'] = dataset['Description'].str.contains('lights', case=False)

# Calculate daily average power usage
daily_avg_power_usage = dataset.groupby(dataset['Time Stamp'].dt.date)['Power used'].mean()

# Calculate daily occupancy rate
daily_occupancy_rate = dataset.groupby(dataset['Time Stamp'].dt.date)['Vacancy'].mean()

# Create an interaction feature
dataset['Power_Usage_x_Occupancy'] = dataset['Power used'] * dataset['Vacancy']


from sklearn.model_selection import train_test_split

# Assuming 'Power used' is the target variable
X_train, X_test, y_train, y_test = train_test_split(X, dataset['Power used'], test_size=0.2, random_state=42)

# Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Train the model
reg = LinearRegression().fit(X_train, y_train)

# Make predictions
y_pred = reg.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error : {mse}")

# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor

# Train the model
tree = DecisionTreeRegressor().fit(X_train, y_train)

# Make predictions
y_pred = tree.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error : {mse}")

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

# Train the model
forest = RandomForestRegressor().fit(X_train, y_train)

# Make predictions
y_pred = forest.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error : {mse}")


from sklearn import svm

# Train the model
model = svm.SVR().fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error : {mse}")


from sklearn.ensemble import GradientBoostingRegressor

# This function will provide current settings for the room
def get_current_state(room):
    current_state = dataset.loc[dataset['Room'] == room].values[-1]  # Gets the last row of the room's data
    return current_state

class Optimizer:
    def __init__(self, model):
        self.model = model

    # Predict next power usage
    def predict_power_usage(self, x):
        return self.model.predict(x)

    # Feedback system to adjust settings for light and heating various rooms 
    def adjust_settings(self, prediction, current_state, vacancy, room):
            UPPER_LIMIT = 800   # Define upper and lower
            LOWER_LIMIT = 200     # limit based on computations

            if prediction > UPPER_LIMIT and vacancy > 0:
                return "Decrease Energy Usage of lighting and heating systems in Room " + str(room)

            elif prediction < LOWER_LIMIT and vacancy < 10:
                return "Increase Energy Usage of lighting and heating systems in Room " + str(room)

            else:
                return "Maintain Energy Usage of lighting and heating systems in Room " + str(room)

# Assuming 'Power used' is the target variable
X_train, X_test, y_train, y_test = train_test_split(X, dataset['Power used'], test_size=0.2, random_state=42)

# Gradient Boosting Regressor
model = GradientBoostingRegressor().fit(X_train, y_train)

# Call optimizer
optimizer = Optimizer(model)

rooms = dataset['Room'].unique()

for room in rooms:
    current_state = get_current_state(room)  # Function to get current state
    prediction = optimizer.predict_power_usage(current_state)
    vacancy = current_state[7]  # Index for 'Vacancy'
    action = optimizer.adjust_settings(prediction, current_state, vacancy, room)
    print(f"Predicted Power Usage: {prediction}, Recommended Action: {action}")
