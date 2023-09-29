import requests
from pprint import pprint
from datetime import datetime

def get_weather(city):
    params = {
        'q': city,
        'appid': '3c1cb747915411442c58c8ba08353754',
        'units': 'metric'
    }
    url = 'https://api.openweathermap.org/data/2.5/weather'
    response = requests.get(url, params=params)
    return response.json()

def format_weather(weather):
    timestamp = datetime.fromtimestamp(weather['dt'])
    temperature = weather['main']['temp']
    wind_speed = weather['wind']['speed']
    feels_like = weather['main']['feels_like']
    main_weather = weather['weather'][0]['main']
    description = weather['weather'][0]['description']

    print(f"Timestamp: {timestamp}, Temperature: {temperature}°C, Wind Speed: {wind_speed} m/s, Feels Like: {feels_like}°C, Weather: {main_weather}, Description: {description}")


city = input('Enter your city: ')
weather_data = get_weather(city)
print(f"Current Weather for {city}:")
format_weather(weather_data)


def thermostat(ac_on, acPower, temp_inside, temp_outside, temp_desired, start, end, t):
    # Current time in seconds
    curr_time = datetime.now().hour*3600 + datetime.now().minute*60 + datetime.now().second

    # Checking if the current time is within the interval
    if start <= curr_time <= end:
        ac_on = False
        print("The system is off according to the programmed time.")
    else:
        diff_temp = abs(temp_inside - temp_outside)
        if diff_temp > 5:  # You can set the value according to your preference
            print("Consider adjusting your temperature settings. The difference between indoor and outdoor temperatures is significant.")

        # Checking whether the inside temperature is more than desired + hysteresis 
        if temp_inside > temp_desired + 2:  
            ac_on = True
            ac_heatflow = acPower
        elif temp_inside < temp_desired -2:  # Checking whether the inside temperature is less than desired - hysteresis 
            ac_on = False
            ac_heatflow = 0
        else:
            ac_heatflow = 0 if not ac_on else acPower
    
    return ac_heatflow, ac_on
