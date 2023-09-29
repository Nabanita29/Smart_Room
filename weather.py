import requests
from pprint import pprint

def get_forecast(city):
    params = {
        'q': city,
        'appid': '3c1cb747915411442c58c8ba08353754',
        'units': 'metric'
    }
    url = 'https://api.openweathermap.org/data/2.5/forecast'
    response = requests.get(url, params=params)
    return response.json()

def format_forecast(forecast):
    date_time = forecast['dt_txt']
    temperature = forecast['main']['temp']
    wind_speed = forecast['wind']['speed']
    feels_like = forecast['main']['feels_like']
    weather = forecast['weather'][0]['main']
    description = forecast['weather'][0]['description']
    print(f"Date: {date_time}, Temperature: {temperature}°C, Wind Speed: {wind_speed} m/s, Feels Like: {feels_like}°C, Weather: {weather}, Description: {description}")

def filter_forecast(forecast_data, desired_date):
    filtered_forecast = [forecast for forecast in forecast_data['list'] if forecast['dt_txt'].split()[0] == desired_date]
    return filtered_forecast

city = input('Enter your city: ')
forecast_data = get_forecast(city)
print(f"Average Forecast for {city}:")
average_forecast = forecast_data['list'][0]
format_forecast(average_forecast)

print()
print("Do you have any special plans for the weekend? Maybe a meeting? Or perhaps a date? Let's check the weather forecast for your plans.")

plan = input("Enter your exciting plan!: ")
desired_date = input("Enter the date (YYYY-MM-DD) for which you want the forecast: ")

filtered_forecast = filter_forecast(forecast_data, desired_date)

if filtered_forecast:
    print(f"Forecast for {plan} on {desired_date}:")
    for forecast in filtered_forecast:
        format_forecast(forecast)
else:
    print(f"No forecast available for {plan} on {desired_date}.")
