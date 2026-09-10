import requests


def get_weather(latitude, longitude):

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,rain",
        "hourly": "rain",
        "past_days": 1,
        "forecast_days": 3,
        "timezone": "auto"
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        print("Open-Meteo status:", response.status_code)

        if response.status_code != 200:
            print("API error:", response.text)
            return None

        data = response.json()

        current = data.get("current", {})

        temperature = current.get("temperature_2m", 0)
        humidity = current.get("relative_humidity_2m", 0)
        current_rain = current.get("rain", 0)

        hourly = data.get("hourly", {})
        rainfall = hourly.get("rain", [])

        recent_rainfall = rainfall[-72:]

        rain72 = sum(
            value for value in recent_rainfall
            if value is not None
        )

        return {
            "temperature": temperature,
            "humidity": humidity,
            "current_rain": current_rain,
            "rain72": round(rain72, 2)
        }

    except Exception as e:
        print("Weather API connection error:", e)
        return None


if __name__ == "__main__":

    latitude = 25.86
    longitude = 91.88

    weather = get_weather(latitude, longitude)

    if weather:
        print("\nWeather data received successfully!")
        print(weather)
    else:
        print("No weather data received.")