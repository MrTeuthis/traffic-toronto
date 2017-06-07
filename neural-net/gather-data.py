import pickle
import re
import datetime

print("opening weather pickle dump")
weather = pickle.load(open("weather/weather-dump.pkl", "rb"))

print("creating dictionary for weather data")
weatherDict = dict()
for row in weather:
    timestamp = [int(_) for _ in re.split(":| |-", row[0])]
    year = timestamp[0]
    month = timestamp[1]
    day = timestamp[2]
    hour = timestamp[3]
    temperature = row[1]
    dewPoint = row[2]
    humidity = row[3]
    time = tuple([year, month, day, hour])
    weatherDict[time] = [temperature, dewPoint, humidity]

print("opening speed pickle dump")
speed = pickle.load(open("speed/speed-dump.pkl", "rb"))

print("finding matching weather/speed data")
data = []
for row in speed:
    timestamp = [int(_) for _ in re.split(":| |-|T", row[0])]
    year = timestamp[0]
    month = timestamp[1]
    day = timestamp[2]
    hour = timestamp[3]
    minute = timestamp[4]
    time = tuple([year, month, day, hour])
    location_x = row[1]
    location_y = row[2]
    relativeTime = row[3]
    volume = row[4]
    currentDate = datetime.datetime(year, month, day)
    baselineDate = datetime.datetime(year, 1, 1)
    secondBaselineDate = datetime.datetime(year+1, 1, 1)
    oneYear = secondBaselineDate - baselineDate
    timePassed = currentDate - baselineDate
    timeOfYear = float(timePassed.total_seconds())/float(oneYear.total_seconds())
    weekday = currentDate.weekday()
    timeOfDay = (float(hour) + float(minute)/60.0)/24.0
    if time in weatherDict.keys():
        temperature = weatherDict[time][0]
        dewPoint = weatherDict[time][1]
        humidity = weatherDict[time][2]
        data.append([timeOfYear, weekday, timeOfDay, location_x, location_y, temperature, dewPoint, humidity, relativeTime, volume])

print("dumping weather/speed data to pickle file")
output = open('data/complete-dump.pkl', 'wb')
pickle.dump(data, output)
output.close()
