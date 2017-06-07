import pickle
import csv

print("reading files")

weather = []

for year in range(2002, 2018):
    for month in range(1,13):
        if not (year==2017 and month > 4):
            #calculate length of month for filename
            if month in [1,3,5,7,8,10,12]:
                daysPerMonth = 31
            elif month != 2:
                daysPerMonth = 30
            elif year in [2004, 2008, 2012, 2016]:
                daysPerMonth = 29
            else:
                daysPerMonth = 28
            
            #create name of file for particular year + month
            fileString = "eng-hourly-"
            if month <10:
                fileString += "0" + str(month)
            else:
                fileString += str(month)
            fileString += "01" + str(year) + "-"
            if month <10:
                fileString += "0" + str(month)
            else:
                fileString += str(month)
            fileString += str(daysPerMonth)
            fileString += str(year)
            fileString += ".csv"
            
            with open(fileString, newline="") as inputFile:
                inputReader = csv.reader(inputFile)
                for row in inputReader:
                    if row != [] and row[0][0]=="2":
                        weather.append(row)

print("checking for valid data points")

validWeather = []

def isFloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

for row in weather:
    if len(row) >=10 and isFloat(row[6]) and isFloat(row[8]) and isFloat(row[10]):
        timestamp = row[0]
        temperature = float(row[6])
        dewPoint = float(row[8])
        humidity = float(row[10])
        validWeather.append([timestamp, temperature, dewPoint, humidity])

print("dumping to pickle")

output = open('weather-dump.pkl', 'wb')
pickle.dump(validWeather, output)
output.close()
