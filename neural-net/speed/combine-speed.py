import pickle
import csv
import shapefile

print("reading shapes")

sf = shapefile.Reader("bluetooth_routes_wgs84.shp")
shapes = sf.shapes()
records = sf.records()
shapeDict = dict()
for i in range(len(shapes)):
    resultID = records[i][0]
    location_x = (shapes[i].bbox[0]+shapes[i].bbox[2])/2
    location_y = (shapes[i].bbox[1]+shapes[i].bbox[3])/2
    shapeDict[resultID] = records[i][1:3] + [location_x] + [location_y]

print("reading speeds")

speeds = []
for year in range(2014,2018):
    fileName = "bt_" + str(year) + ".csv"
    with open(fileName,newline="") as inputFile:
        reader = csv.reader(inputFile)
        next(reader)
        for row in reader:
            speeds.append(row)

print("combining data")

data = []
for row in speeds:
    timestamp = row[3]
    travelTime = float(row[1])
    resultID = row[0]
    normalTime = float(shapeDict[resultID][0])
    volume = float(row[2])
    location_x = shapeDict[resultID][2]
    location_y = shapeDict[resultID][3]
    relativeTime = travelTime/normalTime
    data.append([timestamp, location_x, location_y, relativeTime, volume])

print("dumping to pickle")

output = open('speed-dump.pkl', 'wb')
pickle.dump(data, output)
output.close()
