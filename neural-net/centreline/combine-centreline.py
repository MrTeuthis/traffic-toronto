import shapefile
import pickle

sf = shapefile.Reader("CENTRELINE_WGS84.shp")
shapes = sf.shapes()
records = sf.records()
centrelineDict = dict()

for i in range(len(shapes)):
    center_x = (shapes[i].bbox[0]+shapes[i].bbox[2])/2.0
    center_y = (shapes[i].bbox[1]+shapes[i].bbox[3])/2.0
    centrelineDict[records[i][0]] = [center_x, center_y]

output = open('centreline-dump.pkl', 'wb')
pickle.dump(centrelineDict, output)
output.close()

