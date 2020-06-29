########## IMPORT LIBRARIES ########## 

# sys is required to use the open function to write on file
import sys
# pandas is needed to read the csv file and to perform some basic operations on dataframes
import pandas as pd
# ST_AsGeoJSON returns a json object, so we can use json.load to parse it
import json
# to read and write on csv
import csv
# transform coordinates into scring codes 
import sklearn.neighbors as neigh
# compute the nearest neighbors of a set of points
from sklearn.neighbors import NearestNeighbors
# to perform normalization of the dataframe
from sklearn import preprocessing
# operations on arrays
import numpy as np
# date format 
import datetime
# manipulation of time format
import time


########## IMPORT MY SCRIPTS ########## 

from build_loc_feat import from_array_to_string


########## MAIN FUNCTION ##########

def main():

    if len(sys.argv) <= 2:
        return -1

    stop = sys.argv[1]
    id_area = sys.argv[2]

    df_areas = pd.read_csv('../../datasets/in/Traj' + stop + 'min/vehicle_areas.csv')

    df_areas = df_areas[df_areas["area"] == int(id_area)]
    bottom_left_y_min = df_areas["bottom_left_y"].min()
    bottom_left_x_min = df_areas["bottom_left_x"].min()
    top_right_y_min = df_areas["top_right_y"].max()
    top_right_x_min = df_areas["top_right_x"].max()

    with open('../../datasets/POIdict.json', 'r') as f:
        dict_poi = json.load(f)

    file_name_in = ['../../datasets/euro_pofw.csv', '../../datasets/euro_poi.csv', '../../datasets/euro_transport_traffic.csv']

    file_name_out = '../../datasets/athens_POI.csv'
    header = "fclass,category,lon,lat,name\n"

    # write header
    with open(file_name_out, 'w', newline='\n') as f:
        f.write(header)

    for f in file_name_in:
        df = pd.read_csv(f)

        for _, row in df.iterrows():
            lat = row["lat"]
            lon = row["lon"]

            # if the poi is in the area selected
            if bottom_left_y_min < lat < top_right_y_min and bottom_left_x_min < lon < top_right_x_min:
                fclass = row["fclass"]
                # if it's of a category we're interested in
                if fclass in dict_poi.keys():
                    category = dict_poi[fclass]

                    # write the row in the dataset
                    with open(file_name_out, 'a', newline='\n') as f:
                        line = [fclass, category, lon, lat]
                        f.write(from_array_to_string(line)+"\n")
    
    return 0


if __name__ == "__main__":
    main()