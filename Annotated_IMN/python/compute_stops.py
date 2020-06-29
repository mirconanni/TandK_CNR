########## IMPORT LIBRARIES ########## 

# sys is required to use the open function to write on file
import sys
# pandas is needed to read the csv file and to perform some basic operations on dataframes
import pandas as pd
# ST_AsGeoJSON returns a json object, so we can use json.load to parse it
import json
# psycopg2 is a library to execute sql queries in python
import psycopg2
# required to read/write on csv file
import csv

########## IMPORT MY SCRIPTS ########## 

from geo_partition import swap_xy


########## FUNCTION DEFINITION ##########

# given a dataframe reads the trajectory column and returns arrays of starting and ending points
def extract_stops(df):
    coords = []

    # for each trajectory
    for el in  df["trajcoord"]:
        y = json.loads(el)
        c = swap_xy(y["coordinates"])
        coords.append(c)
    
    start_points = []
    end_points = []
    for t in coords:
        start_points.append(t[0])
        end_points.append(t[-1])
    return start_points, end_points 


def read_params(sys):
    if len(sys.argv) <= 5:
        return -1

    stop = sys.argv[1]
    id_area = sys.argv[2]
    month =  sys.argv[3]
    n_months = sys.argv[4] 
    week = sys.argv[5] # 0 = whole month, 1 = first 2 weeks, 2 = last 2 weeks
    
    month_code = month
    if n_months != "1":
        for m in range(1, int(n_months)):
            month_code += "_" + str(int(month)+m)
        
    return stop, id_area, month_code, week


########## MAIN FUNCTION ##########

def main():
        
    stop, id_area, month_code, week = read_params(sys)

    file_name = '../../datasets/in/Traj' + stop + 'min/area'+id_area+'_month'+month_code+'_week'+ week

    # open dataset containing the information relative to the area of each vehicle
    df = pd.read_csv(file_name+'.csv')

    # takes the points
    start_points, end_points = extract_stops(df)

    df["start_point"] = start_points
    df["end_point"] = end_points

    df = df.drop(columns=['trajcoord'])

    df.to_csv(file_name +'_stops.csv', index=False)

    return 0


if __name__ == "__main__":
    main()