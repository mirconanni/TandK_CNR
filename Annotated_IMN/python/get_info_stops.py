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

########## MAIN FUNCTION ##########

def main():
    if len(sys.argv) <= 9:
        return -1

    stop = sys.argv[1]
    id_area = sys.argv[2]
    month = sys.argv[3]
    week = sys.argv[4] # 0 = whole month, 1 = first 2 weeks, 2 = last 2 weeks
	
	db_name = sys.argv[5]
	host = sys.argv[6]
	user = sys.argv[7]
	password = sys.argv[8]
	port = sys.argv[9]

    file_name = '../../datasets/in/Traj' + stop + 'min/area'+id_area+'_month'+month+'_week'+ week

    # open dataset containing the information relative to the area of each vehicle
    df = pd.read_csv(file_name+'_stops.csv')

    # open connection to database
    conn = psycopg2.connect(db_name, host=host, user=user, password=password, port=port)
    cursor = conn.cursor()

    eng_status_start = []
    gas_stati_start = []

    eng_status_end = []
    gas_stati_end = []

    for index, row in df.iterrows():
        vid = row['vehicle']
        tid = row['tid']

        # get info start point
        lat = row['start_point'].split(',')[0][1:]
        lon = row['start_point'].split(',')[1][1:-1]

        # query in dataset event con vehicle, tid e lat e long
        cursor.execute("SELECT enginestatus, closetogasstation FROM tak.vodafone_zel1_evnt_" + stop +
                "min WHERE vehicle = '" + str(vid) + "' AND tid = '"+ str(tid) + "' AND lat = '"+ str(lat) + "' AND lon = '" + str(lon) + "'" )

        for ec in cursor:
            eng_status_start.append(list(ec)[0])
            gas_stati_start.append(list(ec)[1])

        
        # get info end point
        lat = row['end_point'].split(',')[0][1:]
        lon = row['end_point'].split(',')[1][1:-1]

        # query in dataset event con vehicle, tid e lat e long
        cursor.execute("SELECT enginestatus, closetogasstation FROM tak.vodafone_zel1_evnt_" + stop +
                "min WHERE vehicle = '" + str(vid) + "' AND tid = '"+ str(tid) + "' AND lat = '"+ str(lat) + "' AND lon = '" + str(lon) + "'" )

        for ec in cursor:
            eng_status_end.append(list(ec)[0])
            gas_stati_end.append(list(ec)[1])

    # closes connection to the server
    conn.close()

    df["eng_status_start"] = eng_status_start
    df["gas_stati_start"] = gas_stati_start
    df["eng_status_end"] = eng_status_start
    df["gas_stati_end"] = gas_stati_start

    df.to_csv(file_name +'_stop_info.csv', index=False)

    return 0


if __name__ == "__main__":
    main()