########## IMPORT LIBRARIES ##########

# psycopg2 is a library to execute sql queries in python
import psycopg2
# sys is required to use the open function to write on file
import sys
# pandas is needed to read the csv file and to perform some basic operations on dataframes
import pandas as pd

########## MAIN FUNCTION ##########

def main():

    if len(sys.argv) <= 11:
        return -1
    
    stop = sys.argv[1]
    id_area = sys.argv[2]
    year = sys.argv[3]
    month = sys.argv[4]
    n_months = sys.argv[5] # number of consecutive months
    week = sys.argv[6] # 0 = whole month, 1 = first 2 weeks, 2 = last 2 weeks
	
	db_name = sys.argv[7]
	host = sys.argv[8]
	user = sys.argv[9]
	password = sys.argv[10]
	port = sys.argv[11]

    month_code = month 
    
    time_interval = ""

    if n_months == "1":
        date_interval = " and date_part('month', start_time) = '" + month + "'"
    else:
        months_list = "'" + month + "'"
        for m in range(1, int(n_months)):
            month_code += "_" + str(int(month)+m)
            months_list += ", '" + str(int(month)+m) +"'"
        date_interval = " and date_part('month', start_time) in (" + months_list + ")"

    if week == "1":
        time_interval = " and start_time < '"+year+"-"+month+"-15'"
    if week == "2":
        time_interval = " and start_time > '"+year+"-"+month+"-15'"

    # open dataset containing the information relative to the area of each vehicle
    df = pd.read_csv('../../datasets/in/Traj' + stop + 'min/vehicle_areas.csv')

    # extract the vehicles with only a selected area
    df_area = df[df["area"] == float(id_area)]

    # open connection to database
    conn = psycopg2.connect(db_name, host=host, user=user, password=password, port=port)
    # the cursor allows Python code to execute PostgreSQL command in a database session.
    cur = conn.cursor()

    file_name = '../../datasets/in/Traj' + stop + 'min/area'+id_area+'_month'+month_code+'_week'+ week +'.csv'

    header = True
    # for each vehicle
    for v in df_area["vehicle"]:

        # take all trajectories of that vehicle in the month selected
        query1 = """
            SELECT vehicle, tid, ST_AsGeoJSON(traj) as trajcoord, company, vehicletype, 
            length, duration, start_time, end_time
            FROM tak.vodafone_zel1_traj_""" + stop + """min
            WHERE vehicle = '""" + v + "'" + time_interval + date_interval +" and length > 0.2"

        # the fist time write also the header for the csv
        if header:
            outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER".format(query1)
                    # write the record in the corresponding file
            with open(file_name, 'w', encoding='utf-8') as f:
                cur.copy_expert(outputquery, f)
            header = False
        else:
            outputquery = "COPY ({0}) TO STDOUT WITH (FORMAT csv)".format(query1)
                    # write the record in the corresponding file
            with open(file_name, 'a', encoding='utf-8') as f:
                cur.copy_expert(outputquery, f)

    conn.close()
    
    return 0


if __name__ == "__main__":
    main()