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

########## FUNCTION DEFINITION ##########

# Create a list of stops taking the first and the last point
def create_stop_list(traj_list):
    stop_list = []
    for t in traj_list:
        stop_list.append(t[0])
        stop_list.append(t[-1])
    return stop_list

# Given a trajectory for each point swaps the order of the x and the y
def swap_xy(traj):
    new_traj = []
    for i in traj:
        new_traj.append([i[1], i[0], i[2]])
    return new_traj

# computes the area of a rectangle
def rect_extension(r):
    return (r[1][0] - r[0][0]) * (r[1][1] - r[0][1])

# Given two rectangles checks if thier area is of the same magnitude
def similar_extension(cur_rect, other_r):
    cur_extension = rect_extension(cur_rect)
    other_extension = rect_extension(other_r)
    extension_difference = abs(cur_extension - other_extension)
    if extension_difference < 3 * min(cur_extension, other_extension):
        return True
    return False

# Check if two rectangles are overlapping
def is_overlapping(cur_rect, other_rect):
    if cur_rect[1][0] < other_rect[0][0] or cur_rect[0][0] > other_rect[1][0]:
        return False
    if cur_rect[1][1] < other_rect[0][1] or cur_rect[0][1] > other_rect[1][1]:
        return False
    return True

# Given the current rectangle checks if it is overlapping and it has a similar extension
# with at least 75% of rectangles in that area
def compute_area(cur_rect, n_areas, rect_list):
    for i in range(1, n_areas):
        list_area_i = rect_list[i]
        n_elem = len(list_area_i)
        n_overlap = 0
        for r in list_area_i:
            if is_overlapping(cur_rect, r) and similar_extension(cur_rect, r):
                n_overlap += 1
        if n_overlap > 0:
            perc = float(n_overlap) / n_elem
            if perc > 0.75:
                return i
    return n_areas

# Look for the min and max lat and lon of all trajectories to identify the rectangle
def compute_coord_rec(traj_list):
    stop_list = create_stop_list(traj_list)
    min_lat = +90
    min_lon = +180
    max_lat = -90
    max_lon = -180
    for point in stop_list:
        if point[0] < min_lat:
            min_lat = point[0]
        if point[0] > max_lat:
            max_lat = point[0]
        if point[1] < min_lon:
            min_lon = point[1]
        if point[1] > max_lon:
            max_lon = point[1]
    bottom_left = [min_lat, min_lon]
    top_right = [max_lat, max_lon]
    return bottom_left, top_right

# Queries the dataset for the trajectories of that vehicle, extract the coordinates
# with json load, get list of trajectories
def get_list_traj(cursor, stop, v_id):
    cursor.execute("SELECT ST_AsGeoJSON(traj) as trajcoord FROM tak.vodafone_zel1_traj_" + stop +
                   "min WHERE vehicle = '" + v_id + "'")
    traj_list = []
    for t in cursor:
        y = json.loads(t[0])
        c = swap_xy(y["coordinates"])
        traj_list.append(c)
    return traj_list

# Open a new csv file and writes the header
def write_header(file_name):
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        header = ["vehicle", "vehicletype", "bottom_left_y", "bottom_left_x", "top_right_y", "top_right_x", "area"]
        writer.writerow(header)


########## MAIN FUNCTION ##########

def main():
    if len(sys.argv) <= 6:
        return -1
           
    stop = sys.argv[1]  # stop can be either 5 or 10
	
	db_name = sys.argv[2]
	host = sys.argv[3]
	user = sys.argv[4]
	password = sys.argv[5]
	port = sys.argv[6]

    # open vehicle list.csv as df
    df = pd.read_csv('../../datasets/in/Traj' + stop + 'min/vehicle_list.csv')

    # open connection to database
    conn = psycopg2.connect(db_name, host=host, user=user, password=password, port=port)
    cursor = conn.cursor()

    # create new csv vehicle areas and write header
    file_name = '../../datasets/in/Traj' + stop + 'min/vehicle_areas.csv'
    write_header(file_name)

    n_areas = 1
    rect_list = dict()

    for v_id in df["vehicle"]:
        vehicletype = " ".join(df[df["vehicle"] == v_id]["vehicletype"].to_string().split()[1:]).strip()

        # query to the dataset to get the list of trajectories of that vehicle
        traj_list = get_list_traj(cursor, stop, v_id)

        # computes the bound coordinates of all trajectories of that vehicle
        bottom_left, top_right = compute_coord_rec(traj_list)
        cur_rect = [bottom_left, top_right]

        # decide to which area assign the new vehicle
        area_id = compute_area(cur_rect, n_areas, rect_list)
        if area_id == n_areas:
            rect_list[area_id] = []
            n_areas += 1

        rect_list[area_id].append([bottom_left, top_right])

        # stores a line for each vehicle with information about its location
        with open(file_name, 'a', newline='') as file:
            row = [v_id, vehicletype, bottom_left[0], bottom_left[1], top_right[0], top_right[1], area_id]
            writer = csv.writer(file, delimiter=',', quotechar="\"")
            writer.writerow(row)

    # closes connection to the server
    conn.close()

    return 0


if __name__ == "__main__":
    main()