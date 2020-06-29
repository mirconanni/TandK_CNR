########## IMPORT LIBRARIES ########## 

# sys is required to use the open function to write on file
import sys
# read input json file and write on output json file
import json
# pandas is needed to read the csv file and to perform some basic operations on dataframes
import pandas as pd
import os

########## IMPORT MY SCRIPTS ########## 

from imn_light_measures import *
from detect_movements import traj_dict, clear_tuples4json, serialize_graph
from compute_stops import read_params

########## FUNCTION DEFINITION ##########

# get list of points from stop lists with time, label "from" or "to" and traj_id
def get_complete_point_list(df_v):
    start_points_string = list(df_v["start_point"])
    end_points_string = list(df_v["end_point"])
    tid = list(df_v["tid"])

    points = []
    # in csv points are stored as strings
    for i in range(len(start_points_string)):
        p_start = start_points_string[i]
        p_0 = (p_start.split(','))[0][1:]
        p_1 = (p_start.split(','))[1][1:]
        p_2 = (p_start.split(','))[2][1:-1]
        points.append([float(p_0), float(p_1), float(p_2), 'f', tid[i]])

        p_end = end_points_string[i]
        p_0 = (p_end.split(','))[0][1:]
        p_1 = (p_end.split(','))[1][1:]
        p_2 = (p_end.split(','))[2][1:-1]
        points.append([float(p_0), float(p_1), float(p_2), 't', tid[i]])

    return points

# from the locations and the movements of a vehicle builds the imn
def build_imn(locations_v, movements_v, df_v, df_v_stops, reg_loc=True, events=None):

    n_traj = len(df_v)

    # takes the location list
    loc_res = locations_v
    # takes the movement list
    mov_res = movements_v
    
    points = get_complete_point_list(df_v_stops)
    
    n_locs = len(loc_res['location_points'])
    n_movs = len(mov_res['movement_traj'])

    # create a dict from traj_id to list of coordinates
    trajectories = traj_dict(df_v)

    # dict from location to list of points
    regular_locs = loc_res['location_points']
    n_reg_locs = len(regular_locs)

    if reg_loc:
        # filter the regular locations, don't like it very much
        regular_locs, _= detect_regular_locations(loc_res['location_points'], mov_res['loc_nextlocs'])
        n_reg_locs = len(regular_locs)

    # compute a set of stats for each location
    lf_res = get_locations_features(points, mov_res['traj_from_to_loc'], loc_res['location_points'], loc_res['location_prototype'], trajectories, df_v_stops)
    # compute the avg and the std time spent in each location
    lf_dur, lf_ac, mf_ac = get_locations_durations(trajectories, mov_res['traj_from_to_loc'], mov_res['lft_mid'])
    
    # for each location, adds staying time information if present
    for loc, _ in lf_res.items():
        if int(loc) in lf_dur:
            lf_res[loc].update(lf_dur[int(loc)])
        if int(loc) in lf_ac:
            lf_res[loc]['timeslot_count'] = lf_ac[int(loc)]

    traj_len_dur = dict()
    for _, row in df_v.iterrows():
        tid = row["tid"]
        traj_len_dur[tid] = {"length": row["length"], "duration": row["duration"]}

            
    # get mean of duration and length of each movement
    mf_res = get_movements_features(mov_res['movement_traj'], traj_len_dur)
    # get the mean length and duration of all the movements
    ms_res = get_movements_stats(mov_res['movement_traj'], regular_locs, traj_len_dur)
    # join the movements info collected
    for mov, _ in mf_res.items():
        if mov in mf_ac:
            mf_res[mov]['timeslot_count'] = mf_ac[mov]

    imn = {
        'regular_locations': list(regular_locs.keys()),
        'location_prototype': loc_res['location_prototype'],
        'location_nextlocs': mov_res['loc_nextlocs'],
        'location_features': lf_res,

        'location_from_to_movement': mov_res['lft_mid'],
        'mov_features': mf_res,

        'n_traj': n_traj,
        'n_reg_traj': ms_res['n_reg_traj'],
        'n_locs': n_locs,
        'n_reg_locs': n_reg_locs,
        'n_movs': n_movs,
        'n_reg_movs': ms_res['n_reg_movs'],
        'graph': mov_res['graph']
    }

    return imn


########## MAIN FUNCTION ##########

def main():

    stop, id_area, month_code, week = read_params(sys)

    file_name_in0 = '../../datasets/in/Traj' + stop + 'min/area'+id_area+'_month'+month_code+'_week'+ week

    file_name_in1 = '../../datasets/out/Traj' + stop + 'min/locations_area'+id_area+'_month'+month_code+'_week'+ week

    file_name_in2 = '../../datasets/out/Traj' + stop + 'min/movements_area'+id_area+'_month'+month_code+'_week'+ week

    file_name_out = '../../datasets/out/Traj' + stop + 'min/imn_light_area'+id_area+'_month'+month_code+'_week'+ week

    restart = False
    # create or clear the output file
    # with open(file_name_out+'.json', 'w') as out:
    #     out.write("{")

    # open dataset containing the information relative to the area of each vehicle
    df = pd.read_csv(file_name_in0+'.csv') 
    df_stops = pd.read_csv(file_name_in0+'_stops.csv') 

    with open(file_name_in1+'.json', 'r') as f:
        file_j = json.load(f)

    # for each vehicle in that area and time
    for v in file_j.keys():

        if restart:

            df_v = df[df["vehicle"] == v]
            df_v.reset_index(inplace=True)

            df_v_stops = df_stops[df_stops["vehicle"] == v]
            df_v_stops.reset_index(inplace=True)

            with open(file_name_in1+'.json', 'r') as f:
                file_j = json.load(f)
                locations_v = file_j[v]

            with open(file_name_in2+'.json', 'r') as f:
                file_j = json.load(f)
                movements_v = file_j[v]

            imn = build_imn(locations_v, movements_v, df_v, df_v_stops)

            with open(file_name_out+'.json', 'a+') as outfile:
                    outfile.write('"'+v+'" :')
                    json.dump(clear_tuples4json(imn), outfile, default=serialize_graph)
                    outfile.write(",\n") 
                    
        if (v == "8240_84630"):
            restart = True

    # remove the last n chars
    with open(file_name_out+'.json', 'rb+') as filehandle:
        filehandle.seek(-3, os.SEEK_END)
        filehandle.truncate()
    
    #write something at the end of the file
    with open(file_name_out+'.json', 'a') as f:
        f.write("\n}")

    
    return 0


if __name__ == "__main__":
    main()