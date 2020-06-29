########## IMPORT LIBRARIES ########## 

# sys is required to use the open function to write on file
import sys
# graphs library
import networkx as nx
# builds dictionaruies
from collections import defaultdict
# read input json file and write on output json file
import json
# pandas is needed to read the csv file and to perform some basic operations on dataframes
import pandas as pd
# use date format
import datetime
# numpy array
import numpy as np
# makes graphs serializable
from networkx.readwrite import json_graph

import os


########## IMPORT MY SCRIPTS ########## 

from distance_func import trajectory_distance
from compute_stops import swap_xy, read_params

########## FUNCTION DEFINITION ##########

# transform graph into serializable version
def serialize_graph(o):
    if isinstance(o, nx.DiGraph):
        return json_graph.node_link_data(o, {'link': 'edges', 'source': 'from', 'target': 'to'})
    else:
        return o.__str__()

# convert an object to string
def key2str(k):
    if isinstance(k, tuple):
        return str(k)
    elif isinstance(k, datetime.time):
        return str(k)
    elif isinstance(k, np.int64):
        return str(k)
    elif isinstance(k, np.float64):
        return str(k)
    return k

# convert dict keys to string in order to write them on json
def clear_tuples4json(o):
    if isinstance(o, dict):
        return {key2str(k): clear_tuples4json(o[k]) for k in o}
    return o

# create a dict from traj_id to list of coordinates
def traj_dict(df_v):
    trajectories = dict()
    for _, row in df_v.iterrows():
        tid = row["tid"]
        y = json.loads(row["trajcoord"])
        c = swap_xy(y["coordinates"])
        trajectories[tid] = c
    return trajectories

# create a dict from traj tid to start-end point id
def get_traj_from_to(df_v, n_traj):
    traj_from_to = dict()
    node_id = 0
    for i, row in df_v.iterrows():
        tid = row["tid"]
        traj_from_to[tid] = [node_id, node_id+1]
        node_id += 2
    return traj_from_to

# given the cluster for each point and the trajectories compute the movements across the locations
def movements_detection(pid_lid, df_v):

    n_traj = len(df_v)

    # get traj dict from tid to start-end point id
    traj_from_to = get_traj_from_to(df_v, n_traj)

    traj_from_to_loc = dict()
    loc_from_to_traj = defaultdict(list)
    loc_nextlocs = defaultdict(lambda: defaultdict(int))

    # for each trajectory
    for tid, from_to in traj_from_to.items():
        # cluster point from
        loc_from = pid_lid[str(from_to[0])]
        # cluster point to
        loc_to = pid_lid[str(from_to[1])]
        # dict from tid to cluster from to
        traj_from_to_loc[tid] = [loc_from, loc_to]
        # dict from cluster from_to to tid
        loc_from_to_traj[(loc_from, loc_to)].append(tid)
        # number of arcs connecting two locations
        loc_nextlocs[loc_from][loc_to] += 1

    # create graph 
    G = nx.DiGraph()
    # for combination of starting point and ending point
    for loc_from in loc_nextlocs:
        for loc_to in loc_nextlocs[loc_from]:
            # get the number of trajectory among those points
            nbr_traj = loc_nextlocs[loc_from][loc_to]
            # append an edge with weight = number of trajectories
            G.add_edge(loc_from, loc_to, weight=nbr_traj)

    movement_traj = dict()
    lft_mid = dict()
    # for each edge (mid = edge id, lft = cluster from_to)
    for mid, lft in enumerate(loc_from_to_traj):
        # dict from edge id to cluster from_to and list of tid among those points
        movement_traj[mid] = [lft, loc_from_to_traj[lft]]
        # dict from cluster from_to to edge id
        lft_mid[lft] = mid
    
    trajectories = traj_dict(df_v)

    movement_prototype = dict()
    # for each edge
    for mid in movement_traj:
        # take list of trajectories
        traj_in_movement = movement_traj[mid][1]

        # if weight is at least 2
        if len(traj_in_movement) > 2:
            prototype = None
            min_dist = float('inf')
            # for each trajectories combination
            for tid1 in traj_in_movement:
                tot_dist = 0.0
                traj1 = trajectories[tid1] ######## change
                for tid2 in traj_in_movement:
                    traj2 = trajectories[tid2] ###### change
                    # compute the distance between the two trajectories
                    dist = trajectory_distance(traj1, traj2)
                    # sum all the distances found
                    tot_dist += dist
                # take as prototype the trajectory that most resemble all the others 
                if tot_dist < min_dist:
                    min_dist = tot_dist
                    prototype = traj1
            movement_prototype[mid] = prototype
        else:
            movement_prototype[mid] = trajectories[traj_in_movement[0]]

    res = {
        'movement_traj': movement_traj,
        'movement_prototype': movement_prototype,
        'loc_nextlocs': loc_nextlocs,
        'traj_from_to_loc': traj_from_to_loc,
        'lft_mid': lft_mid,
        'graph': G
    }

    return res


########## MAIN FUNCTION ##########

def main():
       
    stop, id_area, month_code, week = read_params(sys)

    file_name_in1 = '../../datasets/in/Traj' + stop + 'min/area'+id_area+'_month'+month_code+'_week'+ week

    # open dataset containing the information relative to the area of each vehicle
    df = pd.read_csv(file_name_in1+'.csv') 

    file_name_in2 = '../../datasets/out/Traj' + stop + 'min/locations_area'+id_area+'_month'+month_code+'_week'+ week

    file_name_out = '../../datasets/out/Traj' + stop + 'min/movements_area'+id_area+'_month'+month_code+'_week'+ week

    restart = False

    #if restart:
    # # create or clear the output file
    # with open(file_name_out+'.json', 'w') as out:
    #    out.write("{")

    # for each vehicle in that area and time
    for v in df.vehicle.unique(): 

        if restart:
            df_v = df[df["vehicle"] == v]
            df_v.reset_index(inplace=True)

            with open(file_name_in2+'.json', 'r') as f:
                file_j = json.load(f)
                locations_v = file_j[v]

                mov_res = movements_detection(locations_v['pid_lid'], df_v)

                # the resulting file will have one line for each vehicle in that area and time
                # each line contains the list of points belonging to each cluster
                with open(file_name_out+'.json', 'a+') as outfile:
                    outfile.write('"'+v+'" :')
                    json.dump(clear_tuples4json(mov_res), outfile, default=serialize_graph)
                    outfile.write(",\n") 

        if (v == "9730_95120"):
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