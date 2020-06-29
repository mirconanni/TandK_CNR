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
# numpy is required to use some mathematical functions
import numpy as np
# provides functions to decode and encode Geohashes to and from lat/lon coords
import geohash as ghh 
# required to define a dictionary
from collections import defaultdict
# write results in json file
import json
# transform coordinates into scring codes 
from geohash import encode

import datetime


########## IMPORT MY SCRIPTS ########## 

from compute_stops import read_params
from tosca import Tosca, thompson_test
from distance_func import spherical_distance, spherical_distances


########## FUNCTION DEFINITION ##########

# get list of points from stop lists
def get_point_list(df_v):
    start_points_string = list(df_v["start_point"])
    end_points_string = list(df_v["end_point"])

    points = []
    # in csv points are stored as strings
    for i in range(len(start_points_string)):
        p_start = start_points_string[i]
        p_0 = (p_start.split(','))[0][1:]
        p_1 = (p_start.split(','))[1][1:]
        points.append([float(p_0), float(p_1)])

        p_end = end_points_string[i]
        p_0 = (p_end.split(','))[0][1:]
        p_1 = (p_end.split(','))[1][1:]
        points.append([float(p_0), float(p_1)])

    return points

# compute a gross approximation of the minimum and maximum number of clusters
def get_min_max(points):
    min_loc = defaultdict(int)
    max_loc = defaultdict(int)

    # each point is encoded with two different precisions
    for p in points:
        # encode is a geohash function that transform a point to a short string of letters and digits
        # the more the precision, the smaller the area it describes
        min_loc[encode(float(p[0]), float(p[1]), precision=5)] += 1
        max_loc[encode(float(p[0]), float(p[1]), precision=7)] += 1

    # we count the number of different clusters
    return len(min_loc), len(max_loc)

# the radial distance of a point from the axis of rotation
def radius_of_gyration(points, centre_of_mass, dist):
    rog = 0
    for p in points:
        rog += dist(p, centre_of_mass)
    rog = 1.0*rog/len(points)
    return rog

# Shannon entropy is used to measure the distribution of ingoing and outgoing trips
def entropy(x, classes=None):
    if len(x) == 1:
        return 0.0
    val_entropy = 0
    n = np.sum(x)
    for freq in x:
        if freq == 0:
            continue
        p = 1.0 * freq / n
        val_entropy -= p * np.log2(p)
    if classes is not None and classes:
        val_entropy /= np.log2(classes)
    return val_entropy

# given a vehicle id, a df containing its list of stops write the list of locations on the output file
def detect_locations(v, df_v):
    # extract list of stop points
    points = get_point_list(df_v)

    # compute min and max k for xmeans
    min_k, max_k = get_min_max(points)
    
    cluster_res = dict()
    cuts = dict()
    min_dist=50.0

    # runs the tosca algorithm 5 times, each time store the centers and the cut distance
    # relative to the k found
    for _ in range(0, 3):
        try:
            tosca = Tosca(kmin=min_k, kmax=max_k, xmeans_df=spherical_distances,
                        singlelinkage_df=spherical_distance, is_outlier=thompson_test,
                        min_dist=min_dist, verbose=False)
            tosca.fit(points)

            cluster_res[tosca.k_] = tosca.cluster_centers_
            cuts[tosca.k_] = tosca.cut_dist_
        except ValueError:
            pass
    # if always value error stop
    if len(cluster_res) == 0:
        return None
    # of all the different results of tosca take the one with min k
    index = np.min(list(cluster_res.keys()))
    centers = cluster_res[index]
    loc_tosca_cut = cuts[index]

    # calculate the pairwise distance between points and medoids
    distances = spherical_distances(np.asarray(points), np.asarray(centers))
    # calculates labels according to minimum distance
    # argmin returns the indices of the minimum values along an axis
    labels = np.argmin(distances, axis=1)

    # create two dict, one from cluster label to position of relative point in point list
    # the second from cluster label to its center
    location_points = defaultdict(list)
    location_prototype = dict()
    for pid, lid in enumerate(labels):
        location_points[lid].append(pid)
        location_prototype[lid] = list(centers[lid])

    # rename locations from bigger to smaller
    pid_lid = dict()
    location_support = dict()
    # sort cluster labels according to cluster size
    location_sorted = sorted(location_points.keys(), key=lambda x: len(location_points[x]), reverse=True)
    new_location_points = defaultdict(list)
    new_location_prototype = defaultdict(list)
    # rename clusters according to position in sorted, so the biggest cluster is 1
    for new_lid, lid in enumerate(location_sorted):
        # update previu lists of points and labels
        new_location_points[new_lid] = location_points[lid]
        new_location_prototype[new_lid] = location_prototype[lid]
        # store the length of each cluster
        location_support[new_lid] = len(location_points[lid])
        # create a dict from position of the point in the list to cluster label 
        for pid in location_points[lid]:
            pid_lid[pid] = new_lid

    location_points = new_location_points
    location_prototype = new_location_prototype

    # statistical information for users analysis
    cm = np.mean(points, axis=0)
    rg = radius_of_gyration(points, cm, spherical_distance)
    en = entropy(list(location_support.values()), classes=len(location_support))

    res = {'location_points': location_points,
                'location_prototype': location_prototype,
                'pid_lid': pid_lid,
                'rg': rg,
                'entropy': en,
                'loc_tosca_cut': loc_tosca_cut
          }
    return res

    
########## MAIN FUNCTION ##########

def main():

    stop, id_area, month_code, week = read_params(sys)

    file_name_in = '../../datasets/in/Traj' + stop + 'min/area'+id_area+'_month'+month_code+'_week'+ week

    # open dataset containing the information relative to the area of each vehicle
    df = pd.read_csv(file_name_in+'_stops.csv')  

    file_name_out = '../../datasets/out/Traj' + stop + 'min/locations_area'+id_area+'_month'+month_code+'_week'+ week

    #restart = False

    # if restart:
    #create or clear the output file
    with open(file_name_out+'.json', 'w') as out:
        out.write("{")
        
    # for each vehicle in that area and time
    for v in df.vehicle.unique():

        #print("computing ", v, " starting at ", datetime.datetime.now())
        
        #if restart:
        df_v = df[df["vehicle"] == v]
        res = detect_locations(v, df_v)

        # the resulting file will have one line for each vehicle in that area and time
        # each line contains the list of points belonging to each cluster
        with open(file_name_out+'.json', 'a+') as outfile:
            outfile.write('"'+v+'" :')
            json.dump(res, outfile)
            outfile.write(",\n") 

        #if (v == "1110_13180"):
        #    restart = True
    
    with open(file_name_out+'.json', 'a') as out:
        out.write("}")
        

    return 0


if __name__ == "__main__":
    main()