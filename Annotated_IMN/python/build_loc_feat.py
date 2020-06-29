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

from compute_stops import read_params
from distance_func import spherical_distance


########## FUNCTION DEFINITION ##########

# 
def geographic_charac(df_poi, df_c, file_name_out):
    
    loc_proto_lat = df_c["loc_proto_lat"]
    loc_proto_lon = df_c["loc_proto_lon"]
    locs_proto = [list(x) for x in zip(loc_proto_lat, loc_proto_lon)]

    poi_lat = df_poi ["lat"]
    poi_lon = df_poi ["lon"]
    poi_coords = [list(x) for x in zip(poi_lat, poi_lon)]

    # get category list
    categories = ["gas", "parking", "pier", "hotel", "food", "leisure", "shop", "service", "supermarket"]

    print("centrality poi")
    # count the number of poi for each category in a radius of 1 km
    centrality_poi = []
    neigh = NearestNeighbors(radius=500, metric=spherical_distance)
    neigh.fit(poi_coords)

    for i in range(len(loc_proto_lat)):
        loc_x = loc_proto_lat[i]
        loc_y = loc_proto_lon[i]

        rng = neigh.radius_neighbors([[loc_x, loc_y]])
        nei_index = list(rng[1][0])
        
        count_c = dict.fromkeys(categories, 0)

        for j in nei_index:
            p_c = df_poi.iloc[j]["category"]

            count_c[p_c] += 1
        centrality_poi.append(list(count_c.values()))

    df_g1 = pd.DataFrame(centrality_poi, columns=["n_"+c for c in categories])

    df_g1.to_csv(file_name_out+"_geo.csv", mode = "w", index=False)

    print("count_nearest_neighbour")
    knei_poi = []
    # count how many poi for each category there are in the top 10 nearest neighbours 
    neigh = NearestNeighbors(n_neighbors=30, metric=spherical_distance) 
    neigh.fit(poi_coords) 
    
    distances, indices = neigh.kneighbors(locs_proto)
    # for each point take list of 11 neighbours indices
    for nei_index in indices:
        count_c = dict.fromkeys(categories, 0)

        for j in nei_index:
            p_c = df_poi.iloc[j]["category"]

            count_c[p_c] += 1

        knei_poi.append(list(count_c.values()))

    df_g2 = pd.DataFrame(knei_poi, columns=["k_"+c for c in categories])
    df_g = pd.concat([df_g1, df_g2], axis=1)

    df_g.to_csv(file_name_out+"_geo.csv", mode = "w", index=False)

    print("dist_nearest_neighbour")
    dist_poi = []
    # take the min distance for each category in the top 10 nearest neighbours
    for i, d in zip(indices, distances):
        dist_c = dict.fromkeys(categories, 10000)

        for j in range(len(d)):
            p_c = df_poi.iloc[i[j]]["category"]

            if d[j] < dist_c[p_c]:
               dist_c[p_c] = d[j]

        dist_poi.append(list(dist_c.values()))

    df_g3 = pd.DataFrame(dist_poi, columns=["d_"+c for c in categories])
    df_g = pd.concat([df_g, df_g3], axis=1)

    df_g.to_csv(file_name_out+"_geo.csv", mode = "w", index=False)

    return df_g
    
# compute the distance of the k nearest neighbour
def _compute_rev_centrality(locs_proto, df_c, file_name_out):
    neigh = NearestNeighbors(n_neighbors=25, metric=spherical_distance) 
    neigh.fit(locs_proto) 

    n_neigh = [1, 3, 5, 8, 10, 20]
    rev_centrality = []
    distances, _ = neigh.kneighbors(locs_proto)

    for n in n_neigh:
        r = []
        for d in distances:
            r.append(d[n])
        rev_centrality.append(r)

    df_c = df_c.assign(rev_centrality1=rev_centrality[0], rev_centrality3=rev_centrality[1], rev_centrality5=rev_centrality[2], 
    rev_centrality8=rev_centrality[3], rev_centrality10=rev_centrality[4], rev_centrality20=rev_centrality[5])

    df_c.to_csv(file_name_out+"_coll.csv", mode = "w", index=False)

    return df_c

# compute the number of locations of other vehicles in a radius d
def _compute_centrality(loc_proto_lat, loc_proto_lon, locs_proto, df_c, file_name_out):
    distances = [1000,5000,15000]
    centrality = []
    for d in distances:
        c = []

        neigh = NearestNeighbors(radius=d, metric=spherical_distance)
        neigh.fit(locs_proto) 

        for i in range(len(loc_proto_lat)):
            loc_x = loc_proto_lat[i]
            loc_y = loc_proto_lon[i]

            rng = neigh.radius_neighbors([[loc_x, loc_y]])
            nei_index = list(rng[1][0])
            
            c.append(len(nei_index))

        centrality.append(c)

    df_c = df_c.assign(centrality1K=centrality[0], centrality5K=centrality[1], centrality15K=centrality[2])

    df_c.to_csv(file_name_out+"_coll.csv", mode = "w", index=False)

    return df_c

# compute the frequency wrt the other vehicles to that location
def _compute_exclusivity(loc_proto_lat, loc_proto_lon, locs_proto, df_i, file_name_out):
    neigh = NearestNeighbors(radius=200, metric=spherical_distance) 
    neigh.fit(locs_proto) 

    exclusivity = []

    for i in range(len(loc_proto_lat)):
        loc_x = loc_proto_lat[i]
        loc_y = loc_proto_lon[i]
        sup_loc = df_i.iloc[i]["support"]

        rng = neigh.radius_neighbors([[loc_x, loc_y]])
        nei_index = list(rng[1][0])
        support_list = df_i.iloc[nei_index]["support"]
        tot_sup = sum(support_list)

        exclusivity.append(sup_loc / tot_sup)

    df_c = pd.DataFrame(exclusivity, columns=['exclusivity'] )

    df_c.to_csv(file_name_out+"_coll.csv", mode = "w", index=False)

    return df_c

# compute a set of feature of each location relative to the whole set of locations
def collective_charac (df_i, file_name_out):
    loc_proto_lat = df_i ["loc_proto_lat"]
    loc_proto_lon = df_i ["loc_proto_lon"]

    locs_proto = [list(x) for x in zip(loc_proto_lat, loc_proto_lon)]

    print("exclusivity")
    _compute_exclusivity(loc_proto_lat, loc_proto_lon, locs_proto, df_i, file_name_out)   

    df_c = pd.read_csv(file_name_out+"_coll.csv")

    print("centrality")
    _compute_centrality(loc_proto_lat, loc_proto_lon, locs_proto, df_c, file_name_out)

    df_c = pd.read_csv(file_name_out+"_coll.csv")

    print("reverse")
    _compute_rev_centrality(locs_proto, df_c, file_name_out)
    
    df_c = pd.read_csv(file_name_out+"_coll.csv")
    
    return df_c

# compute the number of seconds in a string representing time
def tot_sec(t):
    x = time.strptime(t,'%H:%M:%S')
    return datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()

# transform an array to a string of the elements
def from_array_to_string(a):
    s = ""
    for ai in a:
        s += str(ai) + ","
    return s[:-1]

# extract the movement duration and length
def _mov_stats(loc_feat):
    m = []
    mov_stats = loc_feat["mov_stats"]
    mov_stats_leave = mov_stats["leaving"]
    mov_stats_arrive = mov_stats["arriving"]
    
    m.append(0 if mov_stats_leave["duration"] == [] else np.mean(mov_stats_leave["duration"]))
    m.append(0 if mov_stats_arrive["duration"] == [] else np.mean(mov_stats_arrive["duration"]))
    m.append(0 if mov_stats_leave["length"] == [] else np.mean(mov_stats_leave["length"]))
    m.append(0 if mov_stats_arrive["length"] == [] else np.mean(mov_stats_arrive["length"]))
    m.append(1 if mov_stats_leave["duration"] == [] else np.std(mov_stats_leave["duration"]))
    m.append(1 if mov_stats_arrive["duration"] == [] else np.std(mov_stats_arrive["duration"]))
    m.append(1 if mov_stats_leave["length"] == [] else np.std(mov_stats_leave["length"]))
    m.append(1 if mov_stats_arrive["length"] == [] else np.std(mov_stats_arrive["length"]))
    return m

# compute the number of leave/arriving per timeslot
def _n_leave_arrive(loc_feat):
    time_leave = loc_feat["time_leave"]
    time_arrive = loc_feat["time_arrive"]
    time_leave_weekday = time_leave["weekdays"]
    time_leave_weekend = time_leave["weekend"]
    time_arrive_weekday = time_arrive["weekdays"]
    time_arrive_weekend = time_arrive["weekend"]

    a = []

    a.append(len(time_leave_weekday["day"]) + len(time_arrive_weekday["day"]))
    a.append(len(time_leave_weekend["day"]) + len(time_arrive_weekend["day"]))
    a.append(len(time_leave_weekday["night"]) + len(time_arrive_weekday["night"]))
    a.append(len(time_leave_weekend["night"]) + len(time_arrive_weekend["night"]))

    t_sec = [tot_sec(x) for x in time_leave_weekday["day"]]
    a.append(0 if time_leave_weekday["day"] == [] else np.mean(t_sec))
    t_sec = [tot_sec(x) for x in time_arrive_weekday["day"]]
    a.append(0 if time_arrive_weekday["day"] == [] else np.mean(t_sec))
    t_sec = [tot_sec(x) for x in time_leave_weekend["day"]]
    a.append(0 if time_leave_weekend["day"] == [] else np.mean(t_sec))
    t_sec = [tot_sec(x) for x in time_arrive_weekend["day"]]
    a.append(0 if time_arrive_weekend["day"] == [] else np.mean(t_sec))
    t_sec = [tot_sec(x) for x in time_leave_weekday["night"]]
    a.append(0 if time_leave_weekday["night"] == [] else np.mean(t_sec))
    t_sec = [tot_sec(x) for x in time_arrive_weekday["night"]]
    a.append(0 if time_arrive_weekday["night"] == [] else np.mean(t_sec))
    t_sec = [tot_sec(x) for x in time_leave_weekend["night"]]
    a.append(0 if time_leave_weekend["night"] == [] else np.mean(t_sec))
    t_sec = [tot_sec(x) for x in time_arrive_weekend["night"]]
    a.append(0 if time_arrive_weekend["night"] == [] else np.mean(t_sec))

    return a

# compute the avg and std staytime in a location
def _get_staytime(loc_feat):
    s = []
    # avg and std staying time
    if "avg_staytime" in loc_feat:
        avg_stay = loc_feat["avg_staytime"]
        avg_weekday = avg_stay["weekdays"]
        avg_weekend = avg_stay["weekend"]
        s.append(avg_weekday["day"])
        s.append(avg_weekend["day"])
        s.append(avg_weekday["night"])
        s.append(avg_weekend["night"])
    else:
        s.extend([0,0,0,0])

    #std_stay = get_avg_std_stay("std_staytime", loc_feat)
    if "std_staytime" in loc_feat:
        std_stay = loc_feat["std_staytime"]
        std_weekday = std_stay["weekdays"]
        std_weekend = std_stay["weekend"]
        s.append(std_weekday["day"])
        s.append(std_weekend["day"])
        s.append(std_weekday["night"])
        s.append(std_weekend["night"])
    else:
        s.extend([1,1,1,1])
    return s

# count how many different next location
def get_n_next_locations(loc, imn_v):
    if loc in imn_v["location_nextlocs"].keys():
        next_locs = imn_v["location_nextlocs"][loc].keys()
        if loc in next_locs:
            return len(next_locs) -1
        else:
            return len(next_locs)
    else:
        return 0

# find if a location is regular
def is_reg_location(loc, imn_v):
    if loc in imn_v["regular_locations"]:
        return 1
    else:
        return 0

# compute a set of characteristics of the locations of a vehicle
def v_individual_charac (v, imn_v):
    n_loc = imn_v["n_locs"]
    df_array = []

    for loc in range(n_loc):
        loc_str = str(loc)

        # insert the id of the vehicle
        row = [v]
        # insert the id of the location
        row.append(loc)

        # is loc regular
        is_r = is_reg_location(loc_str, imn_v)
        row.append(is_r)

        # take loc prototype
        loc_proto = imn_v["location_prototype"][loc_str]
        row.append(loc_proto[0])
        row.append(loc_proto[1])

        # number of nextlocs
        n_next_locs = get_n_next_locations(loc_str, imn_v)
        row.append(n_next_locs)

        loc_feat = imn_v["location_features"][loc_str]

        # location radius
        row.append(loc_feat["loc_rg"])
        # location entropy
        row.append(loc_feat["loc_entropy"])
        # location support
        row.append(loc_feat["loc_support"])

        # avg and std staytime
        s = _get_staytime(loc_feat)
        row.extend(s)

        # number and time of leave/arriving per timeslot
        a = _n_leave_arrive(loc_feat)
        row.extend(a)        

        # stats on the movement
        m = _mov_stats(loc_feat)
        row.extend(m)        

        df_array.append(row)

    return df_array

    
########## MAIN FUNCTION ##########

def main():

    stop, id_area, month_code, week = read_params(sys)

    file_name_in = '../../datasets/out/Traj' + stop + 'min/imn_light_area'+id_area+'_month'+month_code+'_week'+ week

    file_name_out = '../../datasets/out/Traj' + stop + 'min/loc_feat_area'+id_area+'_month'+month_code+'_week'+ week

    # load json with imn
    with open(file_name_in+'.json', 'r') as f:
        file_j = json.load(f)

    columns_df_i = ["is_regular", "loc_proto_lat", "loc_proto_lon", "n_next_locs", "radius", "entropy", "support", 
    "avg_stay_weekday_day", "avg_stay_weekend_day", "avg_stay_weekday_night", "avg_stay_weekend_night", 
    "std_stay_weekday_day", "std_stay_weekend_day", "std_stay_weekday_night", "std_stay_weekend_night",
    "n_stop_weekday_day", "n_stop_weekend_day", "n_stop_weekday_night", "n_stop_weekend_night",
    "avg_leave_weekday_day", "avg_arrive_weekday_day", "avg_leave_weekend_day", "avg_arrive_weekend_day", 
    "avg_leave_weekday_night", "avg_arrive_weekday_night", "avg_leave_weekend_night", "avg_arrive_weekend_night", 
    "avg_leave_mov_duration", "avg_arrive_mov_duration", "avg_leave_mov_length", "avg_arrive_mov_length",
    "std_leave_mov_duration", "std_arrive_mov_duration", "std_leave_mov_length", "std_arrive_mov_length"]

    columns_df_c = ["exclusivity", "centrality1K", "centrality5K", "centrality15K", 
    "rev_centrality1", "rev_centrality3", "rev_centrality5", "rev_centrality8", "rev_centrality10", "rev_centrality20"]

    categories = ["gas", "parking", "pier", "hotel", "food", "leisure", "shop", "service", "supermarket"]
    columns_df_g = ["n_"+c for c in categories]+["k_"+c for c in categories]+["d_"+c for c in categories]

    header = "vehicle,loc_id,"+ from_array_to_string(columns_df_i)+"\n"

    # write header
    with open(file_name_out+"_indiv.csv", 'w', newline='') as f:
        f.write(header)

    # for each vehicle in that area and time perform the individual characterization
    for v in file_j.keys():

        imn_v = file_j[v]

        df_v_i_array = v_individual_charac(v, imn_v)
        df_v_i = pd.DataFrame(df_v_i_array, columns=["vehicle", "loc_id"] + columns_df_i)

        df_v_i.to_csv(file_name_out+"_indiv.csv", mode = "a", header = False, index=False)

    #read individual df
    df_i = pd.read_csv(file_name_out+"_indiv.csv")
#
#    #compute collective df
#    df_c = collective_charac(df_i, file_name_out)
#    df_c.to_csv(file_name_out+"_coll.csv", mode = "w", index=False)
#
#    #read collective df
#    df_c = pd.read_csv(file_name_out+"_coll.csv")
#
#    df_ic = pd.concat([df_i, df_c], axis=1)
#
#    # compute geo df
#    df_poi = pd.read_csv('../../datasets/athens_POI.csv')
#    df_g = geographic_charac(df_poi, df_ic, file_name_out)
#
#    # read geo df
#    df_g = pd.read_csv(file_name_out+"_geo.csv")
#
#    df = pd.concat([df_ic, df_g], axis=1)
#
#    df.to_csv(file_name_out+"_complete.csv", mode = "w", index=False)
    # df = pd.read_csv(file_name_out+"_complete.csv")

    # # apply log scale to a set of features
    # cols_to_log =  ["n_next_locs", "radius", "entropy", "support", "avg_stay_weekday_day", "avg_stay_weekend_day", 
    # "avg_stay_weekday_night", "avg_stay_weekend_night", "std_stay_weekday_day", "std_stay_weekend_day", 
    # "std_stay_weekday_night", "std_stay_weekend_night", "n_stop_weekday_day", "n_stop_weekend_day", 
    # "n_stop_weekday_night", "n_stop_weekend_night", "avg_leave_weekend_day", "avg_arrive_weekend_day", 
    # "avg_leave_weekday_night", "avg_arrive_weekday_night", "avg_leave_weekend_night", "avg_arrive_weekend_night", 
    # "avg_leave_mov_duration", "avg_arrive_mov_duration", "avg_leave_mov_length", "avg_arrive_mov_length",
    # "std_leave_mov_duration", "std_arrive_mov_duration", "std_leave_mov_length", "std_arrive_mov_length",
    # "exclusivity", "rev_centrality1", "rev_centrality3", "rev_centrality5", "rev_centrality8", "rev_centrality10", "rev_centrality20"]

    # categories = ["gas", "parking", "pier", "hotel", "food", "leisure", "shop", "service", "supermarket"]
    # columns_df_g = ["n_"+c for c in categories]+["k_"+c for c in categories]+["d_"+c for c in categories]
    # cols_to_log = cols_to_log + columns_df_g
    # for col in cols_to_log:
    #     c = np.add(df[col], 1)
    #     df[col] = np.log(c)

    # df.to_csv(file_name_out+"_compl_log.csv", mode = "w", index=False)

    # # normalize df
    # min_max_scaler = preprocessing.MinMaxScaler() # StandardScaler
    # df[columns_df_i+columns_df_c+columns_df_g] = min_max_scaler.fit_transform(df[columns_df_i+columns_df_c+columns_df_g])

    # df.to_csv(file_name_out+"_compl_log_norm.csv", mode = "w", index=False)

        
    return 0


if __name__ == "__main__":
    main()