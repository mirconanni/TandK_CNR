########## IMPORT LIBRARIES ########## 

# graphs library
import networkx as nx
# date format
import datetime
# math functions
import math
# encoding of coordinates into a coded string 
from geohash import encode
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
# basic math functions
from math import log


########## IMPORT MY SCRIPTS ########## 

from detect_locations import radius_of_gyration, entropy
from distance_func import spherical_distance, closest_point_on_segment


########## FUNCTION DEFINITION ##########


# get the mean length and duration of all the movements
def get_movements_stats(movement_traj, regular_locs, trajectories):
    movement_lengths = list()
    movement_durations = list()
    reg_movement_lengths = list()
    reg_movement_durations = list()
    reg_movs = dict()
    n_reg_traj = 0

    # for each movement id
    for mid in movement_traj:
        # take the array with starting and ending location
        lft = movement_traj[mid][0]
        # get list of traj id in that movement
        traj_in_movement = movement_traj[mid][1]
        for tid in traj_in_movement:
            # list of movement lengths not just for each movement but for all
            movement_lengths.append(trajectories[tid]["length"])
            # list of movement durations not just for each movement but for all
            movement_durations.append(trajectories[tid]["duration"]) 
            if regular_locs is not None:
                # if starting and ending point of the movement are regular locations
                if str(lft[0]) in regular_locs and str(lft[1]) in regular_locs:
                    # just add the movement id in the dict of regular movement
                    reg_movs[mid] = 0
                    # append the traj length to the list of regular movement
                    reg_movement_lengths.append(trajectories[tid]["length"])
                    # append the traj duration to the list of regular movement
                    reg_movement_durations.append(trajectories[tid]["duration"])
                    # increase the number of how many traj are regular movements
                    n_reg_traj += 1

    # filter the outliers in the lengths and in the durations of all the movements
    movement_lengths = interquartile_filter(movement_lengths) 
    movement_durations = interquartile_filter(movement_durations)

    # do the same for regular movements as well
    if len(reg_movement_lengths) > 0:
        reg_movement_lengths = interquartile_filter(reg_movement_lengths)
        reg_movement_durations = interquartile_filter(reg_movement_durations)

    # convert from seconds to datetime format
    avg_mov_duration = datetime.timedelta(seconds=np.mean(movement_durations))
    std_mov_duration = datetime.timedelta(seconds=np.std(movement_durations))

    # do the same for the regular movements, if not present, take all movements
    if len(reg_movement_lengths) > 0:
        avg_reg_mov_duration = datetime.timedelta(seconds=np.mean(reg_movement_durations))
        std_reg_mov_duration = datetime.timedelta(seconds=np.std(reg_movement_durations))
    else:
        avg_reg_mov_duration = avg_mov_duration
        std_reg_mov_duration = std_mov_duration

    res = {
        'n_reg_movs': len(reg_movs),
        'avg_mov_length': np.mean(movement_lengths),
        'std_mov_length': np.std(movement_lengths),
        'avg_mov_duration': avg_mov_duration,
        'std_mov_duration': std_mov_duration,
        'avg_reg_mov_length': np.mean(reg_movement_lengths),
        'std_reg_mov_length': np.std(reg_movement_lengths),
        'avg_reg_mov_duration': avg_reg_mov_duration,
        'std_reg_mov_duration': std_reg_mov_duration,
        'n_reg_traj': n_reg_traj,
    }

    return res

# remove outliers (observations that fall below q1 − 1.5*iqr or above q3 + 1.5*iqr
def interquartile_filter(x):
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    # compute the interquartile range
    iqr = q3 - q1
    y = list()
    # remove outliers
    for xi in x:
        if q1 - 1.5 * iqr <= xi <= q3 + 1.5 * iqr:
            y.append(xi)
    return y

# get mean of duration and length of each movement
def get_movements_features(movement_traj, trajectories):

    res = dict()
    for mid in movement_traj:
        # takes the list of trajectories id of that movement
        traj_in_movement = movement_traj[mid][1]
        movement_support = len(traj_in_movement)
        movement_lengths = list()
        movement_durations = list()
        # for each traj in that movement store its length and duration
        for tid in traj_in_movement:
            movement_lengths.append(trajectories[tid]["length"])
            movement_durations.append(trajectories[tid]["duration"])

        # remove outliers
        movement_lengths = interquartile_filter(movement_lengths)
        movement_durations = interquartile_filter(movement_durations)

        res[mid] = {
            'mov_support': movement_support,
            'typical_mov_length': np.median(movement_lengths),
            'avg_mov_length': np.mean(movement_lengths),
            'std_mov_length': np.std(movement_lengths),
            'typical_mov_duration': datetime.timedelta(seconds=np.median(movement_durations)),
            'avg_mov_duration': datetime.timedelta(seconds=np.mean(movement_durations)),
            'std_mov_duration': datetime.timedelta(seconds=np.std(movement_durations)),
        }

    return res

# divide the hours of the day in 2 slots: day, night
def _get_timeslot(time_loc):
    if datetime.time(8) < time_loc.time() <= datetime.time(19):
        return 'day'
    else:
        return 'night'

# compute the number of seconds in the 
def _count_day_night_time(al):
    day_time = 0
    night_time = 0
    
    # add 0-8 hours
    if al[0].hour < 8:
        if al[1].hour < 8:
            return 0, (al[1]- al[0]).total_seconds()
        else:
            night_time = (28800 - al[0].hour*3600 - al[0].minute*60 - al[0].second)
            al[0] = al[0].replace(hour = 8, minute = 0, second = 0)
    # add 8-19 hours
    if 8 < al[1].hour < 19:
        return (al[1]- al[0]).total_seconds(), night_time
    if al[0].hour < 19:
        day_time = (68400 - al[0].hour*3600 - al[0].minute*60 - al[0].second)
        al[0] = al[0].replace(hour = 19, minute = 0, second = 0)
        # add 19-24 hours
    if al[1].hour > 19:
        night_time += (al[1]- al[0]).total_seconds()
    return day_time, night_time

# compute the avg and the std staytime divided in timeslots
def _compute_loc_staytime(loc_arrive_leave_list):
    staytime_durations = dict()
    loc_arrive_leave_list = default_to_regular(loc_arrive_leave_list)

    # for each location take the list of arrive and leave date
    for loc, arrive_leave_list in loc_arrive_leave_list.items():
        if loc not in staytime_durations:
            staytime_durations[loc] =  {'weekdays': {'day': list(), 'night': list()}, 'weekend': {'day': list(), 'night': list()}} 

        for al in arrive_leave_list:
            # if trajectory is in the same day
            if al[0].day == al[1].day:
                type_of_day = 'weekend' if al[0].weekday() >= 5 else 'weekdays'
                day_time, night_time = _count_day_night_time(al)                
                staytime_durations[loc][type_of_day]["day"].append(day_time)
                staytime_durations[loc][type_of_day]["night"].append(night_time)        
            # if multiple days span
            else:
                # add the rest of the first day
                tot_day_time = 0
                tot_night_time = 0
                type_of_day = 'weekend' if al[0].weekday() >= 5 else 'weekdays'
                day_time, night_time = _count_day_night_time([al[0], al[0].replace(hour = 23, minute = 59, second = 59)])
                tot_day_time += day_time
                tot_night_time += night_time             
                # add middle days
                second_day = al[0] + datetime.timedelta(days=1)
                second_last_day = al[1] - datetime.timedelta(days=1)
                for d in pd.date_range(second_day, second_last_day, normalize = True, freq='D'):
                    type_of_day = 'weekend' if d.weekday() >= 5 else 'weekdays'
                    tot_day_time += 39600
                    tot_night_time += 46800    
                # add the last day
                type_of_day = 'weekend' if al[1].weekday() >= 5 else 'weekdays'
                day_time, night_time = _count_day_night_time([al[1].replace(hour = 0, minute = 0, second = 0), al[1]])
                tot_day_time += day_time
                tot_night_time += night_time                
                staytime_durations[loc][type_of_day]["day"].append(tot_day_time)
                staytime_durations[loc][type_of_day]["night"].append(tot_night_time)
                
    # add the mean and the average staying time for each location in time slot
    res = dict()
    for loc, time_slot in staytime_durations.items():
        weekday = time_slot["weekdays"]
        weekend = time_slot["weekend"]

        mean_weekday_day = 0 if weekday["day"] == [] else np.mean(weekday["day"])
        mean_weekend_day = 0 if weekend["day"] == [] else np.mean(weekend["day"])
        mean_weekday_night = 0 if weekday["night"] == [] else np.mean(weekday["night"])
        mean_weekend_night = 0 if weekend["night"] == [] else np.mean(weekend["night"])

        std_weekday_day = 0 if weekday["day"] == [] else np.std(weekday["day"])
        std_weekend_day = 0 if weekend["day"] == [] else np.std(weekend["day"])
        std_weekday_night = 0 if weekday["night"] == [] else np.std(weekday["night"])
        std_weekend_night = 0 if weekend["night"] == [] else np.std(weekend["night"])
        
        res[loc] = {
            'avg_staytime': {'weekdays': 
                                {'day': mean_weekday_day, 'night': mean_weekday_night}, 
                            'weekend': 
                                {'day': mean_weekend_day, 'night': mean_weekend_night}
                            },
            'std_staytime':  {'weekdays': 
                                {'day': std_weekday_day, 'night': std_weekday_night}, 
                            'weekend': 
                                {'day': std_weekend_day, 'night': std_weekend_night}
                            }
        }
    return res

# counts the staying time in each location divided also for weekdays/weekends and hourly timeslots
def get_locations_durations(trajectories, traj_from_to_loc, loc_ft_mid):

    loc_arrive_leave_list = defaultdict(list)
    last_loc_arrive = None
    last_time_arrive = None

    loc_timeslot_set_andrienko = dict()
    mov_timeslot_set_andrienko = dict()

    # take the list of traj id ordered by the time stamp of the first point in the trajectory
    sorted_traj = sorted(trajectories, key=lambda t: trajectories[t][0][2])

    # for each traj take the starting and ending location, the timestamp of leaving and arrival
    for tid in sorted_traj:
        traj = trajectories[tid]
        loc_leave = traj_from_to_loc[str(tid)][0]
        loc_arrive = traj_from_to_loc[str(tid)][1]
        time_leave = datetime.datetime.fromtimestamp(traj[0][2]*1000)
        time_arrive = datetime.datetime.fromtimestamp(traj[-1][2]*1000)

        # if it's not the first trajectory
        if last_loc_arrive is not None:
            # if starting location is the same as the arrival location before
            if loc_leave == last_loc_arrive:
                # store at that location the staying time
                loc_arrive_leave_list[last_loc_arrive].append([last_time_arrive, time_leave])

            # for each hour from the arrival of the previous trajectory to the leaving time of this one
            for ts in pd.date_range(last_time_arrive, time_leave, freq='H'):
                # if the last arriving location is new
                if last_loc_arrive not in loc_timeslot_set_andrienko:
                    # create a new dict for that location for weekdays and weekends
                    loc_timeslot_set_andrienko[last_loc_arrive] = {'weekdays': defaultdict(set), 'weekend': defaultdict(set)}
                # compute if it's a weedday or a weekend
                type_of_day = 'weekend' if ts.weekday() >= 5 else 'weekdays'
                # divide the hours of the day in 4 timeslots
                timeslot = _get_timeslot(ts)
                day_key = (ts.year, ts.month, ts.day)
                # add the date of the location in a dict from location, weekdays/weekend and hour timeslot
                loc_timeslot_set_andrienko[last_loc_arrive][type_of_day][timeslot].add(day_key)

        # store the arrival location of the trajectory and the time
        last_loc_arrive = loc_arrive
        last_time_arrive = time_arrive

        # if the leaving location is new
        if loc_leave not in loc_timeslot_set_andrienko:
            # create a new dict for that location for weekdays and weekends
            loc_timeslot_set_andrienko[loc_leave] = {'weekdays': defaultdict(set), 'weekend': defaultdict(set)}
        # compute if it's a weedday or a weekend
        type_of_day = 'weekend' if time_leave.weekday() >= 5 else 'weekdays'
        # divide the hours of the day in 4 timeslots
        timeslot = _get_timeslot(time_leave)
        day_key = (time_leave.year, time_leave.month, time_leave.day)
        # add the date of the location in a dict from location, weekdays/weekend and hour timeslot
        loc_timeslot_set_andrienko[loc_leave][type_of_day][timeslot].add(day_key)

        # do the same for hte arrival location
        if loc_arrive not in loc_timeslot_set_andrienko:
            loc_timeslot_set_andrienko[loc_arrive] = {'weekdays': defaultdict(set), 'weekend': defaultdict(set)}
        type_of_day = 'weekend' if time_arrive.weekday() >= 5 else 'weekdays'
        timeslot = _get_timeslot(time_arrive)
        day_key = (time_arrive.year, time_arrive.month, time_arrive.day)
        loc_timeslot_set_andrienko[loc_arrive][type_of_day][timeslot].add(day_key)

        # take the movement id for that couple of locations
        mid = loc_ft_mid[('('+str(loc_leave)+', '+ str(loc_arrive)+')')]
        # build the same type of dict also for that movement (using the time of leave)
        if mid not in mov_timeslot_set_andrienko:
            mov_timeslot_set_andrienko[mid] = {'weekdays': defaultdict(set), 'weekend': defaultdict(set)}
        type_of_day = 'weekend' if time_leave.weekday() >= 5 else 'weekdays'
        timeslot = _get_timeslot(time_leave)
        day_key = (time_leave.year, time_leave.month, time_leave.day)
        mov_timeslot_set_andrienko[mid][type_of_day][timeslot].add(day_key)

    # compute the avg and the std staytime divided in timeslots
    res = _compute_loc_staytime(loc_arrive_leave_list)

    # remove the defaul dict
    # at the end loc_timeslot_count_andrienko is a dict from location to the count of visits to that place
    # divided according to weeddays/weekends and hourly timeslot
    loc_timeslot_count_andrienko = dict()
    for loc in loc_timeslot_set_andrienko:
        loc_timeslot_count_andrienko[loc] = {'weekdays': dict(), 'weekend': dict()}
        for type_of_day in loc_timeslot_set_andrienko[loc]:
            for timeslot in loc_timeslot_set_andrienko[loc][type_of_day]:
                val = len(loc_timeslot_set_andrienko[loc][type_of_day][timeslot])
                loc_timeslot_count_andrienko[loc][type_of_day][timeslot] = val

    # same with the movements ids
    mov_timeslot_count_andrienko = dict()
    for mid in mov_timeslot_set_andrienko:
        mov_timeslot_count_andrienko[mid] = {'weekdays': dict(), 'weekend': dict()}
        for type_of_day in mov_timeslot_set_andrienko[mid]:
            for timeslot in mov_timeslot_set_andrienko[mid][type_of_day]:
                val = len(mov_timeslot_set_andrienko[mid][type_of_day][timeslot])
                mov_timeslot_count_andrienko[mid][type_of_day][timeslot] = val

    return res, loc_timeslot_count_andrienko, mov_timeslot_count_andrienko

# transform a default dict to a regular dict (removes the default values)
def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d

# compute some statistics regarding a single location
def _get_location_features(points_in_loc, traj_from_to_loc, location_prototype, trajectories, df_v_stops):
    # list of points id, sorted according to the timestamp
    sorted_points = sorted(points_in_loc, key=lambda x: points_in_loc[x][2])

    #staytime_dist = defaultdict(int)
    nextloc_count = defaultdict(int)
    #nextloc_dist = defaultdict(lambda: defaultdict(int))

    # take each couple of consecutive points
    for i in range(len(sorted_points)-1):
        pid1 = sorted_points[i]

        # extract if first point is "from" or "to"
        arriving_leaving = points_in_loc[pid1][3]

        # if it's the first point of a trajectory
        if arriving_leaving == 'f':
            # take the traj id
            tid = points_in_loc[pid1][4]
            # take the location id of the end of the trajectory
            next_loc = traj_from_to_loc[str(tid)][1]
            # count the number of traj arriving at that location
            nextloc_count[next_loc] += 1

    # count number of leave and arrive per location in timeslot
    time_leave_dict = {'weekdays': {'day': list(), 'night': list()}, 'weekend': {'day': list(), 'night': list()}}
    time_arrive_dict = {'weekdays': {'day': list(), 'night': list()}, 'weekend': {'day': list(), 'night': list()}}
    mov_stats = {'leaving': {'duration': list(), 'length': list()}, 'arriving': {'duration': list(), 'length': list()}}
    for _, pi_array in points_in_loc.items():
        time = datetime.datetime.fromtimestamp(pi_array[2]*1000)
        type_of_day = 'weekend' if time.weekday() >= 5 else 'weekdays'
        timeslot = _get_timeslot(time)
        tid = pi_array[4]
        traj_length = df_v_stops[df_v_stops["tid"] == tid].iloc[0]["length"]
        traj_duration = df_v_stops[df_v_stops["tid"] == tid].iloc[0]["duration"]

        if pi_array[3] == 'f':
            time_leave_dict[type_of_day][timeslot].append(time.time())
            mov_stats["leaving"]["duration"].append(traj_duration)
            mov_stats["leaving"]["length"].append(traj_length)
        else:
            time_arrive_dict[type_of_day][timeslot].append(time.time())
            mov_stats["arriving"]["duration"].append(traj_duration)
            mov_stats["arriving"]["length"].append(traj_length)

    # take only the coords of the points in that location
    spatial_points = list()
    for p in list(points_in_loc.values()):
        spatial_points.append(p[0:2])
    # compute the radius of gyration and the entropy 
    loc_rg = radius_of_gyration(spatial_points, location_prototype, spherical_distance)
    loc_entropy = entropy(list(nextloc_count.values()), classes=len(nextloc_count))

    res = {
        'nextloc_count': nextloc_count,
        'time_leave': time_leave_dict,
        'time_arrive': time_arrive_dict,
        'mov_stats' : mov_stats,
        'loc_rg': loc_rg,
        'loc_entropy': loc_entropy,
    }

    return res

# compute all the statistics about all the locations
def get_locations_features(points, traj_from_to_loc, location_points, location_prototype, trajectories, df_v_stops):
    res = dict()
    # for each location id
    for lid in location_points:
        # make a subset dict from point id to coord in that location
        points_in_loc = dict()
        for pid in location_points[lid]:
            points_in_loc[pid] = points[pid]

        # get the dict about each location
        lf_res = _get_location_features(points_in_loc, traj_from_to_loc, location_prototype[lid], trajectories, df_v_stops)
        # add the support to the statistics of that location
        lf_res['loc_support'] = len(location_points[lid])
        # store the pointer to that location
        res[lid] = lf_res

    return res

# decide the support threshold to be a regular location
def _get_minimum_support(locations_support):
    x = []
    y = []

    # sort the number of neighbours of each location from least central to most
    sorted_support = sorted(locations_support)

    for i, s in enumerate(sorted_support):
        x.append(1.0 * i)
        y.append(1.0 * s)

    max_d = -float('infinity')
    index = 0

    # takes min and max center location in the graph
    a = [x[0], y[0]]
    b = [x[-1], y[-1]]

    # looks for the biggest knee in the distribution of the number of neighbours
    for i in range(len(x)):
        p = [x[i], y[i]]
        c = closest_point_on_segment(a, b, p)
        # compute the distance from p to c
        d = math.sqrt((c[0]-x[i])**2 + (c[1]-y[i])**2)
        if d > max_d:
            max_d = d
            index = i

    return sorted_support[index]

############### DON'T LIKE IT VERY MUCH. PROBLEMS:
######################## IT IS BASED ON THE NUMBER OF NEIGHBOURS, NOT THE FREQUENCY OF A LOCATION
######################## -> IT DOESN'T CONSIDER IF A LOCATION IS REACHED ONLY IN A PATTERN 
######################## (GO TO B ONLY AFTER BEING TO A, AND GO TO C EVERY TIME AFTER)
########### THE WAY TO GET MIN SUP IS TOO STRICT, CAN INCLUDE MORE LOCATIONS
############### DOESN'T RETURN IF IT'S A DAG GRAPH, JUST REMOVES SINK NODES
# filter only the most central locations in the graph
def detect_regular_locations(location_points, loc_nextlocs):  
    # create a dict from each loc to the number of visited locations next
    loc_support = dict()
    for lid in location_points:
        loc_support[lid] = len(location_points[lid])

    # find the minimum number of neighbours to be considered a regular location
    loc_min_sup = _get_minimum_support(list(loc_support.values()))

    # filter only the regular locations
    # create a dict from id regular location to neighbours and frequency
    regular_locs = dict()
    for lid in loc_support:      
        if loc_support[lid] >= loc_min_sup:
            regular_locs[lid] = {}
            # if location is not sink hole
            if lid in loc_nextlocs:
                regular_locs[lid] = loc_nextlocs[lid]

    return regular_locs, loc_min_sup