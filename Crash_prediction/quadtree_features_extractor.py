import os
import sys
sys.path.append('..')
import gzip
import json
import datetime
import numpy as np
import pandas as pd
from shapely.geometry import Point

from Crash_prediction.Helpers.TaK_Mongo_Connector import TaK_Mongo_Connector
from Crash_prediction.tak_quadtree import lon_lat_to_quadtree_path
import Crash_prediction.crash_config as cfg


def haversine_np(p1, p2):
    """
    Calculate the great circle distance between two shapely points
    on the earth (specified in decimal degrees)
    All args must be of equal length.
    Parameters
    ----------
    p1, p2: shapely points
    Returns
    -------
    km: float
        the earth distance between the two points
    """
    lon1, lat1 = p1.x, p1.y
    lon2, lat2 = p2.x, p2.y
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

def calculate_speed(p0, p1):
    # try:
    lon0, lat0, ts0 = p0
    lon1, lat1, ts1 = p1
    space = haversine_np(Point((lon0, lat0)), Point((lon1, lat1)))
    time = (ts1 - ts0) / 3600
    if time > 0:
        speed = space / time
    else:
        speed = 0
    # except RuntimeWarning:
    #     print(speed, space, time)
    #     raise Exception
    #     return None

    # if np.isnan(speed) or np.isinf(speed):
    #     return 0.0
    return speed


def add_tree_feature(quadtree_features, path, feature, value=1):
    if path not in quadtree_features:
        quadtree_features[path] = {
            'nbr_traj_start': 0,
            'nbr_traj_stop': 0,
            'nbr_traj_move': 0,
            'traj_speed_sum': 0,
            'traj_speed_count': 0,
            'nbr_evnt_A': 0,
            'nbr_evnt_B': 0,
            'nbr_evnt_C': 0,
            'nbr_evnt_Q': 0,
            'nbr_evnt_start': 0,
            'nbr_evnt_stop': 0,
            'speed_A_sum': 0,
            'max_acc_A_sum': 0,
            'avg_acc_A_sum': 0,
            'speed_B_sum': 0,
            'max_acc_B_sum': 0,
            'avg_acc_B_sum': 0,
            'speed_C_sum': 0,
            'max_acc_C_sum': 0,
            'avg_acc_C_sum': 0,
            'speed_Q_sum': 0,
            'max_acc_Q_sum': 0,
            'avg_acc_Q_sum': 0,
            'nbr_crash': 0,
        }
    quadtree_features[path][feature] += value


def quadtrees_features_extract(quadtrees_features, quadtree_data, depth):

    for index in quadtree_data:
        for tid, traj in quadtree_data[index]['trajectories'].items():
            for i, point in enumerate(traj.object):
                lon, lat, _ = point
                path = lon_lat_to_quadtree_path(lon, lat, depth)
                if i == 0:
                    add_tree_feature(quadtrees_features[index], path, 'nbr_traj_start', value=1)
                elif i == len(traj.object) - 1:
                    add_tree_feature(quadtrees_features[index], path, 'nbr_traj_stop', value=1)
                    speed = calculate_speed(traj.object[i - 1], point)
                    add_tree_feature(quadtrees_features[index], path, 'traj_speed_sum', value=speed)
                    add_tree_feature(quadtrees_features[index], path, 'traj_speed_count', value=1)
                else:
                    add_tree_feature(quadtrees_features[index], path, 'nbr_traj_move', value=1)
                    speed = calculate_speed(traj.object[i - 1], point)
                    add_tree_feature(quadtrees_features[index], path, 'traj_speed_sum', value=speed)
                    add_tree_feature(quadtrees_features[index], path, 'traj_speed_count', value=1)

        for event in quadtree_data[index]['events'].values():
            event_type = event['event_type']
            avg_acc = event['avg_acc']
            max_acc = event['max_acc']
            speed = event['speed']
            lon = event['lon']
            lat = event['lat']
            path = lon_lat_to_quadtree_path(lon, lat, depth)
            add_tree_feature(quadtrees_features[index], path, 'nbr_evnt_%s' % event_type, value=1)
            if event_type not in ['start', 'stop']:
                add_tree_feature(quadtrees_features[index], path, 'speed_%s_sum' % event_type, value=speed)
                add_tree_feature(quadtrees_features[index], path, 'max_acc_%s_sum' % event_type, value=max_acc)
                add_tree_feature(quadtrees_features[index], path, 'avg_acc_%s_sum' % event_type, value=avg_acc)

        if quadtree_data[index]['crash'] is not None:
            lon = quadtree_data[index]['crash']['lon']
            lat = quadtree_data[index]['crash']['lat']
            path = lon_lat_to_quadtree_path(lon, lat, depth)
            add_tree_feature(quadtrees_features[index], path, 'nbr_crash', value=1)

    return quadtrees_features


def main():
    area = sys.argv[1]
    overwrite = True
    depth = 16
    store_evry = 100

    mongo_connector = TaK_Mongo_Connector(cfg.mongodb["host"], cfg.mongodb["port"], cfg.mongodb["db"],
                                          cfg.mongodb["user"], cfg.mongodb["password"])


    users_filename = cfg.users_file
    users_list = sorted(pd.read_csv(users_filename).values[:, 0].tolist())

    quadtree_output_filename = cfg.store_files["path_quadtree"] + '%s_quadtree_features.json.gz' % area
    quadtrees_features = dict()

    datetime_from = datetime.datetime.strptime(cfg.imn["from_date"], '%Y-%m-%dT%H:%M:%S.%f')
    datetime_to = datetime.datetime.strptime(cfg.imn["to_date"], '%Y-%m-%dT%H:%M:%S.%f')

    months = pd.date_range(start=datetime_from, end=datetime_to, freq='MS')
    boundaries = [[lm, um] for lm, um in zip(months[:-1], months[1:])]

    index = 0
    data_map = dict()
    for months in boundaries:
        data_map[tuple(months)] = index
        quadtrees_features[index] = dict()
        index += 1

    last_processed_user = None
    if os.path.isfile(quadtree_output_filename) and not overwrite:
        fout = gzip.GzipFile(quadtree_output_filename, 'r')
        quadtrees_features_str = json.loads(fout.readline())
        quadtrees_features = {int(k): v for k, v in quadtrees_features_str.items()}
        last_processed_user = json.loads(fout.readline())
        fout.close()

    input_query = {"adaptive": cfg.traj_seg["adaptive"], "temporal_thr": cfg.traj_seg["time_treshold"],
                   "spatial_thr": cfg.traj_seg["space_treshold"], "max_speed": cfg.traj_seg["max_speed"],
                   "min_length": cfg.imn["min_length"], "min_duration": cfg.imn["min_duration"],
                   "events_crashes": True, "from_date": cfg.imn["from_date"], "to_date": cfg.imn["to_date"]}

    for i, uid in enumerate(users_list):
        if last_processed_user is not None and uid <= last_processed_user:
            continue

        if i % store_evry == 0:
            print(datetime.datetime.now(), '%s %.2f' % (area, i / len(users_list) * 100.0))

        input_query['user_id'] = uid
        imh, events, crashes = mongo_connector.load_imh(cfg.mongodb["input_collection"], **input_query)
        trajectories = imh['trajectories']

        quadtree_data = dict()

        # partitioning trajectories for train and test
        for tid, traj in trajectories.items():
            for lu, index in data_map.items():
                if lu[0] <= pd.Timestamp(traj.start_time(), unit='s') < lu[1]:
                    if index not in quadtree_data:
                        quadtree_data[index] = {'uid': uid, 'crash': None, 'trajectories': dict(), 'events': dict(), }
                    quadtree_data[index]['trajectories'][tid] = traj

        # partitioning events for train and test
        for eid, evnt in events.items():
            for lu, index in data_map.items():
                if lu[0] <= pd.Timestamp(evnt[0]['date'], unit='s') < lu[1] and index in quadtree_data:
                    quadtree_data[index]['events'][eid] = evnt[0]

        # get has crash this month
        for lu, index in data_map.items():
            if index not in quadtree_data:
                continue

            crashes_month = []
            for k, v in crashes.items():
                crash_time = pd.Timestamp(v[0]['date'], unit='s')
                if lu[0] <= crash_time < lu[1]:
                    crashes_month.append(v[0])
            if len(crashes_month) > 0:
                quadtree_data[index]['crash'] = {'lat': crashes_month[0]['lat'], 'lon': crashes_month[0]['lon']}

        quadtrees_features = quadtrees_features_extract(quadtrees_features, quadtree_data, depth)

        if i % store_evry == 0:
            json_str_quadtree = '%s\n' % json.dumps(quadtrees_features)
            json_bytes_quadtree = json_str_quadtree.encode('utf-8')
            json_str_lpu = '%s\n' % json.dumps(last_processed_user)
            json_bytes_lpu = json_str_lpu.encode('utf-8')
            with gzip.GzipFile(quadtree_output_filename, 'w') as fout:
                fout.write(json_bytes_quadtree)
                fout.write(json_bytes_lpu)
            last_processed_user = uid

        # break


if __name__ == "__main__":
    main()