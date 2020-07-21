import sys
#sys.path.append('/Users/omid/IdeaProjects/TandK_CNR/')
sys.path.append("..")

import json
import datetime
import numpy as np
import pandas as pd
import networkx as nx
from networkx.readwrite import json_graph
from IMN_extraction.Helpers.TaK_Mongo_Connector import TaK_Mongo_Connector
from IMN_extraction.individual_mobility_network import build_imn
from IMN_extraction.Helpers.trajectory import Trajectory
import time

__author__ = 'Omid Isfahani Alamdari'

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

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


def clear_tuples4json(o):
    if isinstance(o, dict):
        return {key2str(k): clear_tuples4json(o[k]) for k in o}
    return o


def agenda_converter(o):
    if isinstance(o, datetime.datetime):
        return o.__str__()
    elif isinstance(o, datetime.timedelta):
        return o.__str__()
    elif isinstance(o, Trajectory):
        return o.to_json()
    elif isinstance(o, nx.DiGraph):
        return json_graph.node_link_data(o, {'link': 'edges', 'source': 'from', 'target': 'to'})
    else:
        return o.__str__()


def start_time_map(t):
    m = datetime.datetime.fromtimestamp(t).month
    if m == 1:
        return '01-02', None
    elif m == 12:
        return '11-12', None
    else:
        return '%02d-%02d' % (m-1, m), '%02d-%02d' % (m, m+1)

def build_imn_proc(values):
    wimh, wevents, stk, uid = values
    imn = build_imn(wimh, reg_loc=True, events=wevents, verbose=False)

    customer_obj = {'uid': uid}
    customer_obj[stk] = imn

    json_str = '%s\n' % json.dumps(clear_tuples4json(customer_obj), default=agenda_converter)

    return json_str  #imn


def imn_extract(mongo_host, mongo_port, database, input_collection, output_collection, filename, adaptive,
                max_speed, space_treshold, time_treshold, from_date, to_date, min_traj_nbr, min_length, min_duration):

    # Making connection to the MongoDB database (host, port, database name)
    mongo_connector = TaK_Mongo_Connector(mongo_host, mongo_port, database)

    users_list = pd.read_csv(filename, header= None).values[:, 0].tolist()
    users_list = sorted(users_list)
    print("Number of users ", len(users_list))

    nbr_users = len(users_list)
    print(nbr_users, len(users_list))

    # Building the base query for retrieving Individual Mobility History (IMH)
    input_query = {"adaptive": adaptive, "temporal_thr": time_treshold, "spatial_thr": space_treshold,
                   "max_speed": max_speed, "min_length": min_length, "min_duration": min_duration, "events_crashes": True,
                   "from_date": from_date, "to_date": to_date}

    for i, uid in enumerate(users_list):
        input_query['user_id'] = uid
        start_time = time.time()
        if i % 1 == 0:
            print(datetime.datetime.now(), '%s [%s/%s] - %.2f' % (
                uid, i+1,  nbr_users, i / nbr_users * 100.0))

        imh, events, crashes = mongo_connector.load_imh(input_collection, **input_query)

        if len(imh['trajectories']) < min_traj_nbr:
            print("No trajectories %s" % uid)
            continue

        print("Total number of trajectories extracted %s" % len(imh['trajectories']))
        print("Total number of events %s" % len(events))
        print("Total number of crashes %s" % len(crashes))

        wimh_dict = dict()
        wevents_dict = dict()
        for tid, traj in imh['trajectories'].items():
            st = traj.start_time()
            stk_list = start_time_map(st)
            for stk in stk_list:
                if stk is None:
                    continue
                if stk not in wimh_dict:
                    wimh_dict[stk] = {'uid': uid, 'trajectories': dict()}
                    wevents_dict[stk] = dict()
                wimh_dict[stk]['trajectories'][tid] = traj
                if tid in events:
                    wevents_dict[stk][tid] = events[tid]

        customer_obj = {'uid': uid}

        for stk in wimh_dict:
            wimh = wimh_dict[stk]
            wevents = wevents_dict[stk]
            if len(wimh['trajectories']) < min_traj_nbr // 12:
                continue
            print("buidling IMN for %s" % stk)
            imn = build_imn(wimh, reg_loc=True, events=wevents, verbose=False)
            customer_obj[stk] = imn

        print("--- %s seconds building imn ---" % (time.time() - start_time))

        # Converting the python objects to document
        json_str = '%s' % json.dumps(clear_tuples4json(customer_obj), default=agenda_converter)
        res = json.loads(json_str)

        # insert the IMN document to the MongoDB collection
        mongo_connector.insert_one(output_collection, res)
    print("Done!")
    return 0


def main():
    # mongo_host = "localhost"  # MongoDB host
    # mongo_port = "27017"  # MongoDB port
    # database = "test"  # MongoDB database
    # input_collection = "sisdataset4"  # MongoDB input collection containing user history
    # output_collection = "user_imns"  # MongoDB output collection for IMNs
    #
    # # The file containing the id of users for which IMNs will be extracted
    # users_filename = "users.txt"
    #
    # # Segmentation parameters
    # adaptive = False
    # max_speed = 0.07
    # space_treshold = 0.05
    # time_treshold = 1200
    #
    # # IMN extraction parameters
    # from_date = "2017-04-01T00:00:00.000"
    # to_date = "2017-06-01T00:00:00.000"
    # min_traj_nbr = 100  # minimum number of trajectories in the period to start building IMNs
    # min_length = 1.0  # minimum spatial length of an extracted trajectory
    # min_duration = 60.0  # minimum duration of an extracted trajectory

    mongo_host = sys.argv[1]  # MongoDB host
    mongo_port = sys.argv[2]  # MongoDB port
    database = sys.argv[3]  # MongoDB database
    input_collection = sys.argv[4]  # MongoDB input collection containing user history
    output_collection = sys.argv[5]  # MongoDB output collection for IMNs

    # The file containing the id of users for which IMNs will be extracted
    users_filename = sys.argv[6]

    # Segmentation parameters
    adaptive = str2bool(sys.argv[7])
    max_speed = float(sys.argv[8])
    space_treshold = float(sys.argv[9])
    time_treshold = int(sys.argv[10])

    # IMN extraction parameters
    from_date = sys.argv[11]
    to_date = sys.argv[12]
    min_traj_nbr = int(sys.argv[13])  # minimum number of trajectories in the period to start building IMNs
    min_length = float(sys.argv[14])  # minimum spatial length of an extracted trajectory
    min_duration = int(sys.argv[15])

    imn_extract(mongo_host, mongo_port, database, input_collection, output_collection, users_filename, adaptive,
                max_speed, space_treshold, time_treshold, from_date, to_date, min_traj_nbr, min_length, min_duration)


if __name__ == "__main__":
    main()