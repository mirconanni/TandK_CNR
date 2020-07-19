import os
import sys
sys.path.append("..")
import gzip
import json
import datetime
import numpy as np
import pandas as pd
import networkx as nx
from networkx.readwrite import json_graph
from Crash_prediction.Helpers.TaK_Mongo_Connector import TaK_Mongo_Connector
from Crash_prediction.individual_mobility_network import build_imn
from Crash_prediction.Helpers.trajectory import Trajectory
import time

import Crash_prediction.crash_config as cfg

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


def imn_extract(area, type_user, overwrite=False):
    # Output file name
    output_filename = cfg.store_files["path_imn"] + '%s_imn_%s.json.gz' % (area, type_user)

    # Making connection to the MongoDB database (host, port, database name)
    mongo_connector = TaK_Mongo_Connector(cfg.mongodb["host"], cfg.mongodb["port"], cfg.mongodb["db"],
                                          cfg.mongodb["user"], cfg.mongodb["password"])

    users_list = pd.read_csv(cfg.users_file, header= None).values[:, 0].tolist()
    users_list = sorted(users_list)
    print("Number of users ", len(users_list))

    nbr_users = len(users_list)
    print(nbr_users, len(users_list))

    # if overwrite==False, read the output file and remove those users for which IMN is created.
    if os.path.isfile(output_filename) and not overwrite:
        processed_users = list()
        fout = gzip.GzipFile(output_filename, 'r')
        for row in fout:
            customer_obj = json.loads(row)
            processed_users.append(customer_obj['uid'])
        fout.close()
        print(processed_users)
        users_list = [uid for uid in users_list if uid not in processed_users]

    print(nbr_users, len(users_list))

    # Building the base query for retrieving Individual Mobility History (IMH)
    input_query = {"adaptive": cfg.traj_seg["adaptive"], "temporal_thr": cfg.traj_seg["time_treshold"],
                   "spatial_thr": cfg.traj_seg["space_treshold"], "max_speed": cfg.traj_seg["max_speed"],
                   "min_length": cfg.imn["min_length"], "min_duration": cfg.imn["min_duration"],
                   "events_crashes": True, "from_date": cfg.imn["from_date"], "to_date": cfg.imn["to_date"]}

    for i, uid in enumerate(users_list):
        input_query['user_id'] = uid
        start_time = time.time()
        if i % 1 == 0:
            print(datetime.datetime.now(), '%s [%s/%s] - %.2f' % (
                uid, i+1,  nbr_users, i / nbr_users * 100.0))

        imh, events, crashes = mongo_connector.load_imh(cfg.mongodb["input_collection"], **input_query)

        if len(imh['trajectories']) < cfg.imn["min_traj_nbr"]:
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
            if len(wimh['trajectories']) < cfg.imn["min_traj_nbr"] // 12:
                continue
            print("buidling IMN for %s" % stk)
            imn = build_imn(wimh, reg_loc=True, events=wevents, verbose=False)
            customer_obj[stk] = imn

        print("--- %s seconds building imn ---" % (time.time() - start_time))

        json_str = '%s\n' % json.dumps(clear_tuples4json(customer_obj), default=agenda_converter)
        json_bytes = json_str.encode('utf-8')
        with gzip.GzipFile(output_filename, 'a') as fout:
           fout.write(json_bytes)
    print("Done!")
    return 0


def main():
    area = sys.argv[1]
    type_user = sys.argv[2]  # 'crash' 'nocrash'
    overwrite = str2bool(sys.argv[3])

    imn_extract(area, type_user, overwrite)

if __name__ == "__main__":
    main()