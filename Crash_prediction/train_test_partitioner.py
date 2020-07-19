import os
import sys
sys.path.append('..')
import json
import gzip
import datetime
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from Crash_prediction.Helpers.TaK_Mongo_Connector import TaK_Mongo_Connector
from Crash_prediction.feature_extractor import extract_features, store_features

import Crash_prediction.crash_config as cfg

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def main():
    area = sys.argv[1]
    type_user = sys.argv[2]  # 'crash' 'nocrash'
    overwrite = str2bool(sys.argv[3])

    window = cfg.crash["window"]
    datetime_from = datetime.datetime.strptime(cfg.imn["from_date"], '%Y-%m-%dT%H:%M:%S.%f')
    datetime_to = datetime.datetime.strptime(cfg.imn["from_date"], '%Y-%m-%dT%H:%M:%S.%f')

    path_imn = cfg.store_files["path_imn"]
    path_traintest = cfg.store_files["path_traintest"]
    path_quadtree = cfg.store_files["path_quadtree"]

    print(datetime.datetime.now(), 'Crash Prediction - Train Test Partitioner')
    if not overwrite:
        print(datetime.datetime.now(), '(restart)')

    print(datetime.datetime.now(), 'Reading quadtree')
    quadtree_poi_filename = path_quadtree + '%s_personal_osm_poi_lv17.json.gz' % area
    fout = gzip.GzipFile(quadtree_poi_filename, 'r')
    quadtree = json.loads(fout.readline())
    fout.close()

    print(datetime.datetime.now(), 'Reading quadtree features')
    quadtree_features_filename = path_quadtree + '%s_quadtree_features.json.gz' % area
    fout = gzip.GzipFile(quadtree_features_filename, 'r')
    quadtrees_features_str = json.loads(fout.readline())
    quadtrees_features = {int(k): v for k, v in quadtrees_features_str.items()}
    fout.close()

    processed_users = set()
    if overwrite:
        for index in range(0, 7):
            output_filename = path_traintest + '%s_%s_traintest_%s.json.gz' % (area, type_user, index)
            if os.path.exists(output_filename):
                os.remove(output_filename)
    else:
        processed_users = set()
        for index in range(0, 3):
            output_filename = path_traintest + '%s_%s_traintest_%s.json.gz' % (area, type_user, index)
            if os.path.isfile(output_filename):
                fout = gzip.GzipFile(output_filename, 'r')
                for row in fout:
                    customer_obj = json.loads(row)
                    processed_users.add(customer_obj['uid'])
                fout.close()

    print(datetime.datetime.now(), 'Generating month boundaries')
    months = pd.date_range(start=datetime_from, end=datetime_to, freq='MS')
    boundaries = [[lm, um] for lm, um in zip(months[:-window], months[window:])]
    training_months = list()
    test_months = list()
    for i in range(len(boundaries)-1):
        training_months.append(boundaries[i])
        test_months.append(boundaries[i+1])

    index = 0
    tr_data_map = dict()
    ts_data_map = dict()
    for tr_months, ts_months in zip(training_months, test_months):
        tr_data_map[tuple(tr_months)] = index
        ts_data_map[tuple(ts_months)] = index
        index += 1
        # print(tr_months, ts_months)

    print(datetime.datetime.now(), 'Initializing quadtree features')
    tr_quadtree_features = dict()
    for m in quadtrees_features:
        for lu, index in tr_data_map.items():
            if lu[0].month <= m < lu[1].month:
                if index not in tr_quadtree_features:
                    tr_quadtree_features[index] = dict()
                for path in quadtrees_features[m]:
                    if path not in tr_quadtree_features[index]:
                        tr_quadtree_features[index][path] = {
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
                    for k, v in quadtrees_features[m][path].items():
                        tr_quadtree_features[index][path][k] += v

    ts_quadtree_features = dict()
    for m in quadtrees_features:
        for lu, index in tr_data_map.items():
            if lu[0].month <= m < lu[1].month:
                if index not in ts_quadtree_features:
                    ts_quadtree_features[index] = dict()
                for path in quadtrees_features[m]:
                    if path not in ts_quadtree_features[index]:
                        ts_quadtree_features[index][path] = {
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
                    for k, v in quadtrees_features[m][path].items():
                        ts_quadtree_features[index][path][k] += v

    print(datetime.datetime.now(), 'Connecting to database')

    mongo_connector = TaK_Mongo_Connector(cfg.mongodb["host"], cfg.mongodb["port"], cfg.mongodb["db"],
                                          cfg.mongodb["user"], cfg.mongodb["password"])

    input_query = {"adaptive": cfg.traj_seg["adaptive"], "temporal_thr": cfg.traj_seg["time_treshold"],
                   "spatial_thr": cfg.traj_seg["space_treshold"], "max_speed": cfg.traj_seg["max_speed"],
                   "min_length": cfg.imn["min_length"], "min_duration": cfg.imn["min_duration"],
                   "events_crashes": True, "from_date": cfg.imn["from_date"], "to_date": cfg.imn["to_date"]}


    count = 0
    imn_filedata = gzip.GzipFile(path_imn + '%s_imn_%s.json.gz' % (area, type_user), 'r')
    # imn_filedata = gzip.GzipFile(path_imn + '%s_imn_%s_allcat.json.gz' % (area, type_user), 'r')
    print(datetime.datetime.now(), 'Calculating features and partitioning dataset')
    for row in imn_filedata:
        if len(row) <= 1:
            print('new file started ;-)')
            continue

        user_obj = json.loads(row)
        uid = user_obj['uid']
        input_query['user_id'] = uid
        count += 1
        if uid in processed_users:
            continue

        imh, events, crashes = mongo_connector.load_imh(cfg.mongodb["input_collection"], **input_query)

        trajectories = imh['trajectories']
        print(len(trajectories), len(events), len(crashes))

        tr_data = dict()
        ts_data = dict()

        # partitioning imn for train and test
        for imn_months in user_obj:
            if imn_months == 'uid':
                continue

            m0 = int(imn_months.split('-')[0])
            m1 = int(imn_months.split('-')[1])
            for lu, index in tr_data_map.items():
                # print("NOT in IF m0 m1 is ok %s,, %s" % (m0, m1))
                # print(lu[0], lu[0].month, lu[1].month, lu[0].month <= m0 < m1 < lu[1].month)
                if lu[0].month <= m0 < m1 < lu[1].month:
                    # print("m0 m1 is ok %s,, %s" %(m0,m1))
                    if index not in tr_data:
                        # print("not in trdata")
                        tr_data[index] = {'uid': uid, 'crash': False, 'trajectories': dict(),
                                          'imns': dict(), 'events': dict(),}
                        # print('train', index, lu)
                    tr_data[index]['imns'][imn_months] = user_obj[imn_months]

            for lu, index in ts_data_map.items():
                # print("TSSS : NOT in IF m0 m1 is ok %s,, %s" % (m0, m1))
                # print(lu[0], lu[0].month, lu[1].month, lu[0].month <= m0 < m1 < lu[1].month)
                if lu[0].month <= m0 < lu[1].month:
                    if index not in ts_data:
                        ts_data[index] = {'uid': uid, 'crash': False, 'trajectories': dict(),
                                          'imns': dict(), 'events': dict(),}
                        # print('test', index, lu)
                    ts_data[index]['imns'][imn_months] = user_obj[imn_months]

        # partitioning trajectories for train and test
        for tid, traj in trajectories.items():
            for lu, index in tr_data_map.items():
                if lu[0] <= pd.Timestamp(traj.start_time(), unit='s') < lu[1] and index in tr_data:
                    #print("I'm in IF start time 294")
                    tr_data[index]['trajectories'][tid] = traj
            for lu, index in ts_data_map.items():
                if lu[0] <= pd.Timestamp(traj.start_time(), unit='s') < lu[1] and index in ts_data:
                    #print("I'm in IF start time 298")
                    ts_data[index]['trajectories'][tid] = traj

        # partitioning events for train and test
        for eid, evnt in events.items():
            # print(evnt)
            for lu, index in tr_data_map.items():
                if lu[0] <= pd.Timestamp(evnt[0]['date'], unit='s') < lu[1] and index in tr_data:
                    #print("I'm in IF event 306")
                    tr_data[index]['events'][eid] = evnt[0]
            for lu, index in ts_data_map.items():
                if lu[0] <= pd.Timestamp(evnt[0]['date'], unit='s') < lu[1] and index in ts_data:
                    #print("I'm in IF event 310")
                    ts_data[index]['events'][eid] = evnt[0]

        # get has crash next month
        for lu, index in tr_data_map.items():
            if index not in tr_data:
                continue
            # query = """SELECT * FROM %s WHERE uid = '%s'
            #             AND date >= TO_TIMESTAMP('%s','YYYY-MM-DD HH24:MI:SS')
            #             AND date < TO_TIMESTAMP('%s','YYYY-MM-DD HH24:MI:SS')""" % (
            #     crash_table, uid, str(lu[1]), str(lu[1] + relativedelta(months=1)))
            # Check if there were crashes in the date interval:
            crashes_count = 0
            for k, v in crashes.items():
                crash_time = pd.Timestamp(v[0]['date'], unit='s')
                if lu[1] <= crash_time < lu[1] + relativedelta(months=1):
                    #print("I'm in IF crash 326")
                    crashes_count += 1
            has_crash_next_month = crashes_count > 0
            tr_data[index]['crash'] = has_crash_next_month
            # print('train', index, has_crash_next_month)

        for lu, index in ts_data_map.items():
            if index not in ts_data:
                continue
            # query = """SELECT * FROM %s WHERE uid = '%s'
            #             AND date >= TO_TIMESTAMP('%s','YYYY-MM-DD HH24:MI:SS')
            #             AND date < TO_TIMESTAMP('%s','YYYY-MM-DD HH24:MI:SS')""" % (
            #     crash_table, uid, str(lu[1]), str(lu[1] + relativedelta(months=1)))

            crashes_count = 0
            for k, v in crashes.items():
                crash_time = pd.Timestamp(v[0]['date'], unit='s')
                if lu[1] <= crash_time < lu[1] + relativedelta(months=1):
                    #print("I'm in IF crash 344")
                    crashes_count += 1
            has_crash_next_month = crashes_count > 0
            ts_data[index]['crash'] = has_crash_next_month
            # print('test', index, has_crash_next_month)

        # print("len %s , %s" % (len(tr_data), len(ts_data)))
        tr_features, ts_features = extract_features(uid, tr_data, ts_data, quadtree,
                                                    tr_quadtree_features, ts_quadtree_features)

        # print("len %s , %s" % (len(tr_features), len(ts_features)))
        for index in tr_features:
            if index in ts_features:
                output_filename = path_traintest + '%s_%s_traintest_%s.json.gz' % (area, type_user, index)
                store_obj = {'uid': uid, 'train': tr_features[index], 'test': ts_features[index]}
                # print("going to store!!")
                # print(output_filename)
                store_features(output_filename, store_obj)

        # if count == 100:
        #     break

    imn_filedata.close()


if __name__ == "__main__":
    main()