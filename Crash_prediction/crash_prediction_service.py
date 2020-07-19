import sys
sys.path.append('..')
import json
import networkx as nx
import pandas as pd
import gzip
import pickle
import numpy as np
import sklearn; print("Scikit-Learn", sklearn.__version__)
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from networkx.readwrite import json_graph

from Crash_prediction.Helpers.TaK_Mongo_Connector import TaK_Mongo_Connector
from Crash_prediction.individual_mobility_network import build_imn
from Crash_prediction.Helpers.trajectory import Trajectory
from Crash_prediction.feature_extractor import extract_features, extract_features_data
from Crash_prediction.visualization import visualize_points, visualize_trajectories, visualize_stops
from Crash_prediction.visualization import visualize_locations, visualize_imn, visualize_features, visualize_crash_risk

import Crash_prediction.crash_config as cfg

import datetime

periods_map = {
    'apr': 0,
    'may': 1,
    'jun': 2
}

# periods_map = {
#     'jun': 0,
#     'jul': 1,
#     'aug': 2,
#     'sep': 3,
#     'oct': 4,
#     'nov': 5,
#     'dec': 6,
# }

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

def init(area, period, window, datetime_from, datetime_to, feature_type, clf_type, verbose=True):

    path_dataset = cfg.store_files["path_dataset"]
    path_quadtree = cfg.store_files["path_quadtree"]
    path_eval = cfg.store_files["path_eval"]

    if verbose:
        print(datetime.datetime.now(), 'Partitioning periods ...', end='')
    months = pd.date_range(start=datetime_from, end=datetime_to, freq='MS')
    print(months)
    boundaries = [(lm, um) for lm, um in zip(months[:-window], months[window:])]
    print(boundaries)
    test_months = list()
    for i in range(len(boundaries) - 1):
        test_months.append(boundaries[i + 1])
    print(test_months)
    index = 0
    ts_data_map = dict()
    ts_data_map_rev = dict()
    for ts_months in zip(test_months):
        ts_data_map[ts_months] = index
        ts_data_map_rev[index] = ts_months
        index += 1

    if verbose:
        print(' done.')

    if verbose:
        print(datetime.datetime.now(), 'Reading trained crash classifier ...', end='')
    sel_index = periods_map[period]
    clf = pickle.load(open(path_eval + 'crash_prediction_%s_%s_%s_%s_f1_0.pickle' % (
        area, sel_index, feature_type, clf_type), 'rb'))
    if verbose:
        print(' done.')

    if verbose:
        print(datetime.datetime.now(), 'Reading quadtree ...', end='')
    quadtree_poi_filename = path_quadtree + '%s_personal_osm_poi_lv17.json.gz' % area
    fout = gzip.GzipFile(quadtree_poi_filename, 'r')
    quadtree = json.loads(fout.readline())
    fout.close()

    if verbose:
        print(' done.')

    if verbose:
        print(datetime.datetime.now(), 'Reading quadtree features ...', end='')
    quadtree_features_filename = path_quadtree + '%s_quadtree_features.json.gz' % area
    fout = gzip.GzipFile(quadtree_features_filename, 'r')
    quadtrees_features_str = json.loads(fout.readline())
    quadtrees_features = {int(k): v for k, v in quadtrees_features_str.items()}
    fout.close()

    # quadtrees_features = None
    if verbose:
        print(' done.')

    if verbose:
        print(datetime.datetime.now(), 'Managing features\' names ...', end='')
    features_names = json.load(open(path_dataset + 'features_names.json', 'r'))
    features_map = {'t': 'traj', 'e': 'evnt', 'i': 'imn', 'c': 'col'}

    features = list()
    for ft in feature_type:
        if ft in features_map:
            features.extend(features_names[features_map[ft]])
    if verbose:
        print(' done.')

    res = {
        'ts_data_map': ts_data_map,
        'ts_data_map_rev': ts_data_map_rev,
        'clf': clf,
        'quadtree': quadtree,
        'quadtrees_features': quadtrees_features,
        'features': features,
    }

    return res

def prepare_data4feature_extraction(uid, trajectories, events, imn_list, data_map, sel_index, verbose=True):

    if verbose:
        print(datetime.datetime.now(), 'Preparing data for feature extraction ...', end='')

    data = dict()
    # partitioning imn for train and test
    for imn_months in imn_list:
        if imn_months == 'uid':
            continue

        m0 = int(imn_months.split('-')[0])
        m1 = int(imn_months.split('-')[1])
        for lut, index in data_map.items():
            if index != sel_index:
                continue
            lu = lut[0]
            if lu[0].month <= m0 < m1 < lu[1].month:
                if index not in data:
                    data[index] = {'uid': uid, 'crash': False, 'trajectories': dict(),
                                   'imns': dict(), 'events': dict(), }
                data[index]['imns'][imn_months] = imn_list[imn_months]

    # partitioning trajectories for train and test
    for tid, traj in trajectories.items():
        for lut, index in data_map.items():
            if index != sel_index:
                continue
            lu = lut[0]
            if lu[0] <= pd.Timestamp(traj.start_time(), unit='s') < lu[1] and index in data:
                data[index]['trajectories'][tid] = traj

    # partitioning events for train and test
    for eid, evnt in events.items():
        for lut, index in data_map.items():
            if index != sel_index:
                continue
            lu = lut[0]
            if lu[0] <= pd.Timestamp(evnt[0]['date'], unit='s') < lu[1] and index in data:
                data[index]['events'][eid] = evnt[0]

    if verbose:
        print(' done.')

    return data


def extract_features(uid, data, quadtree, quadtree_features, verbose=True):
    if verbose:
        print(datetime.datetime.now(), 'Extracting features ...', end='')
    features = extract_features_data(uid, data, quadtree, quadtree_features)
    if verbose:
        print(' done.')
    return features


def prepare_features4classification(uid, area, period, sel_index, user_features, features, visual):
    path_traintest = cfg.store_files["path_traintest"]
    mms = MinMaxScaler()
    df_train = pd.read_csv(path_traintest + '%s_train_%s.csv.gz' % (area, sel_index))
    # print(df_train.shape)
    # print(df_train.columns)
    df_train.set_index('uid', inplace=True)
    df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_train.fillna(0, inplace=True)
    df_train = df_train.reset_index().drop_duplicates(subset='uid', keep='first').set_index('uid')
    X_train = df_train[features].values
    mms.fit(X_train)

    for f in features:
        if f not in user_features:
            user_features[f] = 0.0

    df = pd.DataFrame(data=user_features, index=[0])
    df.set_index('uid', inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    # print(df.shape)
    # print(df.columns)
    X = df[features].values
    X = mms.transform(X)

    # if visual:
    #     features_names = json.load(open(cfg.store_files["path_dataset"] + 'features_names.json', 'r'))
    #     features_filename = '%s_%s_%s_features.png' % (uid, area, period)
    #     visualize_features(cfg.store_files["path_visual"] + features_filename, user_features, df_train, features_names)

    return X

def main():
    uid = sys.argv[1]
    area = sys.argv[2]
    period = sys.argv[3]  # jun, jul, aug, sep, oct, nov, dec  See periods_map at line 28
    feature_type = 'teic'
    clf_type = 'RF'
    verbose = True
    visual = False

    # Connecting to the MongoDB
    mongo_connector = TaK_Mongo_Connector(cfg.mongodb["host"], cfg.mongodb["port"], cfg.mongodb["db"],
                                          cfg.mongodb["user"], cfg.mongodb["password"])

    print(datetime.datetime.now(), "\nStarting prediction process")

    window = cfg.crash["window"]
    datetime_from = datetime.datetime.strptime(cfg.imn["from_date"], '%Y-%m-%dT%H:%M:%S.%f')
    datetime_to = datetime.datetime.strptime(cfg.imn["to_date"], '%Y-%m-%dT%H:%M:%S.%f')

    res = init(area, period, window, datetime_from, datetime_to, feature_type, clf_type, verbose=verbose)

    data_map = res['ts_data_map']
    data_map_rev = res['ts_data_map_rev']
    clf = res['clf']
    quadtree = res['quadtree']
    quadtree_features = res['quadtrees_features']
    features = res['features']
    print(data_map_rev)

    input_query = {"adaptive": cfg.traj_seg["adaptive"], "temporal_thr": cfg.traj_seg["time_treshold"],
                   "spatial_thr": cfg.traj_seg["space_treshold"], "max_speed": cfg.traj_seg["max_speed"],
                   "min_length": cfg.imn["min_length"], "min_duration": cfg.imn["min_duration"],
                   "events_crashes": True, "from_date": cfg.imn["from_date"], "to_date": cfg.imn["to_date"],
                   "user_id": uid}


    imh, events, crashes = mongo_connector.load_imh(cfg.mongodb["input_collection"], **input_query)

    if len(imh['trajectories']) < cfg.imn["min_traj_nbr"]:
        print("No trajectories %s" % uid)
        return -1

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

    imn_list = {'uid': uid}

    for stk in wimh_dict:
        wimh = wimh_dict[stk]
        wevents = wevents_dict[stk]
        if len(wimh['trajectories']) < cfg.imn["min_traj_nbr"] // 12:
            continue
        print("buidling IMN for %s" % stk)
        imn = build_imn(wimh, reg_loc=True, events=wevents, verbose=False)
        imn_list[stk] = imn

    if imn_list is None:
        return -1

    print(len(list(imn_list.keys())))
    data = prepare_data4feature_extraction(uid, imh['trajectories'], events, imn_list,
                                           data_map, periods_map[period], verbose=verbose)
    user_features = extract_features(uid, data, quadtree, quadtree_features, verbose=verbose)[periods_map[period]]

    if verbose:
        print(datetime.datetime.now(), 'Running crash prediction ...', end='')
    X = prepare_features4classification(uid, area, period, periods_map[period], user_features, features,
                                        visual=visual)
    Y = clf.predict(X)
    Y_proba = clf.predict_proba(X)

    if verbose:
        print(' done.')

    # crash_flag = Y[0]
    crash_proba = Y_proba[0][1] + 0.1
    crash_flag = int(np.round(crash_proba))

    if crash_flag:
        print(datetime.datetime.now(),
              'User %s is going to have a crash in %s in %s (crash probability %.2f)' % (
                  uid, period.capitalize(), area.capitalize(), crash_proba))
    else:
        print(datetime.datetime.now(),
              'User %s is not going to have a crash in %s in %s (crash probability %.2f)' % (
                  uid, period.capitalize(), area.capitalize(), crash_proba))

    # if visual:
    #     risk_filename = '%s_%s_%s_crash_risk.png' % (uid, area, period)
    #     visualize_crash_risk(cfg.store_files["path_visual"] + risk_filename, uid, area, period, crash_proba, path)


if __name__ == '__main__':
    main()

