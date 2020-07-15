import os
import sys

import json
import gzip
import datetime
import numpy as np
import pandas as pd
import networkx as nx


from collections import defaultdict
from networkx.readwrite import json_graph
from itertools import islice

import trajectory
import database_io
import individual_mobility_network


__author__ = 'Riccardo Guidotti'

    
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
    elif isinstance(o, trajectory.Trajectory):
        return o.to_json()
    elif isinstance(o, nx.DiGraph):
        return json_graph.node_link_data(o, {'link': 'edges', 'source': 'from', 'target': 'to'})
    else:
        return o.__str__()


    
def start_time_map(t):
    m = t.month
    if m == 1:
        return '01-02', None
    elif m == 12:
        return '11-12', None
    else:
        return '%02d-%02d' % (m-1, m), '%02d-%02d' % (m, m+1)


def imn_extract(filename, path, type_user, traj_table, evnt_table,
                min_traj_nbr, min_length, min_duration, area, overwrite=False):  
    output_filename = path + filename

    con = database_io.get_connection()
    cur = con.cursor()

    #users_list = [100225,101127,100742,100747,100690,100578,1003,100191,100192,100193,321463]
    users_list = [100619,100554]
    users_list = sorted(users_list)
    nbr_users = len(users_list)

    print("user ids before checking :")
    print(nbr_users, len(users_list))
    
    if os.path.isfile(output_filename) and not overwrite:
        processed_users = list()
        fout = gzip.GzipFile(output_filename, 'r')
        for row in fout:
            customer_obj = json.loads(row)
            processed_users.append(customer_obj['uid'])
        fout.close()
        users_list = [uid for uid in users_list if uid not in processed_users]
    print("user ids after checking :")
    print(nbr_users, len(users_list))

    for i, uid in enumerate(users_list):
        if i % 1 == 0:
            print(datetime.datetime.now(), '%s %s %s [%s/%s] - %.2f' % (
                traj_table, area, type_user, i,  nbr_users, i / nbr_users * 100.0))
            
        imh = database_io.load_individual_mobility_history(cur, uid, traj_table, min_length, min_duration)
        events = database_io.load_individual_event_history(cur, uid, evnt_table) if evnt_table is not None else None
        
        #imh['trajectories']=dict(list(islice(imh['trajectories'].items(), 200)))
        if len(imh['trajectories']) < min_traj_nbr:
            print('len trajectories]) < min_traj_nbr', len(imh['trajectories']), min_traj_nbr)
            continue
        
        main_imh=imh['trajectories']
        jan_feb_tid=[]
        march_april_id=[]

        for tid, t in imh['trajectories'].items():
            start_time=str(t.start_time())
            if ('2017-01' in start_time) or ('2017-02' in start_time) :
                jan_feb_tid.append(tid)
                
            if ('2017-03' in start_time) or ('2017-04' in start_time) :
                march_april_id.append(tid)
     
        imh['trajectories'] = {x:imh['trajectories'][x] for x in jan_feb_tid}
        imn1 = individual_mobility_network.build_imn(imh, reg_loc=True, events=events, verbose=False)
        period_imn1={"01-02":imn1}
 
        imh['trajectories'] = {x:main_imh[x] for x in march_april_id}
        imn2 = individual_mobility_network.build_imn(imh, reg_loc=True, events=events, verbose=False)
        period_imn2={"03-04":imn2}
        
        customer_obj = {'uid': uid}        
        period_imn1.update(period_imn2)         
        customer_obj.update(period_imn1)

        
        json_str = '%s\n' % json.dumps(clear_tuples4json(customer_obj), default=agenda_converter)
        json_bytes = json_str.encode('utf-8')
        with gzip.GzipFile(output_filename, 'a') as fout:
            fout.write(json_bytes)
    print("done")
    cur.close()
    con.close()

def imn_extract_all_year(filename, path, type_user, traj_table, evnt_table,
                min_traj_nbr, min_length, min_duration, area, overwrite=False):  
    output_filename = path + filename

    con = database_io.get_connection()
    cur = con.cursor()
    #users_list=find_user_list(cur,traj_table):
    #users_list = [100225,101127,100742,100747,100690,100578,1003,100191,100192,100193,318819,100619,100554,100498]
    #users_list = [100843,100836,100827,100795,100747,100717,100681,100669,101293,101194,101091]
    users_list=[7925]
    users_list = sorted(users_list)
    nbr_users = len(users_list)

    print("user ids before checking :")
    print(nbr_users, len(users_list))
    
    if os.path.isfile(output_filename) and not overwrite:
        processed_users = list()
        fout = gzip.GzipFile(output_filename, 'r')
        for row in fout:
            customer_obj = json.loads(row)
            processed_users.append(customer_obj['uid'])
        fout.close()
        users_list = [uid for uid in users_list if uid not in processed_users]
    print("user ids after checking :")
    print(nbr_users, len(users_list))

    for i, uid in enumerate(users_list):
        try:
            if i % 1 == 0:
                print(datetime.datetime.now(), '%s %s %s [%s/%s] - %.2f' % (
                    traj_table, area, type_user, i,  nbr_users, i / nbr_users * 100.0))

            imh = database_io.load_individual_mobility_history(cur, uid, traj_table, min_length, min_duration)
            events = database_io.load_individual_event_history(cur, uid, evnt_table) if evnt_table is not None else None

            #imh['trajectories']=dict(list(islice(imh['trajectories'].items(), 200)))
            if len(imh['trajectories']) < min_traj_nbr:
                print('len trajectories]) < min_traj_nbr', len(imh['trajectories']), min_traj_nbr)
                continue
            imn= individual_mobility_network.build_imn(imh, reg_loc=True, events=events, verbose=False)        
            customer_obj = {'uid': uid}               
            customer_obj.update(imn)

            json_str = '%s\n' % json.dumps(clear_tuples4json(customer_obj), default=agenda_converter)
            json_bytes = json_str.encode('utf-8')
            with gzip.GzipFile(output_filename, 'a') as fout:
                fout.write(json_bytes)
        except(TypeError):
            print("type error")
            continue
    print("done")
    cur.close()
    con.close()
    
    
def imn_extract_for_one_month(filename, path, type_user, traj_table, evnt_table,
                min_traj_nbr, min_length, min_duration, area, overwrite=False):  
    output_filename = path + filename
    con = database_io.get_connection()
    cur = con.cursor()
    users_list=find_user_list(cur,traj_table)
    nbr_users = len(users_list)
    print("user ids before checking :")
    print(nbr_users, len(users_list))
    if os.path.isfile(output_filename) and not overwrite:
        processed_users = list()
        fout = gzip.GzipFile(output_filename, 'r')
        for row in fout:
            customer_obj = json.loads(row)
            processed_users.append(customer_obj['uid'])
        fout.close()
        users_list = [uid for uid in users_list if uid not in processed_users]
    print("user ids after checking :")
    print(nbr_users, len(users_list))

    for i, uid in enumerate(users_list):
        try:
            if i % 1 == 0:
                print(datetime.datetime.now(), '%s %s %s [%s/%s] - %.2f' % (
                    traj_table, area, type_user, i,  nbr_users, i / nbr_users * 100.0))

            imh = database_io.load_individual_mobility_history(cur, uid, traj_table, min_length, min_duration)
            events = database_io.load_individual_event_history(cur, uid, evnt_table) if evnt_table is not None else None
            if len(imh['trajectories']) < min_traj_nbr:
                print('len trajectories]) < min_traj_nbr', len(imh['trajectories']), min_traj_nbr)
                continue

            main_imh=imh['trajectories']
            jan_tid=[]
            for tid, t in imh['trajectories'].items():
                start_time=str(t.start_time())
                if ('2017-01' in start_time):
                    jan_tid.append(tid)

            imh['trajectories'] = {x:imh['trajectories'][x] for x in jan_tid}
            imn1 = individual_mobility_network.build_imn(imh, reg_loc=True, events=events, verbose=False)
            period_imn1={"01":imn1}
            customer_obj = {'uid': uid}               
            customer_obj.update(period_imn1)
            json_str = '%s\n' % json.dumps(clear_tuples4json(customer_obj), default=agenda_converter)
            json_bytes = json_str.encode('utf-8')
            with gzip.GzipFile(output_filename, 'a') as fout:
                fout.write(json_bytes)
        except(TypeError):
            print("type error")
            continue
    print("done")
    cur.close()
    con.close()
    
    
def imn_extract_jan_feb(filename, path, type_user, traj_table, evnt_table,
                min_traj_nbr, min_length, min_duration, area, overwrite=False):  
    output_filename = path + filename
    con = database_io.get_connection()
    cur = con.cursor()

    users_list = [1003,10954]
    users_list = sorted(users_list)
    nbr_users = len(users_list)

    print("user ids before checking :")
    print(nbr_users, len(users_list))
    
    if os.path.isfile(output_filename) and not overwrite:
        processed_users = list()
        fout = gzip.GzipFile(output_filename, 'r')
        for row in fout:
            customer_obj = json.loads(row)
            processed_users.append(customer_obj['uid'])
        fout.close()
        users_list = [uid for uid in users_list if uid not in processed_users]
    print("user ids after checking :")
    print(nbr_users, len(users_list))

    for i, uid in enumerate(users_list):
        if i % 1 == 0:
            print(datetime.datetime.now(), '%s %s %s [%s/%s] - %.2f' % (
                traj_table, area, type_user, i,  nbr_users, i / nbr_users * 100.0))
            
        imh = database_io.load_individual_mobility_history(cur, uid, traj_table, min_length, min_duration)
        events = database_io.load_individual_event_history(cur, uid, evnt_table) if evnt_table is not None else None
        
        #imh['trajectories']=dict(list(islice(imh['trajectories'].items(), 200)))
        if len(imh['trajectories']) < min_traj_nbr:
            print('len trajectories]) < min_traj_nbr', len(imh['trajectories']), min_traj_nbr)
            continue
        
        main_imh=imh['trajectories']
        jan_feb_tid=[]

        for tid, t in imh['trajectories'].items():
            start_time=str(t.start_time())
            if ('2017-01' in start_time) or ('2017-02' in start_time) :
                jan_feb_tid.append(tid)
     
        imh['trajectories'] = {x:imh['trajectories'][x] for x in jan_feb_tid}
        imn1 = individual_mobility_network.build_imn(imh, reg_loc=True, events=events, verbose=False)
        period_imn1={"01-02":imn1}
        customer_obj = {'uid': uid}                
        customer_obj.update(period_imn1)

        
        json_str = '%s\n' % json.dumps(clear_tuples4json(customer_obj), default=agenda_converter)
        json_bytes = json_str.encode('utf-8')
        with gzip.GzipFile(output_filename, 'a') as fout:
            fout.write(json_bytes)
    print("done")
    cur.close()
    con.close()
    
def main():

    path = './'
    users_filename = 'imn10.json.gz'


    min_traj_nbr = 1
    min_length = 1.0
    min_duration = 60.0
    traj_table = 'tak.italy_traj'
    evnt_table = 'tak.italy_evnt'
    
    #imn_extract_all_year(users_filename, path, "C", traj_table, evnt_table,
                #min_traj_nbr, min_length, min_duration, "roma")
    
    imn_extract_jan_feb(users_filename, path, "C", traj_table, evnt_table,min_traj_nbr, min_length, min_duration, "toscany")


if __name__ == "__main__":
    main()
