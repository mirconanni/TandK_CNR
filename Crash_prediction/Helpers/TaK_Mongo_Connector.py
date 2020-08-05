from pymongo import MongoClient, ASCENDING, DESCENDING
import json
import Crash_prediction.Helpers.trajectory_segmenter as trajectory_segmenter
from Crash_prediction.Helpers.trajectory import Trajectory
from Crash_prediction.Helpers.mobility_distance_functions import spherical_distance
from datetime import datetime
from collections import defaultdict

__author__ = 'Omid Isfahani Alamdari'

class TaK_Mongo_Connector:
    """
    The connector to MongoDB
    This class provides methods to communicate with the MongoDB database of WP3.

    """

    def __init__(self, host='localhost', port='27017', db='trackAndKnow', user= "", passwd= ""):
        """
        The constructor for TaK_Mongo_Connector class.

        :param host: (String) The host address of the MongoDB instance.
        :param port: (String) The host port of the MongoDB instance.
        :param db: (String) The name of the database in MongoDB instance.
        """

        #uri = "mongodb://%s:%s@%s" % (config.get('MONGO_HOST'), config.get('MONGO_PORT'))
        if (user=="" or passwd == ""):
            uri = "mongodb://%s:%s" % (host, port)
        else:
            uri = "mongodb://%s:%s@%s:%s" % (user, passwd, host, port)
        #uri = "mongodb://%s:%s@%s:%s" % ("user_name", "password", host, port)

        self._conn = MongoClient(uri)
        print("server version:", self._conn.server_info()["version"])
        self._db = self._conn.get_database(db)

    def __enter__(self):
        """
        Used when the 'with' statement is used to access database.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Used when the 'with' statement is used to access database.
        :param exc_type:
        :param exc_val:
        :param exc_tb:
        :return:
        """
        #self.commit()  //TODO: If transaction processing needed, add this!
        self.connection.close()

    @property
    def connection(self):
        return self._conn

    @property
    def db(self):
        return self._db

    def load_imh(self, collection, user_id, from_date, to_date, adaptive=False,
                 min_length = 1.0, min_duration = 60.0, events_crashes=False, **kwargs):
        """
        Extracts the individual mobility history of the user. Can be adjusted to perform segmentation of trajectories.
        :param collection: (String) The name of the collection.
        :param user_id: (Int) The id of the user.
        :param segment: (Boolean) If set to True, the history will be segmented, otherwise, the trajectories will be
        returned as is in the collection.
        :param adaptive: (Boolean) If set to True, the user adaptive segmentation function is used, otherwise the normal
        one is used.
        :param events: (Boolean) if True, events in the same temporal range have been assigned to trajectories and
        returned.
        :param kwargs: all the parameters related to segmentation.
        :return: The imh, individual mobility history of the user.
        """
        dt_format = '%Y-%m-%dT%H:%M:%S.%f'
        datetime_from = datetime.strptime(from_date, dt_format)
        datetime_to = datetime.strptime(to_date, dt_format)

        traj_list = dict()
        cursor = self.db[collection].find(
            {'VEHICLE_ID': int(user_id), 'TIMESTAMP': {"$gte": datetime_from, "$lte": datetime_to}}, {"_id": 0}).sort(
            [("TIMESTAMP", 1)])

        alltraj = []
        allevents = []
        allcrashes = []
        for row in cursor:
            if row['RECORD_TYPE'] == 'P': # the record is a position fix
                alltraj.append([float(row['location']['lon']), float(row['location']['lat']), int((row['TIMESTAMP']).timestamp())])
            elif row['RECORD_TYPE'] != 'X':  # the record is any event other than crash
                event = {
                    'uid': str(row['VEHICLE_ID']),
                    'event_type': str(row['RECORD_TYPE']),
                    'speed': int(row['SPEED']),
                    'max_acc': int(row['MAX_ACCELERATION']),
                    'avg_acc': int(row['AVG_ACCELERATION']),
                    'angle': int(row['EVENT_ANGLE']),
                    'location_type': str(row['LOCATION_TYPE']),
                    'duration': int(row['DURATION']),
                    'date': int((row['TIMESTAMP']).timestamp()),
                    'lat': float(row['location']['lat']),
                    'lon': float(row['location']['lon'])
                }
                allevents.append(event)
            elif row['RECORD_TYPE'] == 'X':  # the record is a crash event
                crash = {
                    'uid': str(row['VEHICLE_ID']),
                    'speed': int(row['SPEED']),
                    'max_acc': int(row['MAX_ACCELERATION']),
                    'heading': int(row['HEADING']),
                    'date': int((row['TIMESTAMP']).timestamp()),
                    'lat': float(row['location']['lat']),
                    'lon': float(row['location']['lon'])
                }
                allcrashes.append(crash)

        # Trajectory Segmentation
        if not adaptive:
            traj_list = trajectory_segmenter.segment_trajectories(alltraj, int(user_id), **kwargs)
        else:
            traj_list = trajectory_segmenter.segment_trajectories_user_adaptive(alltraj, int(user_id), **kwargs)

        # Filter trajectories based on minlength and minDuration
        traj_list = {k: v for k, v in traj_list.items() if v.duration() > min_duration or v.length() > min_length}

        imh = {'uid': int(user_id), 'trajectories': traj_list}

        if events_crashes and bool(traj_list):  # check if there are trajectories
            events_history = defaultdict(list)
            event_id = 0
            events_index = 0
            for tid in sorted(imh['trajectories']):
                # Add a start event for this trajectory
                sp = imh['trajectories'][tid].start_point()
                event = {
                    'uid': str(user_id),
                    'tid': int(tid),
                    'eid': int(event_id),
                    'event_type': 'start',
                    'speed': -1,
                    'max_acc': -1,
                    'avg_acc': -1,
                    'angle': -1,
                    'location_type': -1,
                    'duration': -1,
                    'status': 0,
                    'date': imh['trajectories'][tid].start_time(),
                    'lat': sp[1],
                    'lon': sp[0]
                }
                events_history[tid].append(event)
                event_id += 1

                # Add other events during this trajectory
                while events_index < len(allevents) and \
                        allevents[events_index]['date'] < imh['trajectories'][tid].start_time():
                    events_index += 1
                while events_index != len(allevents) and allevents[events_index]['date'] <= imh['trajectories'][tid].end_time():
                    allevents[events_index].update({'tid': int(tid), 'eid': int(event_id)})
                    events_history[tid].append(allevents[events_index])
                    # add this event!
                    event_id += 1
                    events_index += 1

                # Add a stop event for this trajectory
                ep = imh['trajectories'][tid].end_point()
                event = {
                    'uid': str(user_id),
                    'tid': int(tid),
                    'eid': int(event_id),
                    'event_type': 'stop',
                    'speed': -1,
                    'max_acc': -1,
                    'avg_acc': -1,
                    'angle': -1,
                    'location_type': -1,
                    'duration': -1,
                    'status': 2,
                    'date': imh['trajectories'][tid].end_time(),  # Handle this!!
                    'lat': ep[1],
                    'lon': ep[0]
                }
                events_history[tid].append(event)
                event_id += 1

            # Adding Crashes
            crashes_history = defaultdict(list)
            crash_id = 0
            crashes_index = 0
            for tid in sorted(imh['trajectories']):
                # Add crashes during this trajectory
                while crashes_index < len(allcrashes) and \
                        allcrashes[crashes_index]['date'] < imh['trajectories'][tid].start_time():
                    crashes_index += 1
                while crashes_index != len(allcrashes) and allcrashes[crashes_index]['date'] <= imh['trajectories'][tid].end_time():
                    allcrashes[crashes_index].update({'tid': int(tid), 'cid': int(crash_id)})
                    crashes_history[tid].append(allcrashes[crashes_index])
                    # add this crash!
                    event_id += 1
                    crashes_index += 1

            return imh, events_history, crashes_history
        else:
            return imh, None, None

    def extract_users_list(self, collection):
        """
        Returns the list of users in the collection.
        :param collection: (String) The name of the collection.
        :return: The distinct list of the users (field 'id' of the documents) in the collection.
        """
        cursor = self.db[collection].distinct("VEHICLE_ID")
        return cursor

    def load_user_all_points(self, collection, user_id):
        """
        Returns alltraj (in the codes of Riccardo)
        :param collection: (String) The name of the collection.
        :param user_id: (Int) The id of the user.
        :return:
        """
        cursor = self.db[collection].find({'id':str(user_id)}, {"_id": 0, "longitude": 1, "latitude": 1, "timestamp": 1}).sort([("timestamp", 1)])
        return cursor

    def read_one_collection(self, collection):
        """
        Returns one document from the collection.
        :param collection: (String) The name of the collection.
        :return: A document in JSON format.
        """
        return self.db[collection].findOne()

    def user_points_count(self, collection, limit):
        """
        Returns the list of users and their total number of points, sorted by the descending order of count.
        :param collection: (String) The name of the collection.
        :param limit: (Int) The number of results to be returned.
        :return: A list of tuples (userid, count)
        """
        cursor = self.db[collection].aggregate([
            {
                '$group': {
                    '_id': "$vehicle",
                    'count': {'$sum': 1}
                }
            },
            {'$sort': {'count': -1}},
            {"$limit": limit}
        ])

        return [(row["_id"], row["count"]) for row in cursor]

    def insert_one(self, collection, doc_object):
        self.db[collection].insert_one(doc_object)

    def insert_many(self, collection, doc_list):
        self.db[collection].insert_many(doc_list)

    def create_index(self, collection, field, asc):
        if asc == True:
            self.db[collection].create_index([(field, ASCENDING)])
        else:
            self.db[collection].create_index([(field, DESCENDING)])

    def import_csv(self, filename, collection):
        """
        Imports each line of the CSV file and inserts the line as a document to the collection of MongoDB.
        :param filename: (String) The path to the CSV file.
        :param collection: (String) The name of the collection to be created in MongoDB database.
        :return: None
        """
        with open(filename) as f:
            for row in f:
                self.db[collection].insert_one(json.loads(row))

    def find_one_point(self, collection, user_id):
        cursor = self.db[collection].find_one({'VEHICLE_ID': int(user_id)})
        return cursor