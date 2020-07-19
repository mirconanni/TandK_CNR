from confluent_kafka import Consumer

#from . import trajectory


__author__ = 'Omid Isfahani Alamdari'
def load_individual_mobility_history(user_id, topic= 'vfi-batch-1-enriched', segment=True, adaptive=False, **kwargs):
    """
    Extracts the individual mobility history of the user. Can be adjusted to perform segmentation of trajectories.
    :param collection: (String) The name of the collection.
    :param user_id: (Int) The id of the user.
    :param segment: (Boolean) If set to True, the history will be segmented, otherwise, the trajectories will be
    returned as is in the collection.
    :param adaptive: (Boolean) If set to True, the user adaptive segmentation function is used, otherwise the normal
    one is used.
    :param kwargs: all the parameters related to segmentation.
    :return: The imh, individual mobility history of the user.
    """
    conf = {'bootstrap.servers': 'static.165.253.201.195.clients.your-server.de:9093,static.166.253.201.195.clients.your-server.de:9093,static.171.253.201.195.clients.your-server.de:9093',
            'group.id': 'cnr',
            'session.timeout.ms': 6000,
            'auto.offset.reset': 'beginning',
            'security.protocol': 'SSL',
            'ssl.ca.location': '/home/wp3user05/ssl/CARoot.pem',
            'ssl.key.location': '/home/wp3user05/ssl/key.pem',
            'ssl.certificate.location': '/home/wp3user05/ssl/certificate.pem',
            #'ssl.truststore.location': '/home/wp3user05/ssl/ca-cert',
            #'ssl.truststore.password': 'Tkc5*p',
            #'ssl.keystore.location': '/home/wp3user05/ssl/kafka.client.keystore.jks',
            #'ssl.keystore.password': 'Tkc5*p',
            'ssl.key.password': 'Tkc5*p'
            }
    c = Consumer(conf)
    # Subscribe to topic
    c.subscribe([topic])
    #TODO given interval!!!!
    #traj_list = []
    traj_list = dict()

    cursor = self.db[collection].find(
        {'id':str(user_id)}, {"_id": 0, "longitude": 1, "latitude": 1, "timestamp": 1}).sort(
        [("timestamp", 1)])
    alltraj = [[row['longitude'], row['latitude'], int(row['timestamp']/1000)] for row in cursor]


    if not adaptive:
        traj_list = trajectory_segmenter.segment_trajectories(alltraj, user_id, **kwargs)
    else:
        traj_list = trajectory_segmenter.segment_trajectories_user_adaptive(alltraj, user_id, **kwargs)

    imh = {'uid': user_id, 'trajectories': traj_list}
    return imh

def main():
    topic = 'vfi-batch-1-enriched'
    user_id = '16770_128690'
    load_individual_mobility_history(user_id, 'vfi-batch-1-enriched', segment=True, adaptive=False)


if __name__ == '__main__':
    main()