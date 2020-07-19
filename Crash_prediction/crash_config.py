# Configuration and default values to run the scripts.

# MongoDB parameters
mongodb = {
    "host": "localhost",  # MongoDB host
    "port": "27017",  # MongoDB port
    "user": "",  # MongoDB username
    "password": "",  # MongoDB password
    "db": "test",  # MongoDB database
    "input_collection": "dataset4"  # MongoDB input collection containing user history
}

# Segmentation parameters
traj_seg = {
    "adaptive": True,
    "max_speed": 0.07,
    "space_treshold": 0.05,
    "time_treshold": 1200
}

# IMN extraction parameters
imn = {
    "from_date": "2017-01-01T00:00:00.000",  # Start date/time of the period for which the IMN is going to be extracted
    "to_date": "2018-01-01T00:00:00.000",  # End date/time of the period for which the IMN is going to be extracted
    "min_traj_nbr": 300,  # minimum number of trajectories in the period to start building IMNs
    "min_length": 1.0,  # minimum duration of an extracted trajectory
    "min_duration": 60,  # Whether to overwrite the output file or add to it.
}

crash = {
    "window": 4
}

# Parameters to define where to store the imns for training, trained classifiers and results
store_files_path = 'data/'  # The base path of the output
store_files = {
    "path_imn": store_files_path + 'imn/',  # imn_extract.py will store the results in this folder
    "path_dataset": store_files_path + 'dataset/',  # information about the users for training and feature names
    "path_traintest": store_files_path + 'traintest/',  # path for storing train/test data
    "path_quadtree": store_files_path + 'quadtree/',  # path containing the quadtrees and their features
    "path_eval": store_files_path + 'evaluation/',  # path to store the result of training and classifiers
    "path_visual": store_files_path + 'vis/'  # path to store the visualizations
}

# The file containing the id of users for which IMNs will be extracted
users_file = store_files["path_dataset"] + "users.txt"
