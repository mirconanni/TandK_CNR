![Track and Know project](../EV_simulation/fig/tak.jpg "Track and Know project")
H2020 - Grant Agreement n. 780754

# Crash Prediction
An open-source implementation of the algorithm for crash prediction based on the mobility history of a user. The mobility history includes poisition information, and optional events and crashes of the user during a specific period of time. It includes the code for extracting Individual Mobility Networks (IMNs) and Location Extraction, for ease of use.

## Code Workflow
The code includes different files with various functionalities as shows in the figure.
* In the first step, the imn_extractor.py should be run to extract IMNs for a set of users. These IMNs is used for training the crash prediction machine learning models. The script receives three parameters, a name for the area, and a name for the type of the users (among crash, nocrash) mostyly used for organizing the models, and a boolean parameter overwrite to indicate whether the the previously extracted IMNs file should be overwritten.
```bash
python3 imn_extractor.py area user_type overwrite
```

* To enable the computation of context features, the mobility data are aggregated on a quadtree data structure which can be generated using the quadtree_features_extractor.py script. The script receives two parameters, the name of the area and overwrite.
```bash
python3 quadtree_features_extractor.py area overwrite
```

* Next, given the generated IMNs and the quadtree features, the data should be partitioned into train and test subsets using the train_test_partitioner.py script. 
```bash
python3 train_test_partitioner.py area type_user overwrite
```

* Then, the script make_prediction_dataset.py prepares the data for training the models and crash predicition.
```bash
python3 make_prediction_dataset.py area
```

* In the next step, classifiers are generated using the script crash_prediction.py for the dataset and stored in Python pickle files.
```bash
python3 crash_prediction.py area overwrite
```

In the last step, the script crash_predicition_service.py can be used to compute the crash_risk_probability for a user within a speicific area and time period. The code also includes visualization components that draw the IMNs produced and plots of the features of a single user. The scripts receives three parameters, the id of the user uid, the area and period within which the crash risk probability of the user is requested.
```bash
python3 crash_predicition_service.py uid area period
```

![Crash Prediction Workflow](./fig/crash_prediction_workflow.png "Crash Prediction Workflow" | width=400)

## Configuration Parameters
There are some parameters used for connecting to MongoDB database, the desired period of data, trajectory segmentation, IMN extraction, and the location to store the results which can be modified in the file crash_config.py (the default values are set):
For MongoDB:
* host: The hostname of the MongoDB database
* port: The port number of the MongoDB database
* user: The username to connect to the MongoDB database
* password: The password to connect to the MongoDB database
* db: The name of the MongoDB database to read input data
* input_col: The collection in the MongoDB database that contains the input data (positions, events and crashes)

For trajectory segmentation:
* adaptive: If True, the adpative approach introduced in [[1]](#1) will be used for trajectory segmentation. Otherwise, the normal trajectory segmentation based on max_speed, space_treshold, and time_treshold will be used.
* max_speed: Used for trajectory segmentation
* space_treshold: If the spatial distance between two consecutive points **A** and **B** in user's positions sequence is higher than this threshold, the point A is considered as the last point of the previous trajectory and point B is considered as the starting point of the new trajectory.
* time_treshold: If the temporal difference between two consecutive points **A** and **B** in user's positions sequence is higher than this threshold, the point A is considered as the last point of the previous trajectory and point B is considered as the starting point of the new trajectory.

For IMN extraction:
* from_date: Start date and time of the period for which the IMN is going to be extracted
* to_date: End date and time of the period for which the IMN is going to be extracted
* min_traj_nbr: minimum number of trajectories in the period to start building IMNs
* min_length: minimum spatial length of an extracted trajectory to be considered in the IMN extraction process
* min_duration: minimum temporal duration of an extracted trajectory to be considered in the IMN extraction process

For crash prediction workflow:
* window: The number of months to be processed perior to the starting day of a month period for which the crash prediction is executed.

For storing the results, there are parameters to indicate the folder names.

and finally:
* users_filename: The filename that contains the id of the users for which IMN should be extracted for training.


## Input Data
Each record in the input mobility history collection should contain at least these fields:
* VEHICLE_ID: that is the identifier of the user
* RECORD_TYPE: that identifies the type of the record: 'P' for position fix, 'X' for crash and harsh event types: 'A' = acceler, 'B' = braking, 'C' = cornering, 'Q' = quick lateral movement
* location: object which contains two double fileds of lat and lon.
* TIMESTAMP: The date/time of the record

It should be highlighted that the IMN extractor can produce IMNs without the event and crash information. If the events information are available, the event and crash records, in adition to the above mandatory fields, can also contain:
{SPEED, MAX_ACCELERATION, AVG_ACCELERATION, EVENT_ANGLE, LOCATION_TYPE, DURATION, HEADING}

## Acknowledgement
This work is partially supported by the E.C. H2020 programme under the funding scheme Track & Know, G.A. 780754, [Track&Know](https://trackandknowproject.eu)

## References
The software is one of the results of research activities that led to the following publications:

* [<div id="1">[1] Agnese Bonavita, Riccardo Guidotti, Mirco Nanni.
Self-Adapting Trajectory Segmentation.
In EDBT/ICDT Workshop on Big Mobility Data Analytics (BMDA 2020), CEUR, vol 2578, 2020.</div>](http://ceur-ws.org/Vol-2578/BMDA3.pdf)

* <div id="2">[2] Riccardo Guidotti and Mirco Nanni. Crash Prediction and Risk Assessment with Individual Mobility Networks. To appear In IEEE MDM Conference 2020</div>
