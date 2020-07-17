# Individual Mobility Network Extractor
An open-source implementation of the algorithm for extracting Individual Mobility Network (IMN) based on the mobility history of a user. The mobility history includes poisition information, and optional events and crashes of the user during a specific period of time.

## Input Parameters
The main application can be executed using the following command:
```bash
python3 imn_extractor.py host port db input_col output_col users_filename adaptive max_speed space_treshold time_treshold from_date to_date min_traj_nbr min_length min_duration
```
The input parameters are:
* host: The hostname of the MongoDB database
* port: The port number of the MongoDB database
* db: The name of the MongoDB database to read input data
* input_col: The collection in the MongoDB database that contains the input data (positions, events and crashes)
* output_col: The name of the collection to store the IMN
* users_filename: The filename that contains the id of the users for which IMN should be extracted
* adaptive: If True, the adpative approach introduced in [[1]](#1) will be used for trajectory segmentation. Otherwise, the normal trajectory segmentation based on max_speed, space_treshold, and time_treshold will be used.
* max_speed: Used for trajectory segmentation
* space_treshold: If the spatial distance between two consecutive points **A** and **B** in user's positions sequence is higher than this threshold, the point A is considered as the last point of the previous trajectory and point B is considered as the starting point of the new trajectory.
* time_treshold: If the temporal difference between two consecutive points **A** and **B** in user's positions sequence is higher than this threshold, the point A is considered as the last point of the previous trajectory and point B is considered as the starting point of the new trajectory.
* from_date: Start date and time of the period for which the IMN is going to be extracted
* to_date: End date and time of the period for which the IMN is going to be extracted
* min_traj_nbr: minimum number of trajectories in the period to start building IMNs
* min_length: minimum spatial length of an extracted trajectory to be considered in the IMN extraction process
* min_duration: minimum temporal duration of an extracted trajectory to be considered in the IMN extraction process

## Input Data
Each record in the input mobility history collection should contain at least these fields:
* VEHICLE_ID: that is the identifier of the user
* RECORD_TYPE: that identifies the type of the record: 'P' for position fix, 'X' for crash and harsh event types: 'A' = acceler, 'B' = braking, 'C' = cornering, 'Q' = quick lateral movement
* location: object which contains two double fileds of lat and lon.
* TIMESTAMP: The date/time of the record

It should be highlighted that the IMN extractor can produce IMNs without the event and crash information. If the events information are available, the event and crash records, in adition to the above mandatory fields, can also contain:
{SPEED, MAX_ACCELERATION, AVG_ACCELERATION, EVENT_ANGLE, LOCATION_TYPE, DURATION, HEADING}

## Example Usage
```bash
python3 imn_extractor.py localhost 27017 testdb dataset4 user_imns users.txt False 0.07 0.05 1200 2017-04-01T00:00:00.000 2017-06-01T00:00:00.000 100 1.0 60
```

## Acknowledgement
This work is partially supported by the E.C. H2020 programme under the funding scheme Track & Know, G.A. 780754, [Track&Know](https://trackandknowproject.eu)

## References
The software is one of the results of research activities that led to the following publications:

* [<div id="1">[1] Agnese Bonavita, Riccardo Guidotti, Mirco Nanni.
Self-Adapting Trajectory Segmentation.
In EDBT/ICDT Workshop on Big Mobility Data Analytics (BMDA 2020), CEUR, vol 2578, 2020.</div>](http://ceur-ws.org/Vol-2578/BMDA3.pdf)

* <div id="2">[2] Riccardo Guidotti and Mirco Nanni. Crash Prediction and Risk Assessment with Individual Mobility Networks. To appear In IEEE MDM Conference 2020</div>
