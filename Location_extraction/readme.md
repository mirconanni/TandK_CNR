![Track and Know project](../EV_simulation/fig/tak.jpg "Track and Know project")
H2020 - Grant Agreement n. 780754

# Location Extractor
An open-source implementation of the algorithm for adaptive trajectory segmentation introduced in [[1]](#1) and extracting the locations based on the mobility history of a user. The mobility history includes poisition information of the user during a specific period of time.

## Input Parameters
Adaptive trajectory segmentation receives the following required parameters:
* points: list of points, each point as a list of longitude, latitude and unix timestamp.
* uid: the id of the user, to be added to the resulting trajectory objects.

and some optional parameters, among which the important ones are:
* max_speed: Used for trajectory segmentation
* space_treshold: If the spatial distance between two consecutive points **A** and **B** in user's positions sequence is higher than this threshold, the point A is considered as the last point of the previous trajectory and point B is considered as the starting point of the new trajectory.
* time_treshold: If the temporal difference between two consecutive points **A** and **B** in user's positions sequence is higher than this threshold, the point A is considered as the last point of the previous trajectory and point B is considered as the starting point of the new trajectory.

The location extractor receives the following input parameters:
* kmin: minimum *k* for XMeans clustering
* kmax: maximum *k* for XMeans clustering
* xmeans_df: distance function for XMeans clustering
* singlelinkage_df: distance function for computing pairwise distances between cluster centers returned by XMeans 
* is_outlier: a statistical test function for identifying outliers
* min_dist: used for getting the cut distances

## Example Usage on Geolife Dataset
In this example, the data of one user (user id 064) in the Geolife trajectory dataset [[2]](#2) is used to perform adaptive trajectory segmentation and identify locations. The code below is presented in script *geolife_example.py*.

```python
import glob
import pandas as pd

import Location_extraction.Helpers.trajectory_segmenter as trajectory_segmenter
from Location_extraction.Helpers.tosca import *

mypath = "Geolife Trajectories 1.3/Data/064/Trajectory/"
uid = 64
all_points = []
for f in sorted(glob.glob(mypath + "*.plt")):
    points = pd.read_csv(f, skiprows=6, header=None, parse_dates=[[5, 6]])
    points['ts'] = points['5_6'].astype(int) // 10**9
    all_points.extend(points[[1,0,'ts']].values.tolist())
```

Adaptive trajectory segmenter recieves the whole position history of the user as input and returns a dictionary containing the segmented trajectories. As an optional post-processing step, trajectories can be filtered to remove those with very short spatial length and temporal duration.

```python
# Adaptive segmentation of trajectories
traj_list = trajectory_segmenter.segment_trajectories_user_adaptive(all_points, int(uid), temporal_thr=1200, spatial_thr=50, max_speed=0.07)

# Removing extremely short trajectories
traj_list = {k: tr for k, tr in traj_list.items() if tr.duration() > 60 or tr.length() > 1.0}
```

Then, only the spatial information of start and end points of trajectories are fed into the **fit** method of Tosca location extractor which is instantiated with different parameters such as the functions for calculating distances and checking for outliers.

```python
# Retrieving the start and end points of trajectories for identifying locations
spatial_start_end_points = []
for traj in traj_list.values():
    spatial_start_end_points.append([traj.start_point()[0], traj.start_point()[1]])  # lon, lat
    spatial_start_end_points.append([traj.end_point()[0], traj.end_point()[1]])  # lon, lat

# Number of runs for location extractor
nrun = 5

centers_min, centers_max = get_min_max(spatial_start_end_points)

cluster_res = dict()
cuts = dict()
for runid in range(0, nrun):
    try:
        tosca = Tosca(kmin=centers_min, kmax=centers_max, xmeans_df=spherical_distances,
                      singlelinkage_df=spherical_distance, is_outlier=thompson_test,
                      min_dist=50.0, verbose=False)
        tosca.fit(np.asarray(spatial_start_end_points))
        cluster_res[tosca.k_] = tosca.cluster_centers_
        cuts[tosca.k_] = tosca.cut_dist_
    except ValueError:
        pass

if len(cluster_res) == 0:
    return None
```

At the end, the location labels can be generated based on the distances of points to cluster centers.

```python
index = np.min(list(cluster_res.keys()))
centers = cluster_res[index]
loc_tosca_cut = cuts[index]

# calculate distances between points and medoids
distances = spherical_distances(np.asarray(spatial_start_end_points), np.asarray(centers))

# calculates labels according to minimum distance
labels = np.argmin(distances, axis=1)

# build clusters according to labels and assign point to point identifier
location_points = defaultdict(list)
location_prototype = dict()
for pid, lid in enumerate(labels):
    location_points[lid].append(pid)
    location_prototype[lid] = list(centers[lid])

print(location_points)  # Contains the mapping from locations to points
print(location_prototype)  # Contains the latitude and longitude of the location centroids
```

## Acknowledgement
This work is partially supported by the E.C. H2020 programme under the funding scheme Track & Know, G.A. 780754, [Track&Know](https://trackandknowproject.eu)

## References
The software is one of the results of research activities that led to the following publications:

* [<div id="1">[1] Agnese Bonavita, Riccardo Guidotti, Mirco Nanni.
Self-Adapting Trajectory Segmentation.
In EDBT/ICDT Workshop on Big Mobility Data Analytics (BMDA 2020), CEUR, vol 2578, 2020.</div>](http://ceur-ws.org/Vol-2578/BMDA3.pdf)

* <div id="2">[2] Geolife Trajectory Dataset (https://www.microsoft.com/en-us/download/details.aspx?id=52367)
