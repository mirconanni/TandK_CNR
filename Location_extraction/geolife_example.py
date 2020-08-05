import glob
import pandas as pd

import Location_extraction.Helpers.trajectory_segmenter as trajectory_segmenter
from Location_extraction.Helpers.tosca import *

def main():
    mypath = "Geolife Trajectories 1.3/Data/064/Trajectory/"
    uid = 64
    all_points = []
    for f in sorted(glob.glob(mypath + "*.plt")):
        points = pd.read_csv(f, skiprows=6, header=None, parse_dates=[[5, 6]])
        points['ts'] = points['5_6'].astype(int) // 10**9
        all_points.extend(points[[1,0,'ts']].values.tolist())

    # Adaptive segmentation of trajectories
    traj_list = trajectory_segmenter.segment_trajectories_user_adaptive(all_points, int(uid), temporal_thr=1200, spatial_thr=50, max_speed=0.07)

    # Removing extremely small trajectories
    traj_list = {k: tr for k, tr in traj_list.items() if tr.duration() > 60 or tr.length() > 1.0}

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

    print(location_points)
    print(location_prototype)
    print(loc_tosca_cut)

if __name__ == "__main__":
    main()