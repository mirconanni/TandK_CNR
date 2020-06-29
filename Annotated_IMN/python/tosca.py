########## IMPORT LIBRARIES ########## 

# basic statistical operations
import numpy as np
# statistical functions module
from scipy import stats
# define a standard dictionary
from collections import defaultdict
# pairwise distance of points in array
from scipy.spatial.distance import pdist
# hierarchical cluster methods
from scipy.cluster.hierarchy import linkage, fcluster
# math functions
from math import sqrt

########## IMPORT MY SCRIPTS ########## 

from xmeans import XMeans
from distance_func import spherical_distance, euclidean_distance, euclidean_distances


########## FUNCTION DEFINITION ##########

# create a dict from cluster labels to the points in that cluster
def points_labels_to_clusters(points, labels):
    clusters = defaultdict(list)
    for i in range(0, len(points)):
        clusters[labels[i]].append(points[i])
    return clusters

# given a dict of clusters returns the list of points and labels
def clusters_to_points_labels(clusters):
    points = list()
    labels = list()
    for c, pl in clusters.items():
        for p in pl:
            points.append(p)
            labels.append(c)
    return points, labels

# given a set of points and a distance function extract the medoid 
def compute_medoids(points, dist):
    center = None
    min_sse = float('infinity')
    # for each point compute the distance to all other points
    for p1 in points:
        sse = 0
        for p2 in points:
            d = dist(p1, p2)
            sse += d
        # if it's the smallest one then the point is the medoid
        if sse < min_sse:
            min_sse = sse
            center = p1
    return center

# given the list of points returns the list of medoids
def get_clusters_medoids(clusters, dist):
    centers = list()
    # for each cluster take the list of points
    for _, pl in clusters.items():
        # append to the list the medoid of that cluster
        centers.append(compute_medoids(pl, dist))
    return centers

# one possible cut criteria, is an outlier detection method to find
# the difference that does not respect the distribution is selected
def thompson_test(data, point, alpha=0.05):
    sd = np.std(data)
    mean = np.mean(data)
    delta = abs(point - mean)
    # t_alpha_halfs is the critical Student's t value based on alpha
    t_a2 = stats.t.ppf(1-(alpha/2.), len(data)-2)
    # compute a rejection region based on the mean and the std of the data
    tau = (t_a2 * (len(data)-1)) / (sqrt(len(data)) * sqrt(len(data)-2 + t_a2**2))

    return delta > tau * 2*sd

# another possible cut criteria (not used)
# defines an interval out of which a value is considered an outlier
def interquartile_range_test(data, point):
    median = np.percentile(data, 50)
    data_low = []
    data_high = []
    for d in data:
        if d <= median:
            data_low.append(d)
        else:
            data_high.append(d)
    q1 = np.percentile(data_low, 50)
    q3 = np.percentile(data_high, 50)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    higher_bound = q3 + 1.5 * iqr
    return point < lower_bound or point > higher_bound

# returns the first distance that is signicantly dissimilar from the previous one
def get_cut_distance(data_link, is_outlier, min_k=float('infinity'), min_dist=0, min_size=3):
    diff_list = list()
    # for each row in the linkage matrix
    for i in range(1, len(data_link)):
        # get the current distance and the previous one
        dist_i_1 = data_link[i-1][2]
        dist_i = data_link[i][2]
        # compute the difference in the distance
        diff_dist = abs(dist_i - dist_i_1)

        # if no difference passes the test then no cut is possible
        # the dendogram is cut at level zero and no aggregation is needed
        if i > min_k:
            return np.mean([data_link[0][2], data_link[1][2]])

        # min_size is the minimum number of observations required to do the statistical test
        # if thompson test is passed then cut here
        if len(diff_list) > min_size and dist_i >= min_dist and is_outlier(np.asarray(diff_list), diff_dist):
            return dist_i_1

        # save the current difference in the distance
        diff_list.append(diff_dist)

        # if there's not enough observations but the distance is already too big then cut here
        if i <= min_size and dist_i >= min_dist:
            return dist_i_1

    return 0.0


########## MAIN CLASS ##########

class Tosca:

    # initialize the fields of the class
    def __init__(self, kmin=2, kmax=10, xmeans_df=euclidean_distances, singlelinkage_df=euclidean_distance,
                 is_outlier=thompson_test, min_dist=0, verbose=0):
        self.kmin = kmin
        self.kmax = kmax
        self.xmeans_df = xmeans_df
        self.singlelinkage_df = singlelinkage_df
        self.is_outlier = is_outlier
        self.min_dist = min_dist
        self.verbose = verbose

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.k_ = -1
        self.data_ = None

    # execute tosca on the dataset
    def fit(self, data):
        # initialize xmeans 
        xmeans = XMeans(kmin=self.kmin, kmax=self.kmax, distance_function=self.xmeans_df, verbose=self.verbose)
        # fit xmeans on the dataset
        xmeans.fit(data)
        # store the results: labels, clusters and centers
        xmeans_labels = xmeans.labels_
        # create a dict from cluster label to the points in it
        xmeans_clusters = points_labels_to_clusters(data, xmeans_labels)
        # compute the medoids of the clusters
        xmeans_centers = get_clusters_medoids(xmeans_clusters, euclidean_distance)

        # create a dictionary from points to clusters label
        inverse_xmeans_clusters = dict()
        for cluster_id, points_cluster in xmeans_clusters.items():
            for p in points_cluster:
                inverse_xmeans_clusters[(p[0], p[1])] = cluster_id

        # compute the pairwise distances between the centers of xmeans
        data_dist = pdist(xmeans_centers, metric=self.singlelinkage_df)
        # performs the single linkage
        data_link = linkage(data_dist, method='single', metric=self.singlelinkage_df) 

        # compute the height at which to cut the dendogram
        cut_dist = get_cut_distance(data_link, is_outlier=self.is_outlier, min_k=self.kmin, min_dist=self.min_dist)
        self.cut_dist_ = cut_dist
        # forms flat clusters from the hierarchical clustering defined by the linkage matrix
        singlelinkage_labels = fcluster(data_link, cut_dist, 'distance')
        # create a dict from cluster label to the points in it
        singlelinkage_clusters = points_labels_to_clusters(xmeans_centers, singlelinkage_labels)

        # aggregate clusters according to the cut distance
        tosca_clusters = defaultdict(list)
        for sl_cid, sl_pl in singlelinkage_clusters.items():
            # for each point in the cluster
            for p in sl_pl:
                # takes the cluster label given from x means
                xm_cid = inverse_xmeans_clusters[(p[0], p[1])]
                # merges the clusters of xmeans
                tosca_clusters[sl_cid].extend(xmeans_clusters[xm_cid])
        # compute the medoids of the new clusters
        tosca_centers = get_clusters_medoids(tosca_clusters, euclidean_distance)
        # get the list of points and cluster labels
        data, tosca_labels = clusters_to_points_labels(tosca_clusters)
        # final value of k after cluster aggregation
        tosca_k = len(tosca_centers)

        self.cluster_centers_ = tosca_centers
        self.labels_ = tosca_labels
        self.k_ = tosca_k
        self.data_ = data

        return self