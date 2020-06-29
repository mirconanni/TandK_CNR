########## IMPORT LIBRARIES ########## 

# necessary for math formula
import numpy as np
# compute trigonometric functions
from math import radians, degrees, sin, cos, asin, acos, sqrt


########## IMPORT MY SCRIPTS ########## 

# needed for distance functions
from distance_func import euclidean_distances
# compute the k means step in xmean
from kmeans import KMeans


########## FUNCTION DEFINITION ##########

# compute the cluster variance of all the clusters given as params
def _cluster_variance(num_points, clusters, centroids):
    s = 0
    denom = float(num_points - len(centroids))
    #for each combination cluster-centroid compute the distance from the points to that cluster
    for cluster, centroid in zip(clusters, centroids):
        distances = euclidean_distances(cluster, centroid)
        s += (distances*distances).sum()
    return s / denom if 0 != denom else 0

# compute the total log likelihood of all the clusters given as params
def _loglikelihood(num_points, num_dims, clusters, centroids):
    ll = 0
    # for each cluster
    for cluster in clusters:
        fRn = len(cluster)
        t1 = fRn * np.log(fRn)
        t2 = fRn * np.log(num_points)
        # if the variance returned is 0 take the smallest possible value > than 0
        variance = _cluster_variance(num_points, clusters, centroids) or np.nextafter(0, 1)
        t3 = ((fRn * num_dims) / 2.0) * np.log((2.0 * np.pi) * variance)
        t4 = (fRn - 1.0) / 2.0
        ll += t1 - t2 - t3 - t4
    return ll

# compute the bic value given the list of cluster points and the list of centroids
def bic(clusters, centroids):
    num_points = sum(len(cluster) for cluster in clusters)
    # compute the number of dimensions, in our case it's always 2: x and y
    num_dims = clusters[0][0].shape[0]
    # compute the loglikelihood of all the points
    log_likelihood = _loglikelihood(num_points, num_dims, clusters, centroids)
    # compute the number of free parameters, hence the weight of the penalty
    num_params = len(clusters) * (num_dims + 1)
    return log_likelihood - num_params / 2.0 * np.log(num_points)


########## MAIN CLASS ########## 

class XMeans:

    # initialize the fields of the class
    def __init__(self, kmin=2, kmax=10, distance_function=euclidean_distances,
                 n_init=10, cluster_centers='k-means++', max_iter=300,
                 tol=1e-4, precompute_distances=True,
                 verbose=0, random_state=None, copy_x=True):
        
        self.kmin = kmin
        self.kmax = kmax
        self.distance_function = distance_function
        self.original_n_init = n_init
        self.n_init = n_init
        self.cluster_centers = cluster_centers
        self.max_iter = max_iter
        self.tol = tol
        self.precompute_distances = precompute_distances
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.labels = None

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.k_ = -1

    # execute x-means on the dataset
    def fit(self, data):
        k = self.kmin
        best_k = -1
        best_bic = - float('inf')
        #best_inertia = -float('inf')
        best_centers = None

        # test at varible values of k
        while k <= self.kmax:
            if self.verbose:
                print('Fitting with k=%d' % k)

            # compute the k-means, can't use the sklean one since it uses the euclidean distance
            k_means = KMeans(init=self.cluster_centers, n_clusters=k, n_init=self.n_init,
                             max_iter=self.max_iter, tol=self.tol, precompute_distances=self.precompute_distances,
                             verbose=False, random_state=self.random_state, copy_x=self.copy_x)
            k_means.fit(data, distance_function=self.distance_function)
            self.labels = k_means.labels_
            self.cluster_centers = k_means.cluster_centers_
            # compute the pairwise distance between all cluster centers
            centroid_distances = self.distance_function(self.cluster_centers, self.cluster_centers)
            # sets the distance between one cluster and itself to infinite
            centroid_distances += np.diag([np.Infinity] * k)
            # the range of each centroid is given by the minimum distance to another center
            centroids_range = centroid_distances.min(axis=-1)

            # try to split dynamically some clusters
            new_cluster_centers = []
            # for each centroid
            for i, centroid in enumerate(self.cluster_centers):
                if self.verbose:
                    print('\tSplitting cluster %d / %d (k=%d)' % ((i+1), len(self.cluster_centers), k))
                # literature version
                # select a random direction, hence a random point
                direction = np.random.random(centroid.shape)
                # compute a step as song as the centroid range along the direction
                # basically I scale the direction point to be at range distance
                vector = direction * (centroids_range[i] / np.sqrt(direction.dot(direction)))
                # i move the new centroid in the two directions computed
                new_point1 = centroid + vector
                new_point2 = centroid - vector

                if self.verbose:
                    print('\tRunning secondary kmeans')

                # takes all the points belonging to the current cluster
                model_index = (self.labels == i)

                # if there's no points in that cluster go to the next cluster
                if not np.any(model_index):
                    if self.verbose:
                        print('\tDisregarding cluster since it has no citizens')
                    continue
                
                # select only the points belonging to that cluster
                points = np.array(data)[model_index]

                # if the cluster is made of only one point then it cannot be split, take new cluster
                if len(points) == 1:
                    if self.verbose:
                        print('\tCluster made by only one citizen')
                    new_cluster_centers.append(centroid)
                    continue
                
                # compute k means only on the cluster points using the new centroids
                # n init indicates the number of times that k-means will be run with different centroids
                child_k_means = KMeans(init=np.asarray([new_point1, new_point2]), n_clusters=2, n_init=1)

                # fit k means on the points of the cluster
                child_k_means.fit(points, distance_function=self.distance_function)
                cluster1 = points[child_k_means.labels_ == 0]
                cluster2 = points[child_k_means.labels_ == 1]

                # if one of the two new clusters is empty don't use the split
                if len(cluster1) == 0 or len(cluster2) == 0:
                    if self.verbose:
                        print('\tUsing parent')
                    new_cluster_centers.append(centroid)
                    continue

                # compute the bic value for the cluster before and after the split
                bic_parent = bic([points], [centroid])
                bic_child = bic([cluster1, cluster2], child_k_means.cluster_centers_)

                if self.verbose:
                    print('\tbic_parent = %f, bic_child = %f' % (bic_parent, bic_child))

                # if bic in child is improved then do the split
                if bic_child > bic_parent:
                    if self.verbose:
                        print('\tUsing children')
                    # add the two new cluster centers
                    new_cluster_centers.append(child_k_means.cluster_centers_[0])
                    new_cluster_centers.append(child_k_means.cluster_centers_[1])
                else:
                    if self.verbose:
                        print('\tUsing parent')
                    # add the old cluster center
                    new_cluster_centers.append(centroid)

            # if no cluster has been split 
            if k == len(new_cluster_centers):
                # compute the bic of all the points
                bic_k = bic(np.asarray([data]), new_cluster_centers)
                # if best result so far save it
                if bic_k > best_bic:
                    best_bic = bic_k
                    best_centers = self.cluster_centers
                    best_k = k

                # restart the x means increasing k
                k += 1
                # selects initial cluster centers in a smart way to speed up convergence
                self.cluster_centers = 'k-means++'
                self.n_init = self.original_n_init
            else:
                # use the splitted cluster as a base, run the algorithm again to see if they should be split further
                k = len(new_cluster_centers)
                self.cluster_centers = np.array(new_cluster_centers)
                self.n_init = 1
        # if at each iteration we the bic was increased by splitting (didn't find optimal k before reaching max_k)
        if best_k == -1:
            # set best k as the max acceptable
            best_k = self.kmax
            # restart with 
            best_centers = 'k-means++'
            self.n_init = self.original_n_init
        # else at some point we found a stable value for k
        else:
            self.n_init = 1

        if self.verbose:
            print('Refining model with k = %d' % best_k)

        # run k means with best value of k and centers found
        k_means_final = KMeans(init=best_centers, n_clusters=best_k, n_init=self.n_init,
                             max_iter=self.max_iter, tol=self.tol, precompute_distances=self.precompute_distances,
                             verbose=False, random_state=self.random_state, copy_x=self.copy_x)
        k_means_final.fit(data, distance_function=self.distance_function)
        # store the result obtained
        self.cluster_centers_ = k_means_final.cluster_centers_
        self.labels_ = k_means_final.labels_
        self.inertia_ = k_means_final.inertia_
        self.k_ = best_k

        return self
