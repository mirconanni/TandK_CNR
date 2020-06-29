########## IMPORT LIBRARIES ########## 

import warnings
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import row_norms, squared_norm
from sklearn.utils import check_array 
from sklearn.utils import check_random_state
from sklearn.utils import as_float_array
from joblib import Parallel
from joblib import delayed

from sklearn.cluster import _k_means

########## FUNCTION DEFINITION ##########

# init n_clusters seeds according to k-means++
def _k_init(X, n_clusters, x_squared_norms, random_state, n_local_trials=None, distance_function=euclidean_distances):
    """
    X: array or sparse matrix, shape (n_samples, n_features)
        The data to pick seeds for. To avoid memory copy, the input data
        should be double precision (dtype=np.float64).

    n_clusters: integer
        The number of seeds to choose

    x_squared_norms: array, shape (n_samples,)
        Squared Euclidean norm of each data point.

    random_state: numpy.RandomState
        The generator used to initialize the centers.

    n_local_trials: integer, optional
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    """
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features))

    assert x_squared_norms is not None, 'x_squared_norms None in _k_init'

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    center_id = random_state.randint(n_samples)
    centers[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = distance_function(
        centers[0], X, Y_norm_squared=x_squared_norms, squared=True)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(closest_dist_sq.cumsum(), rand_vals)

        # Compute distances to center candidates
        distance_to_candidates = distance_function(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

        # Decide which candidate is the best
        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = np.minimum(closest_dist_sq,
                                     distance_to_candidates[trial])
            new_pot = new_dist_sq.sum()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        # Permanently add best center candidate found in local tries
        centers[c] = X[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return centers

# return a tolerance which is independent of the dataset
def _tolerance(X, tol):
    variances = np.var(X, axis=0)
    return np.mean(variances) * tol

# compute k means up to convergence
def k_means(X, n_clusters, init='k-means++', precompute_distances=True,
            n_init=10, max_iter=300, verbose=False,
            tol=1e-4, random_state=None, copy_x=True, distance_function=euclidean_distances):
    random_state = check_random_state(random_state)

    best_inertia = np.infty
    X = as_float_array(X, copy=copy_x)
    tol = _tolerance(X, tol)

    # subtract of mean of x for more accurate distance computations
    X_mean = X.mean(axis=0)
    # The copy was already done above
    X -= X_mean

    if hasattr(init, '__array__'):
        init = np.asarray(init).copy()
        init -= X_mean
        if n_init != 1:
            warnings.warn(
                'Explicit initial center position passed: '
                'performing only one init in k-means instead of n_init=%d'
                % n_init, RuntimeWarning, stacklevel=2)
            n_init = 1

    # precompute squared norms of data points
    x_squared_norms = row_norms(X, squared=True)

    best_labels, best_inertia, best_centers = None, None, None
    
    # For a single thread, less memory is needed if we just store one set
    # of the best results (as opposed to one set per run per thread).
    for _ in range(n_init):
        # run a k-means once
        labels, inertia, centers = _kmeans_single(
            X, n_clusters, max_iter=max_iter, init=init, verbose=verbose,
            precompute_distances=precompute_distances, tol=tol,
            x_squared_norms=x_squared_norms, random_state=random_state, distance_function=distance_function)
        # determine if these results are the best so far
        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia
    if not copy_x:
        X += X_mean
    best_centers += X_mean

    return best_centers, best_labels, best_inertia

# performs a single run of k means
def _kmeans_single(X, n_clusters, x_squared_norms, max_iter=300,
                   init='k-means++', verbose=False, random_state=None,
                   tol=1e-4, precompute_distances=True, distance_function=euclidean_distances):
    random_state = check_random_state(random_state)

    best_labels, best_inertia, best_centers = None, None, None
    # init
    centers = _init_centroids(X, n_clusters, init, random_state=random_state,
                              x_squared_norms=x_squared_norms, distance_function=distance_function)
    if verbose:
        print('Initialization complete')

    # Allocate memory to store the distances for each sample to its
    # closer center for reallocation in case of ties
    distances = np.zeros(shape=(X.shape[0],), dtype=np.float64)

    # iterations
    for i in range(max_iter):
        centers_old = centers.copy()
        # labels assignment is also called the E-step of EM
        labels, inertia = \
            _labels_inertia(X, x_squared_norms, centers,
                            precompute_distances=precompute_distances,
                            distances=distances)

        # # computation of the means is also called the M-step of EM
        # if sp.issparse(X):
        #     centers = _k_means._centers_sparse(X, labels, n_clusters,
        #                                        distances)
        # else:
        #     centers = _k_means._centers_dense(X, labels, n_clusters, distances)

        n_samples = X.shape[0]
        sample_weight = np.ones(n_samples, dtype=X.dtype)
        # computation of the means is also called the M-step of EM
        centers = _k_means._centers_dense(X, sample_weight, labels,
                                              n_clusters, distances)

        if verbose:
            print('Iteration %2d, inertia %.3f' % (i, inertia))

        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia

        if squared_norm(centers_old - centers) <= tol:
            if verbose:
                print("Converged at iteration %d" % i)
            break
    return best_labels, best_inertia, best_centers

# compute labels and inertia using a full distance matrix
def _labels_inertia_precompute_dense(X, x_squared_norms, centers, distances, distance_function=euclidean_distances):
    n_samples = X.shape[0]
    k = centers.shape[0]
    all_distances = distance_function(centers, X, x_squared_norms,
                                        squared=True)
    labels = np.empty(n_samples, dtype=np.int32)
    labels.fill(-1)
    mindist = np.empty(n_samples)
    mindist.fill(np.infty)
    for center_id in range(k):
        dist = all_distances[center_id]
        labels[dist < mindist] = center_id
        mindist = np.minimum(dist, mindist)
    if n_samples == distances.shape[0]:
        # distances will be changed in-place
        distances[:] = mindist
    inertia = mindist.sum()
    return labels, inertia

# returns the labels and the intertia
def _labels_inertia(X, x_squared_norms, centers,
                    precompute_distances=True, distances=None, distance_function=euclidean_distances):
    n_samples = X.shape[0]
    # set the default value of centers to -1 to be able to detect any anomaly
    # easily
    labels = -np.ones(n_samples, np.int32)
    if distances is None:
        distances = np.zeros(shape=(0,), dtype=np.float64)
    # distances will be changed in-place
    if precompute_distances:
        return _labels_inertia_precompute_dense(X, x_squared_norms,
                                                centers, distances, distance_function=distance_function)
    inertia = _k_means._assign_labels_array(
        X, x_squared_norms, centers, labels, distances=distances)
    return labels, inertia

# compute the initial centroids
def _init_centroids(X, k, init, random_state=None, x_squared_norms=None,
                    init_size=None, distance_function=euclidean_distances):
    random_state = check_random_state(random_state)
    n_samples = X.shape[0]

    # number of samples to randomly sample for speeding up the initialization
    if init_size is not None and init_size < n_samples:
        if init_size < k:
            warnings.warn(
                "init_size=%d should be larger than k=%d. "
                "Setting it to 3*k" % (init_size, k),
                RuntimeWarning, stacklevel=2)
            init_size = 3 * k
        init_indices = random_state.random_integers(
            0, n_samples - 1, init_size)
        X = X[init_indices]
        # compute the squared euclidean norm of each data point, can be passed as parameter if already computed
        x_squared_norms = x_squared_norms[init_indices]
        n_samples = X.shape[0]
    elif n_samples < k:
            raise ValueError(
                "n_samples=%d should be larger than k=%d" % (n_samples, k))

    if isinstance(init, str) and init == 'k-means++':
        centers = _k_init(X, k, random_state=random_state,
                          x_squared_norms=x_squared_norms, distance_function=distance_function)
    elif isinstance(init, str) and init == 'random':
        seeds = random_state.permutation(n_samples)[:k]
        centers = X[seeds]
    elif hasattr(init, '__array__'):
        centers = init
    elif callable(init):
        centers = init(X, k, random_state=random_state)
    else:
        raise ValueError("the init parameter for the k-means should "
                         "be 'k-means++' or 'random' or an ndarray, "
                         "'%s' (type '%s') was passed." % (init, type(init)))

    if len(centers) != k:
        raise ValueError('The shape of the initial centers (%s) '
                         'does not match the number of clusters %i'
                         % (centers.shape, k))

    return centers


########## MAIN CLASS ########## 

class KMeans():

    # initialize the fields of the class
    def __init__(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300,
                 tol=1e-4, precompute_distances=True,
                 verbose=0, random_state=None, copy_x=True):

        if hasattr(init, '__array__'):
            n_clusters = init.shape[0]
            init = np.asarray(init, dtype=np.float64)

        self.n_clusters = n_clusters
        # init specifies a method for initialization of the initial centers, or the array of centroids
        self.init = init
        # max number of iterations of the k-means for a single run
        self.max_iter = max_iter
        # tolerance with regards to inertia to declare convergence
        self.tol = tol
        self.precompute_distances = precompute_distances
        # n_init is the number of times the k-means will be run with different centroid seeds
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        # if copy_x is True, then the original data is not modified when precomputing the distances
        self.copy_x = copy_x

    # verify that the number of samples given is larger than k
    def _check_fit_data(self, X):
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d" % (
                X.shape[0], self.n_clusters))
        return X

    # compute k-means clustering on the dataset
    def fit(self, X, y=None, distance_function=euclidean_distances):
        random_state = check_random_state(self.random_state)
        X = self._check_fit_data(X)

        self.cluster_centers_, self.labels_, self.inertia_ = k_means(
            X, n_clusters=self.n_clusters, init=self.init, n_init=self.n_init,
            max_iter=self.max_iter, verbose=self.verbose,
            precompute_distances=self.precompute_distances,
            tol=self.tol, random_state=random_state, copy_x=self.copy_x,
            distance_function=distance_function)
        return self
