########## IMPORT LIBRARIES ########## 
# necessary for math formula
import numpy as np
# compute trigonometric functions
from math import radians, degrees, sin, cos, asin, acos, sqrt
# to check if a matrix is sparse
from scipy.sparse import issparse
# compute operations on matrix
from sklearn.utils.extmath import row_norms, safe_sparse_dot
# check if array is well formed
from sklearn.utils import check_array


########## FUNCTION DEFINITION ##########

# compute the closest point on segment AB from p
def closest_point_on_segment(a, b, p):
    sx1 = a[0]
    sx2 = b[0]
    sy1 = a[1]
    sy2 = b[1]
    px = p[0]
    py = p[1]

    x_delta = sx2 - sx1
    y_delta = sy2 - sy1

    # if a and b are on the same axis return p itself (we travelled on a line)
    if x_delta == 0 and y_delta == 0:
        return p

    # compute the projection of p on ab
    u = ((px - sx1) * x_delta + (py - sy1) * y_delta) / (x_delta * x_delta + y_delta * y_delta)
    # if the projection falls before the segment then the closest point from the segment to p is a
    if u < 0.00001:
        closest_point = a
    # if it falls over the segment then the closest point from the segment to p is b
    elif u > 1:
        closest_point = b
    # if the projection falls in the middle of the segment
    else:
        # compute the coordinates of the closest point
        cp_x = sx1 + u * x_delta
        cp_y = sy1 + u * y_delta
        closest_point = [cp_x, cp_y]

    return closest_point

# compute the distance between two trajectories
def trajectory_distance(tr1, tr2):
    i1 = 0
    i2 = 0
    np = 0

    # takes first point of each trajectory
    last_tr1 = tr1[i1]
    last_tr2 = tr2[i2]

    # compute distance
    dist = spherical_distance(last_tr1, last_tr2)
    np += 1

    while True:

        # compute the distance between one point and the next
        step_tr1 = spherical_distance(last_tr1, tr1[i1+1])
        step_tr2 = spherical_distance(last_tr2, tr2[i2+1])

        # if the fist trajectory does a smaller step
        if step_tr1 < step_tr2:
            # take the next point of traj1
            i1 += 1
            last_tr1 = tr1[i1]
            # the new point of traj 2 is the closest point between the last point two points of traj2 and this new point of traj1
            last_tr2 = closest_point_on_segment(last_tr2, tr2[i2+1], last_tr1)
        # same thing for traj 2
        elif step_tr1 > step_tr2:
            i2 += 1
            last_tr2 = tr2[i2]
            last_tr1 = closest_point_on_segment(last_tr1, tr1[i1+1], last_tr2)
        # ow perform a step on both trajectories
        else:
            i1 += 1
            i2 += 1
            last_tr1 = tr1[i1]
            last_tr2 = tr2[i2]

        # recompute the distance between the two new points
        d = spherical_distance(last_tr1, last_tr2)

        dist += d
        np += 1

        # if we reach the end of either one of the trajectories
        if i1 >= (len(tr1)-1) or i2 >= (len(tr2)-1):
            break

    # add the distance for the remaining points of traj 1
    for i in range(i1, len(tr1)):
        d = spherical_distance(tr2[-1], tr1[i])
        dist += d
        np += 1
    # add the distance for the remaining points of traj 2
    for i in range(i2, len(tr2)):
        d = spherical_distance(tr1[-1], tr2[i])
        dist += d
        np += 1

    # normalize the distance according to the number of points
    dist = 1.0 * dist / np

    return dist

# if dtype of X and Y is float32, then dtype float32 is returned, ow dtype float is returned
def _return_float_dtype(X, Y): 
    if not issparse(X) and not isinstance(X, np.ndarray):
        X = np.asarray(X)

    if Y is None:
        Y_dtype = X.dtype
    elif not issparse(Y) and not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)
        Y_dtype = Y.dtype
    else:
        Y_dtype = Y.dtype

    if X.dtype == Y_dtype == np.float32:
        dtype = np.float32
    else:
        dtype = np.float

    return X, Y, dtype

# checks if the dimensions of two arrays are compatible
def _check_pairwise_arrays(X, Y):
    X, Y, dtype = _return_float_dtype(X, Y)

    if Y is X or Y is None:
        X = Y = check_array(X, accept_sparse='csr', dtype=dtype)
    else:
        X = check_array(X, accept_sparse='csr', dtype=dtype)
        Y = check_array(Y, accept_sparse='csr', dtype=dtype)
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y matrices: "
                         "X.shape[1] == %d while Y.shape[1] == %d" % (
                             X.shape[1], Y.shape[1]))

    return X, Y

# compute the pairwise euclidean distance of elements in two arrays
def euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False):
    X4c = X.reshape(1, -1) if len(X.shape) == 1 else X
    Y4c = Y.reshape(1, -1) if len(Y.shape) == 1 else Y
    X, Y = _check_pairwise_arrays(X4c, Y4c)

    if Y_norm_squared is not None:
        YY = check_array(Y_norm_squared)
        if YY.shape != (1, Y.shape[0]):
            raise ValueError(
                "Incompatible dimensions for Y and Y_norm_squared")
    else:
        YY = row_norms(Y, squared=True)[np.newaxis, :]

    if X is Y:  # shortcut in the common case euclidean_distances(X, X)
        XX = YY.T
    else:
        XX = row_norms(X, squared=True)[:, np.newaxis]

    distances = safe_sparse_dot(X, Y.T, dense_output=True)
    distances *= -2
    distances += XX
    distances += YY
    np.maximum(distances, 0, out=distances)

    if X is Y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        distances.flat[::distances.shape[0] + 1] = 0.0

    return distances if squared else np.sqrt(distances, out=distances)

# compute the pairwise spherical distance of elements in two arrays
def spherical_distances(X, Y=None, Y_norm_squared=None, squared=False):
    X4c = X.reshape(1, -1) if len(X.shape) == 1 else X
    Y4c = Y.reshape(1, -1) if len(Y.shape) == 1 else Y
    X, Y = _check_pairwise_arrays(X4c, Y4c)

    distances = list()

    for x in X:
        dist = list()
        for y in Y:
            dist.append(spherical_distance(x, y))
        distances.append(dist)

    distances = np.array(distances)
    if X is Y:
        distances.flat[::distances.shape[0] + 1] = 0.0
    return distances

# compute the euclidean distance between two points
def euclidean_distance(a, b):
    sq_diff_x = (a[0] - b[0])**2
    sq_diff_y = (a[1] - b[1])**2
    return sqrt(sq_diff_x + sq_diff_y)

# compute the spherical distance using the harvesine distance
def spherical_distance(a, b):
    lat1 = a[1]
    lon1 = a[0]
    lat2 = b[1]
    lon2 = b[0]
    R = 6371000.0
    rlon1, rlat1, rlon2, rlat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = (rlon1 - rlon2) / 2.0
    dlat = (rlat1 - rlat2) / 2.0
    lat12 = (rlat1 + rlat2) / 2.0
    sindlat = sin(dlat)
    sindlon = sin(dlon)
    cosdlon = cos(dlon)
    coslat12 = cos(lat12)
    f = sindlat * sindlat * cosdlon * cosdlon + sindlon * sindlon * coslat12 * coslat12
    f = sqrt(f)
    f = asin(f) * 2.0 # the angle between the points
    f *= R
    return f