import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

# Estimators are split into three functions.
# The "singular" versions (abid, rabid, mle_lid, alid) correspond to the estimate
# for a single point using the k nearest neighbors from the given data set and
# search structure. The parameter "offset" which should be 0 or 1 controls
# whether the point of interest is inside the data set X.
# The "normal plural" versions (abids, ...) compute estimates for all
# points in the data set using the "singular" version with offset=1, as estimates
# are computed for points inside the data set.
# The "extended plural" versions (abids_ext, ...) compute all estimates for all
# points in a data set X using neighborhoods from the extended (e.g. supersampled)
# data set X_ext. They, therefore, use offset=0, as the points are expected to not
# lie within the data set for nearest neighbor computations.


def abids(X,k):
    search_struct = cKDTree(X)
    return np.array([
        abid(X,k,x,search_struct)
        for x in tqdm(X,desc="abids",leave=False)
    ])
def abids_ext(X,X_ext,k):
    search_struct = cKDTree(X_ext)
    return np.array([
        abid(X_ext,k,x,search_struct,offset=0)
        for x in tqdm(X,desc="abids_ext",leave=False)
    ])
def abid(X,k,x,search_struct,offset=1):
    neighbor_norms, neighbors = search_struct.query(x,k+offset)
    neighbors = X[neighbors[offset:]] - x
    normed_neighbors = neighbors / neighbor_norms[offset:,None]
    # Original publication version that computes all cosines
    # coss = normed_neighbors.dot(normed_neighbors.T)
    # return np.mean(np.square(coss))**-1
    # Using another product to get the same values with less effort
    para_coss = normed_neighbors.T.dot(normed_neighbors)
    return k**2 / np.sum(np.square(para_coss))


def rabids(X,k):
    search_struct = cKDTree(X)
    return np.array([
        rabid(X,k,x,search_struct)
        for x in tqdm(X,desc="rabids",leave=False)
    ])
def rabids_ext(X,X_ext,k):
    search_struct = cKDTree(X_ext)
    return np.array([
        rabid(X_ext,k,x,search_struct,offset=0)
        for x in tqdm(X,desc="rabids_ext",leave=False)
    ])
def rabid(X,k,x,search_struct,offset=1):
    d = X.shape[1]
    neighbor_norms, neighbors = search_struct.query(x,k+offset)
    neighbors = X[neighbors[offset:]] - x
    normed_neighbors = neighbors / neighbor_norms[offset:,None]
    # Original publication version that computes all cosines
    # coss = normed_neighbors.dot(normed_neighbors.T)
    # return (k**2-k) / (np.sum(np.square(coss))-k)
    # Using another product to get the same values with less effort
    para_coss = normed_neighbors.T.dot(normed_neighbors)
    return (k**2-k) / (np.sum(np.square(para_coss))-k)


def mle_lids(X,k):
    search_struct = cKDTree(X)
    return np.array([
        mle_lid(X,k,x,search_struct)
        for x in tqdm(X,desc="mle_lids",leave=False)
    ])
def mle_lids_ext(X,X_ext,k):
    search_struct = cKDTree(X_ext)
    return np.array([
        mle_lid(X_ext,k,x,search_struct,offset=0)
        for x in tqdm(X,desc="mle_lids_ext",leave=False)
    ])
def mle_lid(X,k,x,search_struct,offset=1):
    dists, _ = search_struct.query(x,k+offset)
    w = dists[-1]
    return -k * np.sum([np.log(d/w) for d in dists[offset:]])**-1


def alids(X,k):
    search_struct = cKDTree(X)
    return np.array([
        alid(X,k,x,search_struct)
        for x in tqdm(X,desc="alids",leave=False)
    ])
def alids_ext(X,X_ext,k):
    search_struct = cKDTree(X_ext)
    return np.array([
        alid(X_ext,k,x,search_struct,offset=0)
        for x in tqdm(X,desc="alids_ext",leave=False)
    ])
def alid(X,k,x,search_struct,offset=1):
    dists, knni = search_struct.query(x,k+offset)
    inter_dists = euclidean_distances(X[knni])
    w = dists[-1]
    dmat = np.array([dists for _ in range(k+offset)])
    mask = inter_dists + dmat < w
    np.fill_diagonal(mask, False)
    rho = np.sum(mask[offset:,:][:,offset:])
    logs = np.log(inter_dists[mask].flatten() / (w-dmat[mask].flatten()))
    lsum = np.sum(logs)
    if lsum >= 0: return 1
    return -(k+rho) / lsum
