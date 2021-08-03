import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree
from enum import Enum
from numba import njit
from id_estimators import *
from imblearn.over_sampling import SMOTE, ADASYN

class GENERATION(Enum):
	# Defines how initial supersamples are generated.
	COVARIANCE = 0 # Draw supersamples from local multivariate normal distribution
	CHOLESKY = 1 # Draw supersamples uniformly from a d-ball and transform with cholesky decomposition
	EIGEN_ID = 2 # Draw supersamples uniformly from a ID-ball (fixed ID) and transform with eigen decomposition
	EIGEN_IDEST = 3 # Draw supersamples uniformly from a ID-ball (estimated ID) and transform with eigen decomposition
	BALL_1 = 4 # Draw supersamples uniformly from a ball with 1-nearest-neighbor-distance radius
	BALL_K_HALF = 5 # Draw supersamples uniformly from a ball with (k/2)-nearest-neighbor-distance radius
	BALL_K = 6 # Draw supersamples uniformly from a ball with k-nearest-neighbor-distance radius

class CORRECTION(Enum):
	# Defines what correction scheme to use.
	NONE = 0 # Do not apply correction
	MEAN = 1 # Correction using mean of candidates without exponent
	IDW = 2 # Correction using inverse distance weights without exponent
	IDW_ID = 3 # Correction using IDW with fixed exponent
	IDW_IDEST = 4 # Correction using IDW and exponents from ID estimator

class CANDIDATES(Enum):
	# Defines how to produce candidates for the weighted mean in the correction step.
	COVARIANCE = 0 # Compute correction candidates by applying the covariance matrix to the supersample
	CHOLESKY = 1 # Compute correction candidates by applying the cholesky matrix to the supersample

class DISTANCE(Enum):
	# Defines what distance to use for distance-based weighting schemes where distances
	# are computed between a supersampled point and a point in the original data set.
	SYM_MAHAL_MULT = 0 # Symmetrized Mahalanobis distance with multiplication (square root of product)
	SYM_MAHAL_ADD = 1 # Symmetrized Mahalanobis distance with addition (mean of distances)
	SYM_MAHAL_SQ_ADD = 2 # Symmetrized Mahalanobis distance with addition (square root of mean of squared distances)
	MAHAL_FORWARD = 3 # Mahalanobis distance using the covariance matrix of the original data point
	MAHAL_BACKWARD = 4 # Mahalanobis distance using the covariance matrix of the supersampled point
	EUCLIDEAN = 5 # Euclidean distance

# Helper functions
@njit
def sample_d_ball(d,r,n):
    data = np.random.normal(0,1,(n,d))
    data /= np.linalg.norm(data,axis=1)[:,None]
    data *= r * (np.random.sample(n)**(1./d))[:,None]
    return data
@njit
def sample_dc_ball(cov,id,n):
    evals,evecs = np.linalg.eig(cov)
    stds = np.maximum(0,np.real(evals))**.5
    ret = np.random.normal(0,1,(n,cov.shape[0])) * stds
    ret *= (np.random.sample(n)**(1./id) / np.linalg.norm(ret,axis=1))[:,None]
    ret *= stds
    return ret.dot(np.real(evecs).T)
@njit
def mcov(X,ddof=0):
	return 1/(X.shape[0]-ddof) * X.T.dot(X)
@njit
def mahal(x,mean,icov):
	return (x-mean).dot(icov).dot(x-mean)**.5
@njit
def sym_mahal_add(x,y,xicov,yicov):
	return .5 * (mahal(x,y,yicov) + mahal(y,x,xicov))
@njit
def sym_mahal_sq_add(x,y,xicov,yicov):
	return (.5 * (mahal(x,y,yicov)**2 + mahal(y,x,xicov)**2))**.5
@njit
def sym_mahal_mult(x,y,xicov,yicov):
	return (mahal(x,y,yicov) * mahal(y,x,xicov))**.5


# An implementation of the MESS framework with all variants of the
# generation and correction rules introduced in the paper.
# This is much more complicated, than simply implementing one
# set of rules, as certain steps are only required for certain
# rules like precomputing covariance matrices.
def supersample_mess(
	X, k1, k2, ext,
	correction=CORRECTION.NONE,
	static_id=None, estimator=None,
	distance=DISTANCE.SYM_MAHAL_ADD,
	candidates=CANDIDATES.COVARIANCE,
	generation=GENERATION.COVARIANCE,
	decomp_epsilon=1e-10):
	# Search structure for nearest neighbor queries
	tree = cKDTree(X)
	# Initialize output array
	X_ext = np.zeros((X.shape[0]*ext, X.shape[1]))
	# Precompute covariance matrices
	covs = np.array([
		mcov(X[tree.query(x,k1+1)[1][1:]]-x)
		for x in tqdm(X,desc="Computing covariance matrices for X",leave=False)
	])
	# Potentially precompute cholesky decompositions
	if generation in [GENERATION.CHOLESKY] or candidates == CANDIDATES.CHOLESKY:
		chols = np.array([np.linalg.cholesky(c+np.eye(c.shape[0])*decomp_epsilon) for c in covs])
	if generation == GENERATION.EIGEN_IDEST or correction == CORRECTION.IDW_IDEST:
		id_ests = estimator(X, k1)
	# Generate initial (uncorrected) supersamples
	for i,x in tqdm(enumerate(X),total=X.shape[0],desc="Supersampling data",leave=False):
		if generation == GENERATION.COVARIANCE:
			X_ext[ext*i:ext*(i+1)] = np.random.multivariate_normal(
				x, covs[i], ext
			)
		elif generation == GENERATION.CHOLESKY:
			X_ext[ext*i:ext*(i+1)] = (
				sample_d_ball(X.shape[1],1,ext)
				.dot(chols[i].T)
				+ x
			)
		elif generation == GENERATION.EIGEN_ID:
			X_ext[ext*i:ext*(i+1)] = (
				sample_dc_ball(
					covs[i]+np.eye(covs[i].shape[0])*decomp_epsilon,
					static_id,
					ext
				)
				+ x
			)
		elif generation == GENERATION.EIGEN_IDEST:
			X_ext[ext*i:ext*(i+1)] = (
				sample_dc_ball(
					covs[i]+np.eye(covs[i].shape[0])*decomp_epsilon,
					id_ests[i],
					ext
				)
				+ x
			)
		elif generation == GENERATION.BALL_1:
			X_ext[ext*i:ext*(i+1)] = sample_d_ball(
				X.shape[1], tree.query(x,2)[0][-1], ext
			)
		elif generation == GENERATION.BALL_K_HALF:
			X_ext[ext*i:ext*(i+1)] = sample_d_ball(
				X.shape[1], tree.query(x,k1//2+1)[0][-1], ext
			)
		elif generation == GENERATION.BALL_K:
			X_ext[ext*i:ext*(i+1)] = sample_d_ball(
				X.shape[1], tree.query(x,k1+1)[0][-1], ext
			)
	# If no correction, return immediately
	if correction == CORRECTION.NONE: return X_ext
	# Otherwise compute corrections
	# Get nearest neighbors in original data set of all supersamples
	neighborss = tree.query(X_ext,k2)[1]
	# Prepare covariance and inverse covariance matrices
	icovs = np.array([np.linalg.pinv(c) for c in covs])
	if distance in [DISTANCE.SYM_MAHAL_ADD, DISTANCE.SYM_MAHAL_SQ_ADD, DISTANCE.SYM_MAHAL_MULT, DISTANCE.MAHAL_BACKWARD]:
		covs_ext = np.array([
			mcov(X[nn]-x)
			for x,nn in zip(X_ext,neighborss)
		])
		icovs_ext = np.array([np.linalg.pinv(c) for c in covs_ext])
	else: icovs_ext = np.zeros(X_ext.shape[0])
	# Depending on the correction scheme, exponents might be necessary
	if correction in [CORRECTION.MEAN, CORRECTION.IDW]: exponents = np.ones(X.shape[0])
	elif correction == CORRECTION.IDW_ID: exponents = np.full(X.shape[0],static_id)
	elif correction == CORRECTION.IDW_IDEST: exponents = id_ests
	else: raise ValueError("correction must be from the CORRECTION enum")
	# Apply correction
	for i,x in tqdm(enumerate(X_ext),total=X_ext.shape[0],desc="Smoothing supersample",leave=False):
		neighbors = neighborss[i]
		neighbor_dists = np.linalg.norm(X[neighbors]-x,axis=1)
		neighbor_vecs = X[neighbors]
		# Directional vectors from neighbors to sample scaled with covariance
		if candidates == CANDIDATES.COVARIANCE:
			offcenters = np.array([
				(x-y).dot(c)
				for c,y in zip(covs[neighbors], neighbor_vecs)
			])
		elif candidates == CANDIDATES.CHOLESKY:
			offcenters = np.array([
				(x-y).dot(l.T)
				for l,y in zip(chols[neighbors], neighbor_vecs)
			])
		offcenter_norms = np.linalg.norm(offcenters,axis=1)
		# Absolute vectors corrected with covariance scale and constant norm from neighbors
		lcandidates = neighbor_vecs + offcenters * (neighbor_dists / offcenter_norms)[:,None]
		# Computing weights
		icov_x = icovs_ext[i]
		# If using distance-based weights, compute corresponding distances
		if correction in [CORRECTION.IDW, CORRECTION.IDW_ID, CORRECTION.IDW_IDEST]:
			if distance == DISTANCE.SYM_MAHAL_ADD:
				weights = np.array([
					1 / sym_mahal_sq_add(x,y,icov_x,ic)
					for y,ic in zip(neighbor_vecs, icovs[neighbors])
				])
			elif distance == DISTANCE.SYM_MAHAL_SQ_ADD:
				weights = np.array([
					1 / sym_mahal_add(x,y,icov_x,ic)
					for y,ic in zip(neighbor_vecs, icovs[neighbors])
				])
			elif distance == DISTANCE.SYM_MAHAL_MULT:
				weights = np.array([
					1 / sym_mahal_mult(x,y,icov_x,ic)
					for y,ic in zip(neighbor_vecs, icovs[neighbors])
				])
			elif distance == DISTANCE.MAHAL_FORWARD:
				weights = np.array([
					1 / mahal(x,y,ic)
					for y,ic in zip(neighbor_vecs, icovs[neighbors])
				])
			elif distance == DISTANCE.MAHAL_BACKWARD:
				weights = np.array([
					1 / mahal(y,x,icov_x)
					for y in neighbor_vecs
				])
			elif distance == DISTANCE.EUCLIDEAN:
				weights = 1 / np.linalg.norm(neighbor_vecs-x,axis=1)
			else: raise ValueError("distance must be from the DISTANCE enum")
		elif correction == CORRECTION.MEAN:
			weights = np.full(len(lcandidates), 1/len(lcandidates))
		# Potentially exponentiate with prior ID estimate
		weights = weights ** exponents[neighbors]
		weights /= np.sum(weights)
		X_ext[i] = np.sum(lcandidates * weights[:,None],axis=0)
	# Return corrected supersamples
	return X_ext


# This is a bit of a hack, as the SMOTE implementation requires class labels
# and makes the smaller class as big as the larger class.
# For that, we use a fake "majority class" with the desired extended size
# and use the entire data set as a smaller "minority class".
# By overriding the variant argument, different versions of SMOTE like ADASYN
# can be used.
def supersample_smote(X, k, ext, variant=SMOTE, **constr_args):
	oversampler = variant(
		sampling_strategy='minority',
		k_neighbors=k,
		**constr_args
	)
	fake_majority = np.zeros((X.shape[0]*(ext+1),X.shape[1]))
	fake_X = np.concatenate([X,fake_majority],axis=0)
	fake_y = (np.arange(X.shape[0] + fake_majority.shape[0]) < X.shape[0]).astype(int)
	oversampling = oversampler.fit_resample(fake_X,fake_y)
	oversampled_X = oversampling[0][oversampling[1].astype(bool)]
	return oversampled_X[X.shape[0]:]


