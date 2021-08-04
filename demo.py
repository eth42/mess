from re import A
import numpy as np
from plotly import graph_objects as go
from sklearn.datasets import make_swiss_roll
from supersampling import *
from plotting import *
from id_estimators import *
from tqdm import tqdm

# Simple enum to select the supersampling method.
# Feel free, to add other implementations.
class SUPERSAMPLER(Enum):
	MESS = 0
	SMOTE = 1

############################################
################ PARAMETERS ################
############################################

# Parameters for supersampling
k1 = 10
k2 = 10
ext = 20
# Chose supersampling method
supersampling_method = SUPERSAMPLER.MESS
# These are the default values of the MESS supersampling.
# If you want more details on what these parameters mean,
# please refer to the paper and the comments in the enum
# definitions in 'supersampling.py'.
# Do *not* set the correction parameters, as this is set
# further down in the code. By default, the IDW correction
# will be used with the distance specified here.
# If you wish to change this behavior, change the definition
# below and not here.
additional_mess_parameters = dict(
	distance=DISTANCE.SYM_MAHAL_ADD,
	candidates=CANDIDATES.COVARIANCE,
	generation=GENERATION.COVARIANCE,
	decomp_epsilon=1e-10
)
# Specify what ID estimator to use for scatter plot coloring,
# or None to use point norms as colors.
# As the estimator and the extended estimator using supersamples
# need to match, this is automatically assigned in pairs.
# To change the estimator, change the selection index at the
# end of this definition (remember the zero-based index).
estimator, ext_estimator = [
	[None, None],
	[abids, abids_ext],
	[rabids, rabids_ext],
	[mle_lids, mle_lids_ext],
	[alids, alids_ext],
][1]



############################################
########## DATA IMPORT/GENERATION ##########
############################################

# Import or generate data.
X_full = make_swiss_roll(2000,noise=.15)[0]
# Add uniform noise points to the data set.
X_mins = np.min(X_full,axis=0)
X_maxs = np.max(X_full,axis=0)
X_full = np.concatenate([X_full,np.random.sample((100,3))*(X_maxs-X_mins)+X_mins])
# Probably reduce the size of the data set to lower computation time.
X_reduced = X_full[:500]
# Otherwise uncomment this line. The variables X_full and X need to be defined.
# X_reduced = X_full

# Limitations for the color scale in scatter plots should be data set dependent.
# The swiss roll is a 2d manifold in 3d, whereby [1,3] is a reasonable range
# for ID estimates. The proper ID estimates are added as labels to the points
# when hovering over them in the plots. If you change the data set, be sure
# to modify these values!
color_scale_bounds = [1,3]



############################################
################ EXECUTION #################
############################################
if supersampling_method == SUPERSAMPLER.MESS:
	supersampler = lambda X: supersample_mess(X,k1,k2,ext,**additional_mess_parameters)
	corrected_supersampler = lambda X: supersample_mess(X,k1,k2,ext,correction=CORRECTION.IDW,**additional_mess_parameters)
elif supersampling_method == SUPERSAMPLER.SMOTE:
	supersampler = lambda X: supersample_smote(X,k1,ext)
	# SMOTE doesn't have correction, so this will be the same
	corrected_supersampler = lambda X: supersample_smote(X,k1,ext)


# Collection of all data sets to compute IDs and plots for.
# X_full is reduced to the size of X * ext to not fry the CPU while rendering.
Xs = [np.random.permutation(X_full)[:X_reduced.shape[0] * ext], X_reduced]
Xnames = ["Full original data", "Reduced original data"]
scales = [1,1]
if X_full.shape[0] == X_reduced.shape[0]:
	Xs, Xnames, scales = Xs[:1], Xnames[:1], scales[:1]


# Supersample data set (without correction)
print("Supersampling data without correction.")
X_ext = supersampler(X_reduced)
Xs.append(X_ext)
Xnames.append("Supersampled data")
scales.append(ext)
n_supersamples = 1
# Supersample data set with correction if MESS
if supersampling_method == SUPERSAMPLER.MESS:
	print("Supersampling data with correction.")
	X_ext_corr = corrected_supersampler(X_reduced)
	Xs.append(X_ext_corr)
	Xnames.append("Corrected supersampled data")
	scales.append(ext)
	n_supersamples += 1




if estimator is None:
	# Use vector norms instead of ID estimates
	id_estimates = [
		np.linalg.norm(lX,axis=1)
		for lX,scale in tqdm(
			list(zip(Xs,scales)),
			desc="Computing all required ID estimates (or vector norms) for visualization"
		)
	]
else:
	# Compute ID estimates of all points of all data sets for coloring the scatter plot
	id_estimates = [
		(
			np.linalg.norm(lX,axis=1)
			if estimator is None else
			estimator(lX,k1*scale)
		)
		for lX,scale in tqdm(
			list(zip(Xs,scales)),
			desc="Computing all required ID estimates (or vector norms) for visualization"
		)
	]
	# Computed extended ID estimates using the supersampled points
	# for the original data sets.
	ext_Xs = [
		a
		for a in Xs[:-n_supersamples]
		for b in Xs[-n_supersamples:]
	]
	ext_id_estimates = [
		ext_estimator(lX, lX_ext, k1*ext)
		for lX,lX_ext in tqdm(
			[
				[a,b]
				for a in Xs[:-n_supersamples]
				for b in Xs[-n_supersamples:]
			],
			desc="Computing ID estimates of original data with supersamples"
		)
	]
	ext_names = [
		"{:} using {:}".format(a,b.lower())
		for a in Xnames[:-n_supersamples]
		for b in Xnames[-n_supersamples:]
	]
	# Override lists for plotting
	Xs = [*Xs,*ext_Xs]
	Xnames = [*Xnames,*ext_names]
	id_estimates = [*id_estimates,*ext_id_estimates]



############################################
################# PLOTTING #################
############################################
# Force same scale axes and same axes across all plots
coord_mins = np.min(np.concatenate(Xs,axis=0)[:,:3],axis=0)
coord_maxs = np.max(np.concatenate(Xs,axis=0)[:,:3],axis=0)
coord_mids = .5*(coord_mins+coord_maxs)
coord_max_span = np.max(coord_maxs-coord_mins)
layout_args = dict(
	scene=dict(
		xaxis=dict(range=[coord_mids[0]-.5*coord_max_span,coord_mids[0]+.5*coord_max_span]),
		yaxis=dict(range=[coord_mids[1]-.5*coord_max_span,coord_mids[1]+.5*coord_max_span]),
		zaxis=dict(range=[coord_mids[2]-.5*coord_max_span,coord_mids[2]+.5*coord_max_span]),
		aspectmode="manual",
		aspectratio=dict(x=1,y=1,z=1),
	)
)

if estimator is None:
	all_norms = np.concatenate(id_estimates)
	color_scale_bounds = [np.min(all_norms), np.max(all_norms)]
color_scale_dtick = max(.5,2**np.floor(np.log2((color_scale_bounds[1]-color_scale_bounds[0])/10)))

# Render scatter plots and display in browser
animated_figure(
	[
		go.Scatter3d(
				x=lX[:,0],
				y=lX[:,1],
				z=lX[:,2],
				text=["{:} = {:.4f}".format(estimator.__name__ if not estimator is None else "Vector norm",i) for i in ids],
				mode="markers",
				marker=dict(
					size=30 / np.log(lX.shape[0]),
					color=ids,
					cmin=color_scale_bounds[0],
					cmax=color_scale_bounds[1],
					colorbar=dict(
						title=estimator.__name__ if not estimator is None else "Vector norm",
						dtick=color_scale_dtick
					)
				)
		)
		for lX, ids in zip(Xs,id_estimates)
	],
	titles=Xnames,
	names=Xnames,
	instant=True,
	**layout_args
).show(renderer="browser")



# Render histograms and display in browser
if not ext_estimator is None:
	go.Figure(
		[
			go.Histogram(
				x=ids,
				name=name,
				opacity=.7,
				histnorm="probability density",
				xbins=dict(
					start=color_scale_bounds[0],
					end=color_scale_bounds[1],
					size=.05
				)
			)
			for ids,name in zip(id_estimates,Xnames)
		],
		layout=dict(
			barmode="overlay",
			title="ID Histograms",
			xaxis=dict(title=estimator.__name__),
			yaxis=dict(title="% of points")
		)
	).show(renderer="browser")



