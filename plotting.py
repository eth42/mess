import numpy as np
from plotly import graph_objects as go
import re

def show_plots(*traces):
	go.Figure(
		traces,
		layout=dict(
			yaxis=dict(scaleanchor="x")
		)
	).show(renderer="browser")
def gen_mnist(im, transpose=True):
		s = int((im.shape[0] / 3)**.5)
		if 3*s*s == im.shape[0]:
				is_rgb = True
		else:
				is_rgb = False
				s = int(im.shape[0]**.5)
		d = 3 if is_rgb else 1
		x = np.zeros((s,s,3))
		for i in range(3):
				j = s*s*(i % d)
				k = j + s*s
				x[:,:,i] = im[j:k].reshape((s,s))
				if transpose: x[:,:,i] = x[:,:,i].T
				if np.max(x[:,:,i]) < 2: x[:,:,i] *= 255
		return go.Image(z=x)
def animated_figure(traces, titles=None, names=None, instant=False, animation_length=500, **kwargs):
	if type(traces[0]) != list: traces = [[t] for t in traces]
	if titles is None: titles = ["Frame {}".format(i) for i in range(len(traces))]
	if names is None: names = [str(i) for i in range(len(traces))]
	fig = go.Figure(
		data=traces[0],
		frames=[
			go.Frame(data=ts,name=name,layout=dict(title=title))
			for name,ts,title in zip(names,traces,titles)
		],
		layout=dict(
			title=titles[0],
			updatemenus=[dict(
				type="buttons",
				buttons=[dict(
					label="Play",
					method="animate",
					args=[None,dict(
						frame=dict(duration=animation_length),
						mode="immediate",
						transition=dict(duration=0 if instant else animation_length)
					)]
				)]
			)],
			sliders=[dict(steps=[
				dict(
					args=[
						[name],
						dict(
							frame=dict(duration=0 if instant else animation_length),
							mode="immediate",
							transition=dict(duration=0 if instant else animation_length)
						)
					],
					label=name,
					method="animate"
				)
				for name in names
			])],
			**kwargs
		)
	)
	return fig


# Loading data from remote ad hoc and displaying
def gen_plot_name(row):
	if np.isnan(row['k1']):
		return "{:6.3f}±{:6.3f} {:}({:}) - {:}".format(
			row['mean'],row['std'],row['estimator'],row['k3'],row.name
		)
	enum_regex = re.compile(r'^[^\.]+\.(.*)$')
	get_enum_name = lambda s,maxlen: enum_regex.match(s).group(1)[:maxlen] if len(s) > 0 else "-"
	return "{:6.3f}±{:6.3f} {:}({:}) {:} {:} {:} {:} {:d} {:d} {:d} - {:}".format(
		row['mean'],row['std'],row['estimator'],row['k3'],
		get_enum_name(row['gen'],4),
		get_enum_name(row['corr'],20),
		get_enum_name(row['cand'],4),
		get_enum_name(row['dist'],5),
		int(row['k1']),
		int(row['k2']),
		int(row['ext']),
		row.name
	)
def show_histograms(hists,**layout_args):
	go.Figure(hists,layout=dict(**layout_args,barmode="overlay")).show(renderer="browser")



def mscatter(X,**kwargs):
	margs = dict(
		x=X[:,0],
		y=X[:,1],
		mode="markers"
	)
	for k,v in kwargs.items():
		margs[k] = v
	return go.Scatter(**margs)
def mscatter3(X,**kwargs):
	margs = dict(
		x=X[:,0],
		y=X[:,1],
		z=X[:,2],
		mode="markers",
		marker=dict(size=2,color=np.linalg.norm(X,axis=1) if not None in X else None)
	)
	for k,v in kwargs.items():
		margs[k] = v
	return go.Scatter3d(**margs)
def ellipsoid(mean, cov, n_samples=20, scale=1):
	vals,vecs = np.linalg.eig(cov)
	vecs = vecs.T
	ret = []
	for j in range(len(vecs)):
		ret.extend([
			mean
			+ scale * np.cos(a)*vecs[j]*vals[j]**.5
			+ scale * np.sin(a)*vecs[(j+1)%len(vals)]*vals[(j+1)%len(vals)]**.5
			for a in np.linspace(0,2*np.pi,n_samples+1)
		])
		ret.append([None]*mean.shape[0])
	return np.array(ret)



class COLORS:
	ORANGES=[
		"rgb(242,200,91)",
		"rgb(251,164,101)",
		"rgb(248,110,81)",
		"rgb(238,62,56)",
		"rgb(209,25,62)",
	]
	BLUES=[
		"rgb(83,204,236)",
		"rgb(25,116,211)",
		"rgb(0,1,129)",
	]
	GREENS=[
		"rgb(204,255,204)",
		"rgb(179,230,185)",
		"rgb(153,204,166)",
		"rgb(128,179,147)",
		"rgb(102,153,128)",
		"rgb(77,128,108)",
		"rgb(51,102,89)",
		"rgb(26,77,70)",
		"rgb(0,51,51)",
	]
	TOLERANCE=[
		"rgb(51,34,136)",
		"rgb(17,119,51)",
		"rgb(68,170,153)",
		"rgb(136,204,238)",
		"rgb(221,204,119)",
		"rgb(204,102,119)",
		"rgb(170,68,153)",
		"rgb(136,34,85)",
	]
	SHOWCASE=[
		"rgb(51,34,136)",
		"rgb(115,199,185)",
		"rgb(204,102,119)",
	]
