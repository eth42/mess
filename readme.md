# MESS: Manifold Embedding Motivated Super Sampling

This repository contains a demo implementation of the MESS framework from the paper with above title.
The code provided here consists of a small collection of plotting functions based on `plotly`, a few implementations of ID estimators, the MESS framework code as well as a small wrapper function for SMOTE based on the `imblearn` package, and a demo application `demo.py` with configurable parameters that runs the MESS framework and generates two plots:

- An interactive scatter plot that shows the original data as well as supersampled data colored by their respective ID estimates, and
- a histogram plot containing ID histograms of both the original and supersampled data set as well as ID estimates computed for the original data set by using neighborhoods from the supersampled data sets.

The ID estimator is also configurable.
If no ID estimator is specified (`None`), scatter plots are colored by vector norms and no histogram plot is created (generally much faster, as ID estimates can be slow).

## Setup

Simply clone the repository to your local drive and install the required packages via

`python3 -m pip install numpy numba tqdm scipy sklearn imblearn plotly`

or whatever you are using for package installation (conda etc.).
Afterwards, you can run the demo experiment with

`python3 demo.py`

which will run the experiment, giving you a very rough progress report, and plot the results in your browser with `plotly`.

This is by no means optimized code but I commented it quite thoroughly.
If you find bugs, feel free to contact me.
The demo code is structured in parameters, data generation, execution, and plotting, so that you can modify parts of the code as you see fit.
Changing parameters requires extremely little knowledge of Python and should be very accessible to anyone, who has access to the paper.
Changing the data set used in the experiments requires a bit of Python knowledge but not a lot.

Have fun experimenting.

