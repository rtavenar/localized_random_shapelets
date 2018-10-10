# Localized Random Shapelets

Code for the paper "Localized Random Shapelets" submitted to PAKDD 2019.

## Files
 
- `models.py` contains the definition of several variants of the Localized Random Shapelet model
- <https://github.com/axadil/random-shapelets> is used to draw shapelets and compute shapelet transforms from a time series dataset
- `xp_ucr_classif.py` is used to perform classification on UCR data (shapelet-transformed time series)
- `xp_linear_regression.py` is the code used to generate Fig. 3 in the paper
- `xp_visu_shapelets.py` is the code used to generate Fig. 1 in the paper

## Result format

In our UCR experiments, we have drawn 5 sets of 2,000 shapelets for each dataset.
Hence, we have 5 result files (each for one draw), and their format is standard CSV.
We have computed average ranks for all draws and the mean of these values is reported in the paper. 

