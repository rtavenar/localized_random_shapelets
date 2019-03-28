# Localized Random Shapelets

Code for the paper "Localized Random Shapelets" submitted to PAKDD 2019.

## Files
 
- `models.py` contains the definition of several variants of the Localized Random Shapelet model
- <https://github.com/axadil/random-shapelets> is used to draw shapelets and compute shapelet transforms from a time series dataset, code has been modified to draw the shapelets according our procedure
- `plot_shapelet.py` is used to plot shapelet and the distribution in the model interpretability analysis
- `coefficients_computation` is used to compute the coefficients in the model interpretability analysis
- `xp_ucr_classif.py` is used to perform classification on UCR data (shapelet-transformed time series)
- `xp_linear_regression.py` is the code used to generate Fig. 3 in the paper
- `xp_visu_shapelets.py` is the code used to generate Fig. 1 in the paper

## Repositories

- `TwoPatterns_data` contains the TwoPatterns dataset and its transformation used in the model interpretability analysis
	- `TwoPatterns_train_ts.txt` contains the time series of the TwoPatterns train dataset, first column is the class of the times series
	- `TwoPatterns_train_shaprep.csv` contains the time series of the TwoPatterns train dataset transformed by the shapelet representation
	- `TwoPatterns_reglog_weights.txt` contains the weights of the regression logistic trained for the model interpretability analysis

## Result format

In our UCR experiments, we have drawn 2,000 shapelets for each dataset.
Hence, we have one result files, and its format is standard CSV.
We have computed critical diagrams and reported them in the paper. 

