# Localized Random Shapelets

Code for the paper "Localized Random Shapelet model" submitted to ECML 2018.

## Files
 
- `models.py` contains the definition of several variants of the Localized Random Shapelets model
- **TODO** is used to draw shapelets and compute shapelet transforms from a time series dataset
- `xp_ucr_classif.py` is used to perform classification on UCR data (shapelet-transformed time series)
- `xp_linear_regression.py` is the code used to generate Fig. 5 in the paper

## Dataset format

In the `datasets/` folder, you can find datasets denoted `A` and `B` in the paper.
In each subfolder, you will find 6 files (here, for dataset `A`) :

- `synthetic_dataset_A_TS_train.csv` : contains the training time series
- `synthetic_dataset_A_Xtrain.csv` : contains the shapelet transform for training time series
- `synthetic_dataset_A_ytrain.csv` : contains the labels for training time series
- `synthetic_dataset_A_TS_test.csv` : contains the test time series
- `synthetic_dataset_A_Xtest.csv` : contains the shapelet transform for test time series
- `synthetic_dataset_A_ytest.csv` : contains the labels for test time series

## Result format

In our UCR experiments, we have drawn 5 sets of 2,000 shapelets for each dataset.
Hence, we have 5 result files (each for one draw), and their format is standard CSV.
We have computed average ranks for all draws and the mean of these values is reported in the paper. 

