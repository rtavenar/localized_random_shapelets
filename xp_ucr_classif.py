from models import ssgl_multi_layer_perceptron
import numpy
import pandas
import os
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
import argparse


def read_shapelets(dataset_name, path="data/", index=0, shuffle=False):
    fname_hdf5_labels = os.path.join(path, "labels.h5")
    fname_hdf5 = os.path.join(path, "shapelet_transform.h5")

    store_UCR = pandas.HDFStore(fname_hdf5_labels, mode='r')
    y_train = numpy.array(store_UCR['/%s/TRAIN/labels' % dataset_name], dtype=numpy.int32)
    y_test = numpy.array(store_UCR['/%s/TEST/labels' % dataset_name], dtype=numpy.int32)
    store_UCR.close()

    store = pandas.HDFStore(fname_hdf5, mode='r')
    X_train = numpy.array(store['/%s/%d/TRAIN/' % (dataset_name, index)])
    X_test = numpy.array(store['/%s/%d/TEST/' % (dataset_name, index)])
    store.close()

    scaler = StandardScaler().fit(X_train)
    Xtrain_norm = scaler.transform(X_train)
    Xtest_norm = scaler.transform(X_test)

    if shuffle:
        indices = numpy.random.permutation(Xtrain_norm.shape[0])
        Xtrain_norm = Xtrain_norm[indices]
        y_train = y_train[indices]

    y_train_reordered = numpy.zeros(y_train.shape, dtype=numpy.int)
    y_test_reordered = numpy.zeros(y_test.shape, dtype=numpy.int)
    targets = sorted(list(set(y_train)))
    for i, t in enumerate(targets):
        y_train_reordered[y_train == t] = i
        y_test_reordered[y_test == t] = i

    return Xtrain_norm, Xtest_norm, y_train_reordered, y_test_reordered

parser = argparse.ArgumentParser(description='Run LRS with MLP classifier on a given UCR dataset.')
parser.add_argument('--dataset', dest='dataset', default='ElectricDevices', help='name of the dataset to be used')
parser.add_argument('--index', dest='index', type=int, default=0,
                    help='index of the random shapelet draw to be read (default: 0)')
args = parser.parse_args()
dataset = args.dataset
idx_draw = args.index

X_train, X_test, y_train, y_test = read_shapelets(dataset_name=dataset, index=idx_draw, shuffle=True)

n_classes = to_categorical(y_train).shape[1]
n_shapelets = X_train.shape[1] // 2
indices_sparse = numpy.array([0, 1] * n_shapelets)  # sparsity penalization activated on odd indices (1, 3, 5, ...)
groups = numpy.repeat(numpy.arange(n_shapelets), 2)

print(y_train.shape, y_test.shape)


params = {"alpha": [.2, .5, .8], "lbda": [0., 1e-8, 1e-6, 1e-4]}
clf = KerasClassifier(build_fn=ssgl_multi_layer_perceptron,
                      hidden_layers=(256, 128, 64),
                      n_classes=n_classes,
                      dim_input=n_shapelets * 2,
                      epochs=100,
                      batch_size=256,
                      optimizer_str="rmsprop",
                      optimizer_lr=.001,
                      optimizer_decay=.1,
                      groups=groups,
                      indices_sparse=indices_sparse,
                      activation="relu",
                      batch_normalization=True,
                      verbose=2)
cv = StratifiedKFold(n_splits=5)

cv_model = GridSearchCV(estimator=clf, param_grid=params, cv=cv, n_jobs=1, refit=True)
cv_model.fit(X_train, y_train)

print(cv_model.best_params_)
print(cv_model.score(X_test, y_test))
