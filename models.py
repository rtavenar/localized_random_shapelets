from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.regularizers import Regularizer
from keras import backend as K
import numpy
import tensorflow as tf


def ssgl_logistic_regression(n_classes, dim_input, groups=None, indices_sparse=None, alpha=0.5, lbda=0.01,
                             optimizer_str="sgd", optimizer_lr=0.01, optimizer_decay=0., dropout=0.):
    regularizer = SSGL_WeightRegularizer(l1_reg=alpha * lbda, l2_reg=(1. - alpha) * lbda,
                                         indices_sparse=indices_sparse, groups=groups)
    model = Sequential()
    if dropout > 0.:
        model.add(Dropout(input_shape=(dim_input,), rate=dropout))
    if n_classes == 2:
        model.add(Dense(units=1, input_dim=dim_input, activation="sigmoid", kernel_regularizer=regularizer))
        model.compile(loss="binary_crossentropy", optimizer=optimizer_str, metrics=["accuracy"])
    else:
        model.add(Dense(units=n_classes, input_dim=dim_input, activation="softmax",
                              kernel_regularizer=regularizer))
        model.compile(loss="categorical_crossentropy", optimizer=optimizer_str, metrics=["accuracy"])
    K.set_value(model.optimizer.lr, optimizer_lr)
    K.set_value(model.optimizer.decay, optimizer_decay)
    return model


def ssgl_linear_regression(dim_input, groups=None, indices_sparse=None, alpha=0.5, lbda=0.01,
                           optimizer_str="sgd", optimizer_lr=0.01, optimizer_decay=0., dropout=0.):
    regularizer = SSGL_WeightRegularizer(l1_reg=alpha * lbda, l2_reg=(1. - alpha) * lbda,
                                         indices_sparse=indices_sparse, groups=groups)
    model = Sequential()
    if dropout > 0.:
        model.add(Dropout(input_shape=(dim_input,), rate=dropout))
    model.add(Dense(units=1, input_dim=dim_input, activation="linear", kernel_regularizer=regularizer))
    model.compile(loss="mse", optimizer=optimizer_str)
    K.set_value(model.optimizer.lr, optimizer_lr)
    K.set_value(model.optimizer.decay, optimizer_decay)
    return model


def ssgl_multi_layer_perceptron(n_classes, dim_input, hidden_layers, groups=None, indices_sparse=None, alpha=0.5,
                                lbda=0.01, activation="relu", optimizer_str="sgd", optimizer_lr=0.01,
                                optimizer_decay=0., dropout=0., batch_normalization=False):
    K.clear_session()
    if hidden_layers is None:
        return ssgl_logistic_regression(n_classes=n_classes, dim_input=dim_input, groups=groups,
                                        indices_sparse=indices_sparse, alpha=alpha, lbda=lbda,
                                        optimizer_str=optimizer_str, optimizer_lr=optimizer_lr, dropout=dropout)
    else:
        hidden_layers = list(hidden_layers)
    regularizer = SSGL_WeightRegularizer(l1_reg=alpha * lbda, l2_reg=(1. - alpha) * lbda, indices_sparse=indices_sparse,
                                         groups=groups)
    model = Sequential()
    if dropout > 0.:
        model.add(Dropout(input_shape=(dim_input,), rate=dropout))
    model.add(Dense(units=hidden_layers[0], input_dim=dim_input, activation=activation, kernel_regularizer=regularizer))
    if batch_normalization:
        model.add(BatchNormalization())
    for n_units in hidden_layers[1:]:
        if dropout > 0.:
            model.add(Dropout(rate=dropout))
        model.add(Dense(units=n_units, activation=activation))
        if batch_normalization:
            model.add(BatchNormalization())
    if dropout > 0.:
        model.add(Dropout(rate=dropout))
    if batch_normalization:
        model.add(BatchNormalization())
    if n_classes == 2:
        model.add(Dense(units=1, activation="sigmoid"))
        model.compile(loss="binary_crossentropy", optimizer=optimizer_str, metrics=["accuracy"])
    else:
        model.add(Dense(units=n_classes, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer=optimizer_str, metrics=["accuracy"])
    K.set_value(model.optimizer.lr, optimizer_lr)
    K.set_value(model.optimizer.decay, optimizer_decay)
    return model


class SSGL_WeightRegularizer(Regularizer):
    """Semi-Sparse Group Lasso weight regularizer.

    Parameters
    ----------
    l1_reg : float, default 0.
        Per-dimension sparsity penalty parameter.
    l2_reg : float, default 0.
        Group sparsity penalty parameter.
    groups : list of numpy arrays or None, default None.
        List of groups. Each group is defined by a numpy array of shape `(dim_input, )` in which a zero value means
        the corresponding input dimension is not included in the group and a one value means the corresponding input
        dimension is part of the group. None means no group sparsity penalty
        groups numbering must starts at 0 with a continuous increment of 1 ([0,1,2,3...]). Features of the same group must be contiguous.
    indices_sparse : array-like or None, default None.
        numpy array of shape `(dim_input, )` in which a zero value means the corresponding input dimension should not
        be included in the per-dimension sparsity penalty and a one value means the corresponding input dimension should
        be included in the per-dimension sparsity penalty. None means no per-dimension sparsity penalty.
    """
    def __init__(self, l1_reg=0., l2_reg=0., groups=None, indices_sparse=None):
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        if groups is None:
            self.groups = None
        else:
            groups = numpy.array(groups).astype('int32')
            self.p_l = K.variable(numpy.sqrt(numpy.bincount(groups)).reshape((1, -1)))
            self.groups = K.variable(groups, 'int32')
        if indices_sparse is not None:
            self.indices_sparse = K.variable(indices_sparse.reshape((1, -1)))
        else:
            self.indices_sparse = None

    def __call__(self, x):
        loss = 0.
        if self.indices_sparse is not None:
            loss += K.sum(K.dot(self.indices_sparse, K.abs(x))) * self.l1_reg
        if self.groups is not None:
            loss += K.sum(K.dot(self.p_l, K.sqrt(tf.segment_sum(K.square(x), self.groups)))) * self.l2_reg
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__, "l1_reg": self.l1_reg, "l2_reg": self.l2_reg}
