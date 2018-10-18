import numpy

def compute_shapeletscoeff(first_layer_weights):
    """
    Method to compute the coefficient of each shapelet (group coeff, shapelet distance coeff and shapelet localization coeff)

    Parameters
    ----------
    first_layer_weights : matrix of float, shape (n_features, n_units)
        weights from the first layer of mlp classifier, /!\ n_features = n_shapelets*2 /!\

    Returns
    -------
    numpy array, shape (n_features / 2, 3)
        numpy array of indexed shapelets (0 to n_shapelets - 1) with their group coefficient,
        shapelet localization coefficient and shapelet distance coefficient
    """
    if first_layer_weights.ndim ==1:
        flw = first_layer_weights.reshape(-1,1)
    else:
        flw = first_layer_weights
    n_features, n_hidden_units = flw.shape
    n_shapelets = n_features // 2

    submat_localization = flw[::2]
    submat_distances = flw[1::2]
    reshapedmat_full = flw.reshape((n_shapelets, n_hidden_units * 2))

    shapelets_coeff = numpy.empty((n_shapelets, 3))
    shapelets_coeff[:, 0] = numpy.linalg.norm(reshapedmat_full, ord=2, axis=1) / (2 * n_hidden_units)
    shapelets_coeff[:, 1] = numpy.linalg.norm(submat_localization, ord=2, axis=1) / n_hidden_units
    shapelets_coeff[:, 2] = numpy.linalg.norm(submat_distances, ord=2, axis=1) / n_hidden_units
    return shapelets_coeff


