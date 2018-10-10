"""
Inspired from a gallery example in tslearn:
<https://tslearn.readthedocs.io/en/latest/auto_examples/plot_shapelet_locations.html>
"""
import numpy
import matplotlib.pyplot as plt
from keras.optimizers import Adagrad
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.datasets import UCR_UEA_datasets
from tslearn.shapelets import ShapeletModel

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

seed = 0
numpy.random.seed(seed)
ds_name = "EarthQuakes"
X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(ds_name)
X_train = TimeSeriesScalerMinMax().fit_transform(X_train)
X_test = TimeSeriesScalerMinMax().fit_transform(X_test)
n_ts, ts_sz = X_train.shape[:2]
n_classes = len(set(y_train))
n_shapelets = 5
sz_shapelets = int(0.1 * ts_sz)
shapelet_sizes = {sz_shapelets: n_shapelets}

test_ts_id = 0
yrange = [-1, 2]
# LS figure
shp_clf = ShapeletModel(n_shapelets_per_size=shapelet_sizes,
                        optimizer=Adagrad(lr=.1),
                        weight_regularizer=.01,
                        max_iter=50,
                        verbose_level=0,
                        random_state=seed)
shp_clf.fit(X_train, y_train)
predicted_locations = shp_clf.locate(X_test)

plt.figure(figsize=(10, 4))
plt.plot(X_test[test_ts_id].ravel())
for idx_shp, shp in enumerate(shp_clf.shapelets_):
    t0 = predicted_locations[test_ts_id, idx_shp]
    plt.plot(numpy.arange(t0, t0 + len(shp)), shp, linewidth=2)
plt.xticks([])
plt.yticks([0., 1.])
plt.ylim(yrange)
plt.tight_layout()
plt.savefig("shapelets_LS.pdf")

# LRS figure
drawn_shapelets = numpy.empty((n_shapelets, sz_shapelets))
for i in range(n_shapelets):
    idx = numpy.random.randint(n_ts)
    pos = numpy.random.randint(ts_sz-sz_shapelets+1)
    drawn_shapelets[i] = X_train[idx, pos:pos+sz_shapelets, 0]
shp_clf.model.get_layer("shapelets_0_0").set_weights([drawn_shapelets])
predicted_locations = shp_clf.locate(X_test)

plt.figure(figsize=(10, 4))
plt.plot(X_test[test_ts_id].ravel())
for idx_shp, shp in enumerate(shp_clf.shapelets_):
    t0 = predicted_locations[test_ts_id, idx_shp]
    plt.plot(numpy.arange(t0, t0 + len(shp)), shp, linewidth=2)
plt.xticks([])
plt.yticks([0., 1.])
plt.ylim(yrange)
plt.tight_layout()
plt.savefig("shapelets_LRS.pdf")
