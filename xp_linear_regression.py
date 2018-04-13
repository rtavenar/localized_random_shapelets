from models import ssgl_linear_regression

import numpy
import matplotlib.pyplot as plt
import os


def bracket(ax, pos=[0,0], scalex=1, scaley=1, text="",textkw = {}, linekw = {}):
    x = numpy.array([0, 0.05, 0.45,0.5])
    y = numpy.array([0,-0.01,-0.01,-0.02])
    x = numpy.concatenate((x,x+0.5))
    y = numpy.concatenate((y,y[::-1]))
    ax.plot(x*scalex+pos[0], y*scaley+pos[1], clip_on=False,
            transform=ax.get_xaxis_transform(), **linekw)
    ax.text(0.5*scalex+pos[0], (y.min()-0.01)*scaley+pos[1], text,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", **textkw)

font = {'family': 'serif', 'size': 20}
plt.rc('text', usetex=True)
plt.rc('font', **font)
plt.rcParams['text.usetex']=True
plt.rcParams['text.latex.unicode']=True

numpy.random.seed(0)

n_groups = 3
n_features_per_group = 2
n = 1000
noise_level = .01

n_features = n_groups * n_features_per_group
groups = numpy.repeat(numpy.arange(n_groups), n_features_per_group)
ind_sparse = numpy.zeros((n_features, ))
ind_sparse[2] = 0
ind_sparse[4] = 0

beta_star = numpy.array([0., 0., 1.5, 2., 0.005, 0.])

X = numpy.random.randn(n, n_features)
y = numpy.dot(X, beta_star) + noise_level * numpy.random.rand(n)

# Models
model_ssgl = ssgl_linear_regression(dim_input=n_features, groups=groups, indices_sparse=ind_sparse, alpha=.5, lbda=.01)
model_ssgl.fit(X, y.reshape((-1, 1)), epochs=200, verbose=2)
beta_hat_ssgl = model_ssgl.get_weights()[0].reshape((-1, ))
mse_ssgl = numpy.linalg.norm(y - model_ssgl.predict(X).reshape((-1, )))

model_lasso = ssgl_linear_regression(dim_input=n_features, groups=None, indices_sparse=numpy.array([1] * n_features), alpha=1., lbda=.01)
model_lasso.fit(X, y.reshape((-1, 1)), epochs=200, verbose=2)
beta_hat_lasso = model_lasso.get_weights()[0].reshape((-1, ))
mse_lasso = numpy.linalg.norm(y - model_lasso.predict(X).reshape((-1, )))

model_sgl = ssgl_linear_regression(dim_input=n_features, groups=groups, indices_sparse=numpy.array([1] * n_features), alpha=.5, lbda=.01)
model_sgl.fit(X, y.reshape((-1, 1)), epochs=200, verbose=2)
beta_hat_sgl = model_sgl.get_weights()[0].reshape((-1, ))
mse_sgl = numpy.linalg.norm(y - model_sgl.predict(X).reshape((-1, )))

labels = []
for group_id in range(3):
    for feature_id in range(n_features_per_group):
        labels.append("$\\beta_%d^{(%d)}$" % (feature_id + 1, group_id + 1))

beta_star = numpy.concatenate((beta_star[:2], beta_star[4:], beta_star[2:4]))
beta_hat_ssgl = numpy.concatenate((beta_hat_ssgl[:2], beta_hat_ssgl[4:], beta_hat_ssgl[2:4]))
beta_hat_lasso = numpy.concatenate((beta_hat_lasso[:2], beta_hat_lasso[4:], beta_hat_lasso[2:4]))
beta_hat_sgl = numpy.concatenate((beta_hat_sgl[:2], beta_hat_sgl[4:], beta_hat_sgl[2:4]))

plt.figure(figsize=(12, 4))
ind = numpy.arange(n_features)
width = 0.2
plt.bar(ind, numpy.abs(beta_star), width, label="Ground-truth", lw=2, fc=(0, 0, 0, .3), edgecolor='k')
plt.bar(ind+width, numpy.abs(beta_hat_ssgl), width, label="SSGL (MSE=%.2f)" % mse_ssgl, lw=2, fc=(0, 0, 1, .3), edgecolor='b')
plt.bar(ind+2*width, numpy.abs(beta_hat_lasso), width, label="Lasso (MSE=%.2f)" % mse_lasso, lw=2, fc=(0, 1, 0, .3), edgecolor='g')
plt.bar(ind+3*width, numpy.abs(beta_hat_sgl), width, label="SGL (MSE=%.2f)" % mse_sgl, lw=2, fc=(1, 0, 0, .3), edgecolor='r')
plt.gca().set_yscale("log")

bracket(plt.gca(), text="Group 1", pos=[0.05,-.2], scalex=1.5, scaley=2, linekw=dict(lw=1, color="k"))
bracket(plt.gca(), text="Group 2", pos=[2.05,-.2], scalex=1.5, scaley=2, linekw=dict(lw=1, color="k"))
bracket(plt.gca(), text="Group 3", pos=[4.05,-.2], scalex=1.5, scaley=2, linekw=dict(lw=1, color="k"))

plt.legend(loc="upper left")
plt.ylabel("Beta values")
plt.gca().set_xticklabels(labels)
plt.gca().set_xticks(numpy.arange(6) + .3)
plt.tight_layout(pad=2.)
plt.savefig("synth_reg.pdf")
