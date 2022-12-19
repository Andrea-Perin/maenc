from jax import numpy as jnp, random as jrand, vmap, jit
from jax.scipy.linalg import expm
from jax.nn import relu, sigmoid, tanh
from torch.utils.data import Dataset

batch_expm = jit(vmap(expm))


def get_SO2():
    return jnp.array([[[0, -1], [1, 0]], [[0, -1], [1, 0]]])


def get_SO3():
    return jnp.array([[[0, 0, 0],
                       [0, 0, -1],
                       [0, 1, 0]],
                      [[0, 0, 1],
                       [0, 0, 0],
                       [-1, 0, 0]],
                      [[0, -1, 0],
                       [1, 0, 0],
                       [0, 0, 0]]])


def hgmm(nobs: int, dd: int, ops: jnp.ndarray, init_delta: float = 1e-3,
         time_mult: float = 10, *, key: jrand.PRNGKey):
    z0key, tmkey, feats_key, perm_key = jrand.split(key, num=4)
    nops, ld = ops.shape[:2]
    z0 = jrand.normal(z0key, shape=(ld,)) * init_delta + 1
    # and then we pick the transfer parameters
    tms = jrand.uniform(tmkey, shape=(nobs, nops), minval=-time_mult, maxval=time_mult)
    # we now generate the actual latent points
    gens = jnp.einsum('ij,jkl->ikl', tms, ops)
    maps = batch_expm(gens)
    z = maps @ z0
    # Now we generate the gaussian features
    F = jrand.normal(feats_key, shape=(dd, ld)) / ld
    # and lets go with the actual points in high dim space!
    points = tanh((z @ F.T) / jnp.sqrt(ld))
    return points


def HGMM(nobs: int, ld: int, dd: int, nops: int,
         init_delta: float = 1e-3, time_mult: float = 10,
         *, key: jrand.PRNGKey):
    # instantiate the transport operators dictionary elements.
    # These are ld x ld matrices that will be later exponentiated
    # It is possible to prove that the exponential converges for any
    # matrix. We can then pick the operators from some random distribution.
    transfer_key, z0key, tmkey, feats_key, perm_key = jrand.split(key, num=5)
    transfer_dict = jrand.normal(transfer_key, shape=(nops, ld, ld)) / ld
    # SO2 + other stuff
    transfer_dict = jnp.array([[[0, -1], [1, 0]], [[0, -1], [1, 0]]])
    # SO3
    # transfer_dict = jnp.array([[[0, 0, 0],
    #                             [0, 0, -1],
    #                             [0, 1, 0]],
    #                            [[0, 0, 1],
    #                             [0, 0, 0],
    #                             [-1, 0, 0]],
    #                            [[0, -1, 0],
    #                             [1, 0, 0],
    #                             [0, 0, 0]]])
    # we now pick two ld-dimensional points
    z0a, z0b = jrand.normal(z0key, shape=(2, ld)) * init_delta + 1
    # and then we pick the transfer parameters
    tms_a, tms_b = jrand.uniform(tmkey, shape=(2, nobs, nops),
                                 minval=-time_mult, maxval=time_mult)
    # tms_a, tms_b = jrand.normal(tmkey, shape=(2, nobs, nops)) * time_mult
    # we now generate the actual latent points
    gens_a = jnp.einsum('ij,jkl->ikl', tms_a, transfer_dict)
    gens_b = jnp.einsum('ij,jkl->ikl', tms_b, transfer_dict)
    maps_a = batch_expm(gens_a)
    maps_b = batch_expm(gens_b)
    za = maps_a @ z0a
    zb = maps_b @ z0b
    # Now we generate the gaussian features
    Fa, Fb = jrand.normal(feats_key, shape=(2, dd, ld)) / ld
    # and lets go with the actual points in high dim space!
    points_a = tanh((za @ Fa.T) / jnp.sqrt(ld))
    points_b = tanh((zb @ Fb.T) / jnp.sqrt(ld))
    points = jnp.concatenate((points_a, points_b))
    # the labels are just the a/b classes
    labels = jnp.concatenate((-jnp.ones(nobs), jnp.ones(nobs)))
    # do some permutation real quick
    points = jrand.permutation(perm_key, points)
    labels = jrand.permutation(perm_key, labels)
    return points, labels


class HGMMDataset(Dataset):
    """Pytorch-style dataset for hidden group manifold data."""

    def __init__(self, dataset: jnp.ndarray):
        self.dset = dataset

    def __len__(self): return self.dset.shape[0]

    def __getitem__(self, index): return self.dset[index]


if __name__ == "__main__":
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import plotly.express as px
    import pandas as pd
    key = jrand.PRNGKey(0)
    points, labels = HGMM(nobs=1000, ld=2, dd=3, nops=2, init_delta=5e0, time_mult=10, key=key)
    pts = pd.DataFrame(points, columns=['x', 'y', 'z'])
    pts['labels'] = labels
    fig2 = px.scatter_3d(pts, x='x', y='y', z='z', color='labels')
    fig2.write_html("interactive_hgmm_SO2_tanh.html")
    fig2.show()
    PCA = PCA(n_components=2)
    pca_points = points[~jnp.isnan(points).any(axis=1)]
    lowpoints = PCA.fit_transform(pca_points)
    fig, ax = plt.subplots()
    ax.scatter(lowpoints[:, 0], lowpoints[:, 1])
    plt.show()
