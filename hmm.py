from jax import numpy as jnp, random as jrand, vmap, jit
from jax.scipy.linalg import expm
from jax.nn import relu, sigmoid, tanh
# from torch.utils.data import Dataset
import numpy as np  # needed because jax's qr function does not work

from typing import Callable
from functools import wraps, partial

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


# generate operator dictionary mixing parameters according to uniform
def uniform_tms_generator(n: int, n_ops: int, minval: float, maxval: float):
    def inner(key):
        return jrand.uniform(key=key, maxval=maxval, minval=minval, shape=(n, n_ops))
    return inner


# generate feature matrices according to normal
def normal_feats_generator(ld: int, fd: int, mean: float, scale: float):
    def inner(key):
        return jrand.normal(key=key, shape=(fd, ld)) * scale + mean
    return inner


# mix operator dictionary elements and exponentiate them
def get_transforms(tms: jnp.ndarray, op_dict: jnp.ndarray) -> jnp.ndarray:
    gens = jnp.einsum('ij,jkl->ikl', tms, op_dict)
    return batch_expm(gens)


def hgmm_modular(seed_point: jnp.ndarray,
                 op_dict: jnp.ndarray,
                 nobs: int = 1000,
                 tms_generator: Callable = jrand.uniform,
                 feats_generator: Callable = jrand.normal,
                 nonlinearity: Callable = jnp.square,
                 *, key: jrand.PRNGKey):
    # some values that may be interesting
    ld = op_dict.shape[-1]
    # generate some keys
    tmkey, feats_key = jrand.split(key)
    # we now generate the actual latent points
    tms = tms_generator(key=tmkey)
    transforms = get_transforms(tms, op_dict)
    z = transforms @ seed_point
    # Now we generate the gaussian features
    F = feats_generator(key=feats_key)
    # and lets go with the actual points in high dim space!
    points = nonlinearity((z @ F.T) / jnp.sqrt(ld))
    return points


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


# class HGMMDataset(Dataset):
#     """Pytorch-style dataset for hidden group manifold data."""

#     def __init__(self, dataset: jnp.ndarray):
#         self.dset = dataset

#     def __len__(self): return self.dset.shape[0]

#     def __getitem__(self, index): return self.dset[index]


def get_controllable_transport_operators(key, n_ops, ld, minval=-1, maxval=1):
    op_key, eig_key = jrand.split(key)
    get_qs = jit(vmap(lambda x: jnp.linalg.qr(x, mode='complete')[0]))
    ops_base = jrand.normal(key=op_key, shape=(n_ops, ld, ld))
    op_dict_mixers = get_qs(ops_base)
    op_dict_eigvals = jrand.uniform(key=eig_key, shape=(n_ops, ld), minval=minval, maxval=maxval)
    return jnp.einsum('...ij,...j->...ij', op_dict_mixers, op_dict_eigvals)


if __name__ == "__main__":
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    minval, maxval = -0.5, .5
    n_pts = 1000
    n_ops = 2
    ld = 10
    fd = 100
    key = jrand.PRNGKey(0)
    seed_key, op_key, hgmm_key, eta_key = jrand.split(key, num=4)

    # create a parameterized feature generator
    f_gen = normal_feats_generator(ld=ld, fd=fd, mean=0, scale=jnp.sqrt(ld))
    # create a parameterized dict coefficients generator
    t_gen = uniform_tms_generator(n=n_pts, n_ops=n_ops, minval=minval, maxval=maxval)
    # create a seed
    seed = jrand.normal(seed_key, shape=(ld,))
    # create a dictionary of transport operators
    # Idea: maybe doing something at the spectral level? So drawing the
    # eigenvalues from a uniform distribution and then rotate such a thing
    # according to a random orthogonal matrix?
    op_dict = get_controllable_transport_operators(key=op_key, n_ops=n_ops, ld=ld)
    op_dict -= jnp.transpose(op_dict, axes=(0, 2, 1))  # symmetrize the ops
    op_dict /= 2
    # put all into a partial which only depends on the seed
    hgmm_params = dict(op_dict=op_dict, nobs=n_pts, tms_generator=t_gen, feats_generator=f_gen, key=hgmm_key)
    get_points_from_seed = partial(hgmm_modular, **hgmm_params)
    # and now all together
    deltas = []
    points1 = get_points_from_seed(seed_point=seed)
    # eta = jrand.normal(key=eta_key, shape=(ld,))*1e-6
    # for n in range(10):
    #     new_seed = seed + n*eta
    #     points_new = get_points_from_seed(seed_point=new_seed)
    #     deltas.append(((points1-points_new)**2).sum(axis=1).mean())
    # plt.plot(deltas); plt.show()

    # %%
    # import plotly.express as px
    # import pandas as pd
    # key = jrand.PRNGKey(0)
    # points, labels = HGMM(nobs=1000, ld=2, dd=3, nops=2, init_delta=5e0, time_mult=10, key=key)
    # pts = pd.DataFrame(points, columns=['x', 'y', 'z'])
    # pts['labels'] = labels
    # fig2 = px.scatter_3d(pts, x='x', y='y', z='z', color='labels')
    # fig2.write_html("interactive_hgmm_SO2_tanh.html")
    # fig2.show()
    PCA = PCA(n_components=2)
    pca_points = points1[~jnp.isnan(points1).any(axis=1)]
    lowpoints = PCA.fit_transform(pca_points)
    fig, ax = plt.subplots()
    ax.scatter(lowpoints[:, 0], lowpoints[:, 1])
    plt.show()
