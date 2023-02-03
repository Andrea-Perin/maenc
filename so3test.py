from jax import numpy as np, random as jrand, jit, vmap, scipy as sp, grad
from optax import adam, apply_updates
from functools import partial
import matplotlib.pyplot as plt


PRECISION = 1e-4  # controls num precision when drawing from sphere
N = 100
key = jrand.PRNGKey(0)


# generate random numbers uniformly on N-dim sphere
def n_spherical(
    key: jrand.KeyArray,
    size: int,
    n: int,
) -> np.ndarray:
    nkey = key
    total = 0
    blocks = []
    while total < size:
        key, nkey = jrand.split(nkey)
        new_block = jrand.normal(key, shape=(size, n))
        norms = np.linalg.norm(new_block, axis=1, keepdims=True)
        new_block /= norms
        mask = (norms > PRECISION).flatten()
        blocks.append(new_block[mask])
        total += mask.sum().astype(int)
    return (np.stack(blocks)[0])[:size]


three_sphere = partial(n_spherical, n=3)


def ridge_solve(X, y, alpha: float = 1):
    return (np.linalg.inv(X.T@X + alpha*np.eye(X.shape[1])) @ (X.T@y)).T


# %% TRUE PARAMS + DATASET CREATION
# generate the true dataset and the true projection function
zkey, fkey, key = jrand.split(key, num=3)
# generate the latent points on a 2-sphere in 3D space
Z_true = three_sphere(key=zkey, size=N)
# generate the projection matrix according to a mean and covariance
F_true = jrand.normal(key=fkey, shape=(2, 3))
X = Z_true @ F_true.T

# %% RANDOM PROBLEM INIT
# we now want to jointly optimize the approximate function F_hat and the latent
# points Z_hat. These are, at first, drawn randomly
zhatkey, key = jrand.split(key)
Z_hat = three_sphere(key=zhatkey, size=N)
Z_hat_init = Z_hat

# %% DEFINE FUNCTIONS
# We start by solving the ridge regression problem of F with respect to the
# randomly drawn Z_hat
ridge = partial(ridge_solve, alpha=.1, y=X)
def likelihood(Z): return (ridge(Z)**2).sum()
def reconstruction(Z): return ((Z@ridge(Z).T - X)**2).sum()
def manifold(Z): return (np.square(np.linalg.norm(Z, axis=1) - 1)).sum()


def loss(Z):
    return likelihood(Z) + 1*reconstruction(Z) + 1*manifold(Z)


# %% """TRAINING""" LOOP
optimizer = adam(learning_rate=1)
opt_state = optimizer.init(Z_hat)
f_hat_progress = []
Z_hat_progress = []
loss_progress = []
for i in range(100):
    grads = grad(loss)(Z_hat)  # compute gradients wrt likelihood
    updates, opt_state = optimizer.update(grads, opt_state)
    Z_hat = apply_updates(Z_hat, updates)
    # normalize on sphere
    Z_hat /= np.linalg.norm(Z_hat, axis=1, keepdims=True)
    # avg distances from true values
    f_hat_progress.append(((ridge(Z_hat)-F_true)**2).mean())
    Z_hat_progress.append(((Z_hat - Z_true)**2).mean())
    loss_progress.append(((Z_hat@ridge(Z_hat).T - X)**2).mean())


# %% VIZ
fig = plt.figure(figsize=plt.figaspect(.33))
# 2D perf tracking
ax1 = fig.add_subplot(1, 3, 1)
ax1.plot(f_hat_progress, label='f_hat avg error')
ax1.plot(Z_hat_progress, label='Z_hat avg error')
ax1.plot(loss_progress, label='loss')
ax1.legend()
# 3D data viz
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.scatter(*Z_hat_init.T, marker='o', label='initial random Z')
ax2.scatter(*Z_hat.T, marker='o', label='final Z')
ax2.scatter(*Z_true.T, marker='x', label='true Z')
ax2.quiver(*Z_hat_init.T, *(Z_hat - Z_hat_init).T,
           color='orange', label="Z_hat actual progress")
# ax2.quiver(*Z_hat_init.T, *(Z_true - Z_hat_init).T, color='purple', label="Z_hat ideal progress")
ax2.legend()
# 2D data viz
ax3 = fig.add_subplot(1, 3, 3)
ax3.scatter(*X.T, marker='o', label='X true')
ax3.scatter(*(Z_hat@(ridge(Z_hat)).T).T, marker='o', label='X final')
ax3.legend()
plt.show()
