from jax import numpy as np, random as jrand, jit, vmap, scipy as sp, grad
from optax import adam, apply_updates, cosine_decay_schedule
from functools import partial
import matplotlib.pyplot as plt


key = jrand.PRNGKey(0)  # good inversion seed: 124
PRECISION = 1e-4  # controls num precision when drawing from sphere
N_POINTS = 200  # number of generated points (if N_CLASSES>1, per class)
N_CLASSES = 2
Z_DIM = 3  # latent space dimension
X_DIM = 2  # observable space dimension
N_STEPS = 100


# %% RSGD ON SPHERE
def sphere_tangent_projector(x: np.ndarray, n: int) -> np.ndarray:
    # takes an array of points in ambient space of dim n+1 and projects them on
    # the tangent space of a sphere of dimension n
    x = x / np.linalg.norm(x)
    return np.eye(n) - x[:, None]*x


def project_on_tangent_space(theta: np.ndarray, dtheta: np.ndarray, n: int) -> np.ndarray:
    projector = partial(sphere_tangent_projector, n=n+1)
    return projector(theta) @ dtheta


def n_sphere_exponential(t: float, p: np.ndarray, v: np.ndarray) -> np.ndarray:
    # take the vector from the tangent space and place it on the sphere
    nv = np.linalg.norm(v)
    pt = np.cos(nv*t)*p + np.sin(nv*t)*(v / nv)
    return pt / np.linalg.norm(pt)


def get_new_points_n_sphere(eta: float, theta: np.ndarray, dtheta: np.ndarray, n: int) -> np.ndarray:
    dtheta_on_tangent_space = project_on_tangent_space(theta, dtheta, n)
    return n_sphere_exponential(eta, theta, dtheta_on_tangent_space)


# %% UTILITY FUNCTIONS
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


Z_sphere = partial(n_spherical, n=Z_DIM, size=N_POINTS)


def ridge_solve(X, y, alpha: float = 1):
    return (np.linalg.inv(X.T@X + alpha*np.eye(X.shape[1])) @ (X.T@y)).T


def plot_3D_points(ax: plt.Axes, Z_hat: np.ndarray, Z_true: np.ndarray, n: int = N_POINTS):
    # plot all the clouds separately (same color, different marker)
    if Z_hat.shape[0] != Z_true.shape[0]:
        raise ValueError("Wrong dimensions between true and reconstructed")
    # manual legend
    import matplotlib.lines as mlines
    import matplotlib.cm as cm
    z_true = mlines.Line2D([], [], color='black', marker='o',
                           linestyle='None', label="Z_true")
    z_hat = mlines.Line2D([], [], color='black', marker='x',
                          linestyle='None', label="Z_hat")
    # manual color choosing
    n_classes = Z_hat.shape[0] // n
    cmap = cm.get_cmap('tab10')
    # actual plotting
    for iii in range(n_classes):
        c = cmap(iii)
        Z_hat_iii = Z_hat[iii: n*(iii+1)]
        Z_true_iii = Z_true[iii: n*(iii+1)]
        ax.scatter(*Z_hat_iii.T, marker='x', color=c)
        ax.scatter(*Z_true_iii.T, marker='o', color=c)
    ax.legend(handles=[z_true, z_hat])


def plot_2D_scatter(ax: plt.Axes, X_true: np.ndarray, X_hat: np.ndarray, n: int = N_POINTS):
    # plot all the clouds separately (same color, different marker)
    # manual legend
    import matplotlib.lines as mlines
    import matplotlib.cm as cm
    x_true = mlines.Line2D([], [], color='black', marker='o',
                           linestyle='None', label="X_true")
    x_hat = mlines.Line2D([], [], color='black', marker='x',
                          linestyle='None', label="X_hat")
    # manual color choosing
    n_classes = X_true.shape[0] // n
    cmap = cm.get_cmap('tab10')
    # actual plotting
    for iii in range(n_classes):
        c = cmap(iii)
        X_hat_iii = X_hat[iii: n*(iii+1)]
        X_true_iii = X_true[iii: n*(iii+1)]
        ax.scatter(*X_hat_iii.T, marker='x', color=c, alpha=(.8-iii/5)*0.25)
        ax.scatter(*X_true_iii.T, marker='o', color=c, alpha=(.8-iii/5)*0.25)
    ax.legend(handles=[x_true, x_hat])


def create_flow_gif(Zs: np.ndarray):

    def get_plane_pts(Z):
        basis_pts = np.stack(np.meshgrid(np.linspace(-1, 1), np.linspace(-1, 1))).T.reshape(-1, 2)
        F = ridge(Z)
        return basis_pts @ (F / np.linalg.norm(F, axis=1, keepdims=True))

    import matplotlib.animation as anim
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    graph = ax.scatter(*Zs[0].T, marker='o', alpha=0.5)
    plane_pts = ax.scatter(*get_plane_pts(Zs[0]).T, c='grey', alpha=0.01, marker='x')
    title = ax.set_title("Iteration: 0")

    def update_graph(num):
        graph._offsets3d = tuple(Zs[num].T)
        plane_pts._offsets3d = tuple(get_plane_pts(Zs[num]).T)
        title.set_text(f"Iteration: {num}")

    ani = anim.FuncAnimation(fig, update_graph, frames=len(Zs), interval=50)
    return ani


# %% TRUE PARAMS + DATASET CREATION
# generate the true dataset and the true projection function
zkey, fkey, key = jrand.split(key, num=3)
zkeys = jrand.split(zkey, num=N_CLASSES)
fkeys = jrand.split(fkey, num=N_CLASSES)
# generate the latent points on a 2-sphere in 3D space
Z_true_clouds = np.array([Z_sphere(key=k) for k in zkeys])
Z_true_radii = (np.arange(N_CLASSES) + 1)[:, None, None]
Z_true_clouds *= Z_true_radii
# generate the projection matrix according to a standard distribution
F_true = jrand.normal(key=fkey, shape=(X_DIM, Z_DIM))
# project to observation space
X = np.concatenate(Z_true_clouds @ F_true.T)
Y = np.repeat(np.arange(N_CLASSES), N_POINTS)
Z_true_clouds = np.concatenate(Z_true_clouds)


# %% RANDOM PROBLEM INIT
# we now want to jointly optimize the approximate function F_hat and the latent
# points Z_hat. These are, at first, drawn randomly
zhatkey, radii_key, key = jrand.split(key, num=3)
zhatkeys = jrand.split(zhatkey, num=N_CLASSES)
Z_hat_clouds = np.concatenate([Z_sphere(key=k) for k in zhatkeys])
Z_hat_radii = jrand.permutation(
    key=radii_key, x=np.arange(N_CLASSES)+10, independent=True)
Z_hat_radii = np.repeat(Z_hat_radii, N_POINTS)[:, None]
Z_hat_clouds *= Z_hat_radii
Z_hat_init = Z_hat_clouds


# %% DEFINE FUNCTIONS
# We start by solving the ridge regression problem of F with respect to the
# randomly drawn Z_hat
ridge = partial(ridge_solve, alpha=0, y=X)
def likelihood(Z): return (ridge(Z)**2).sum()
def reconstruction(Z): return ((Z@ridge(Z).T - X)**2).sum()
def manifold(Z): return (np.square(np.linalg.norm(Z, axis=1) - 1)).sum()


def loss(Z):
    return likelihood(Z) + 1*reconstruction(Z)  # + 1*manifold(Z)


# %% RIEMANN SGD LOOP
sphere_update = partial(get_new_points_n_sphere, n=2)
sphere_update = jit(vmap(sphere_update, in_axes=(None, 0, 0)))
scheduler = cosine_decay_schedule(init_value=0.05, decay_steps=N_STEPS)
loss_progress = []
Z_progress = []
for i in range(N_STEPS):
    eta = scheduler(i)
    grads = grad(loss)(Z_hat_clouds)  # compute gradients wrt likelihood
    Z_hat_clouds = sphere_update(eta, Z_hat_clouds, -grads)
    Z_progress.append(Z_hat_clouds)
    loss_progress.append(((Z_hat_clouds@ridge(Z_hat_clouds).T - X)**2).mean())
X_hat = Z_hat_clouds@ridge(Z_hat_clouds).T


# %% """TRAINING""" LOOP
# optimizer = adam(learning_rate=1)
# opt_state = optimizer.init(Z_hat_clouds)
# f_hat_progress = []
# Z_hat_progress = []
# loss_progress = []
# for i in range(100):
#     grads = grad(loss)(Z_hat_clouds)  # compute gradients wrt likelihood
#     updates, opt_state = optimizer.update(grads, opt_state)
#     Z_hat_clouds = apply_updates(Z_hat_clouds, updates)
#     # normalize on sphere
#     Z_hat_clouds /= np.linalg.norm(Z_hat_clouds, axis=1, keepdims=True)
#     Z_hat_clouds *= Z_hat_radii
#     # avg distances from true values
#     f_hat_progress.append(((ridge(Z_hat_clouds)-F_true)**2).mean())
#     Z_hat_progress.append(((Z_hat_clouds - Z_true_clouds)**2).mean())
#     loss_progress.append(((Z_hat_clouds@ridge(Z_hat_clouds).T - X)**2).mean())

# # compute the reconstructed X
# X_hat = Z_hat_clouds@ridge(Z_hat_clouds).T
# print(Z_hat_radii, Z_true_radii)

# %% VIZ
# fig = plt.figure(figsize=plt.figaspect(.33))
# # 2D perf tracking
# ax1 = fig.add_subplot(1, 3, 1)
# # ax1.plot(f_hat_progress, label='f_hat avg error')
# # ax1.plot(Z_hat_progress, label='Z_hat avg error')
# ax1.plot(loss_progress, label='loss')
# ax1.legend()
# # 3D data viz
# ax2 = fig.add_subplot(1, 3, 2, projection='3d')
# plot_3D_points(ax2, Z_hat_clouds, Z_true_clouds)
# # 2D data viz
# ax3 = fig.add_subplot(1, 3, 3)
# plot_2D_scatter(ax3, X, X_hat)
# plt.show()
ani = create_flow_gif(Z_progress); plt.show()
