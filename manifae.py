import warnings
# warnings.filterwarnings("ignore")

from jax import numpy as jnp, random as jrand, vmap, jit, tree_util as jtu
from jax.scipy.linalg import expm
import equinox as eqx
from hmm import HGMM, HGMMDataset, hgmm, get_SO3, get_SO2
from torch.utils.data import DataLoader
import optax
from functools import partial
from tqdm import tqdm
import more_itertools as mits


class TransportOperator(eqx.Module):
    phis: jnp.ndarray
    cs: jnp.ndarray

    def __init__(self, dim: int, nops: int, *, key):
        phikey, ckey = jrand.split(key)
        self.phis = jrand.normal(key=phikey, shape=(nops, dim, dim))
        self.cs = jrand.normal(key=ckey, shape=(nops,))

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        return expm(jnp.einsum('i,ijk', self.cs, self.phis)) @ z


@eqx.filter_value_and_grad
def E_psi(c, model, z0, z1):
    z1hat = expm(jnp.einsum('i,ijk', c, model.phis)) @ z0
    likelihood = .5 * jnp.sum((z1 - z1hat)**2)
    frobnorm = jnp.sum(jnp.linalg.norm(model.phis, axis=0))
    sparse = jnp.linalg.norm(c, ord=1)
    return likelihood + (model.gamma/2)*frobnorm + model.zeta*sparse


class TransportOperatorAutoFit(eqx.Module):
    phis: jnp.ndarray
    gamma: float = 1e-1
    zeta: float = 1

    @staticmethod
    @eqx.filter_value_and_grad
    def E_psi(c, model, z0, z1):
        z1hat = expm(jnp.einsum('i,ijk', c, model.phis)) @ z0
        likelihood = .5 * jnp.sum((z1 - z1hat)**2)
        frobnorm = jnp.sum(jnp.linalg.norm(model.phis, axis=0))
        sparse = jnp.linalg.norm(c, ord=1)
        return likelihood + (model.gamma/2)*frobnorm + model.zeta*sparse

    def __init__(self, dim: int, nops: int, *, key):
        phikey, key = jrand.split(key)
        self.phis = jrand.normal(key=phikey, shape=(nops, dim, dim))

    @eqx.filter_jit
    def get_cs(self, z0: jnp.ndarray, z1: jnp.ndarray, rand: bool = False) -> jnp.ndarray:
        cs = jnp.zeros(self.phis.shape[0])
        # minimize the energy
        opt = optax.adam(learning_rate=1e-3)
        opt_state = opt.init(cs)
        for i in range(10):
            loss, grads = self.E_psi(cs, self, z0, z1)
            updates, opt_state = opt.update(grads, opt_state, cs)
            cs = eqx.apply_updates(cs, updates)
        return cs

    def __call__(self, z: jnp.ndarray, c: jnp.ndarray):
        return expm(jnp.einsum('i,ijk', c, self.phis)) @ z


class Autoenc(eqx.Module):
    enc: eqx.nn.Linear
    dec: eqx.nn.Linear

    def __init__(self, dim_x: int, dim_z: int, *, key):
        enckey, deckey = jrand.split(key)
        self.enc = eqx.nn.Linear(dim_x, dim_z, use_bias=True, key=enckey)
        self.dec = eqx.nn.Linear(dim_z, dim_x, use_bias=True, key=deckey)

    def __call__(self, input: jnp.ndarray) -> jnp.ndarray:
        return self.dec(self.enc(input))


class ManifoldAutoenc(TransportOperatorAutoFit):
    enc: eqx.nn.Linear
    dec: eqx.nn.Linear
    phis: jnp.ndarray

    def __init__(self, ae: Autoenc, to: TransportOperator):
        self.enc = ae.enc
        self.dec = ae.dec
        self.phis = to.phis


def train_phase_1(model: Autoenc, loader: DataLoader, n_epochs: int = 10, learning_rate: float = 1e-3):
    """Autoencoder training routine."""
    optim = optax.adam(learning_rate=learning_rate)
    opt_state = optim.init(model)

    @eqx.filter_value_and_grad
    def reconstruction_loss(model, x):
        x_hat = vmap(model)(x)
        return jnp.mean((x-x_hat)**2)

    @eqx.filter_jit
    def make_step(model, x, opt_state):
        loss, grads = reconstruction_loss(model, x)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    for epoch in tqdm(range(n_epochs)):
        for batch_idx, batch in enumerate(loader):
            loss, model, opt_state = make_step(model, batch, opt_state)
    return model


def train_phase_2_new(transport: TransportOperatorAutoFit, zpairs: jnp.ndarray, n_epochs: int = 10, learning_rate: float = 1e-3):
    """Transport operator training routine."""
    optim = optax.adam(learning_rate=learning_rate)
    opt_state = optim.init(transport)

    @eqx.filter_jit
    @partial(vmap, in_axes=(None, 0, 0))
    @eqx.filter_value_and_grad
    def E_psi(model, z0, z1):
        c = model.get_cs(z0, z1)
        z1hat = expm(jnp.einsum('i,ijk', c, model.phis)) @ z0
        likelihood = .5 * jnp.sum((z1 - z1hat)**2)
        frobnorm = jnp.sum(jnp.linalg.norm(model.phis, axis=0))
        return likelihood + (model.gamma/2)*frobnorm

    @eqx.filter_jit
    def make_step(model, opt_state, z0, z1):
        loss, grad = E_psi(model, z0, z1)
        updates, opt_state = optim.update(grad, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state

    # loop over the whole dataset n_epochs times and do the training
    for epoch in tqdm(range(n_epochs)):
        # for (z0, z1) in mits.chunked(zpairs, n=128):
        for pair in mits.chunked(zpairs, n=128):
            print(type(pair))
            transport, opt_state = make_step(transport, opt_state, z0, z1)
    return transport


def train_phase_3_new(manifautoenc: ManifoldAutoenc, loader: jnp.ndarray, n_epochs: int = 10, learning_rate: float = 1e-3):
    """Joint training routine."""
    optim = optax.adam(learning_rate=learning_rate)
    opt_state = optim.init(manifautoenc)

    @eqx.filter_value_and_grad
    def E(model: ManifoldAutoenc, x0, x1):
        z0, z1 = model.enc(x0), model.enc(x1)
        x0hat, x1hat = model.dec(z0), model.dec(z1)
        # energy term
        c = model.get_cs(z0, z1)
        z1hat = expm(jnp.einsum('i,ijk', c, model.phis)) @ z0
        likelihood = .5 * jnp.sum((z1 - z1hat)**2)
        frobnorm = jnp.sum(jnp.linalg.norm(model.phis, axis=0))
        sparse = jnp.linalg.norm(c, ord=1)
        E_psi = likelihood + frobnorm + sparse
        # joint loss
        return jnp.mean((x0-x0hat)**2) + jnp.mean((x1-x1hat)**2) + E_psi

    @eqx.filter_jit
    def make_step(model: ManifoldAutoenc, opt_state, x0: jnp.ndarray, x1: jnp.ndarray):
        loss, grad = E(model, x0, x1)
        updates, opt_state = optim.update(grad, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    losses = []
    for epoch in tqdm(range(n_epochs)):
        for x0batch, x1batch in loader:
            params = dict(
                model=manifautoenc, opt_state=opt_state,
                x0=x0batch, x1=x1batch,
                )
            loss, manifautoenc, opt_state = make_step(**params)
            losses.append(loss)
    return manifautoenc, losses


def train_phase_2(transport: TransportOperator, zpairs: jnp.ndarray, n_epochs: int = 10, learning_rate: float = 1e-3):
    """Transport operator training routine."""
    c_optim = optax.adam(learning_rate=learning_rate)
    c_opt_state = c_optim.init(transport)
    p_optim = optax.adam(learning_rate=learning_rate)
    p_opt_state = p_optim.init(transport)

    # overall energy term
    def E_psi(model: TransportOperator, z0: jnp.ndarray, z1: jnp.ndarray,
              a: float = 1, b: float = 1) -> float:
        likelihood = .5 * jnp.sum((z1 - model(z0))**2)
        frobnorm = jnp.sum(jnp.linalg.norm(model.phis, axis=0))
        sparse = jnp.linalg.norm(model.cs, ord=1)
        return likelihood + (a/2)*frobnorm + b*sparse
    # freeze psis and cs separately
    filter_spec = jtu.tree_map(lambda _: False, transport)
    fixed_cs = eqx.tree_at(lambda to: to.phis, filter_spec, replace=True)
    fixed_psis = eqx.tree_at(lambda to: to.cs, filter_spec, replace=True)
    psi_step = eqx.filter_jit(eqx.filter_value_and_grad(E_psi, arg=fixed_cs))
    c_step = eqx.filter_jit(eqx.filter_value_and_grad(E_psi, arg=fixed_psis))

    # this function is run for every point pair
    def point_pair_step(model, z0, z1, c_opt_state, p_opt_state):
        # cs are assumed to be initialized to 0: TODO make sure of this
        # Compute the correct cs (convergence condition? TODO)
        for step in range(10):
            E_c, grads = c_step(model, z0, z1)
            updates, c_opt_state = c_optim.update(grads, c_opt_state)
            model = eqx.apply_updates(model, updates)
        # Take a single step for the psi
        E_psi, grads = psi_step(model, z0, z1)
        updates, p_opt_state = p_optim.update(grads, p_opt_state)
        model = eqx.apply_updates(model, updates)
        return model, c_opt_state, p_opt_state

    # loop over the whole dataset n_epochs times and do the training
    for epoch in range(n_epochs):
        print(f"Epoch num {epoch}")
        for (z0, z1) in zpairs:
            transport, c_opt_state, p_opt_state = point_pair_step(transport, z0, z1, c_opt_state, p_opt_state)
    return transport


def train_phase_3(autoenc: Autoenc, transport: TransportOperator, loader: jnp.ndarray, n_epochs: int = 10, learning_rate: float = 1e-3):
    """Joint training routine."""
    ae_optim = optax.adam(learning_rate=learning_rate)
    to_optim = optax.adam(learning_rate=learning_rate)
    aeopt_state = ae_optim.init(autoenc)
    toopt_state = to_optim.init(transport)

    # transport operator energy term
    def E_psi(model: TransportOperator, z0: jnp.ndarray, z1: jnp.ndarray,
              a: float = 1, b: float = 1) -> float:
        likelihood = .5 * jnp.sum((z1 - vmap(model)(z0))**2)
        frobnorm = jnp.sum(jnp.linalg.norm(model.phis, axis=0))
        sparse = jnp.linalg.norm(model.cs, ord=1)
        return likelihood + (a/2)*frobnorm + b*sparse

    # reconstruction loss term
    def reconstruction_loss(model: Autoenc, x):
        x_hat = vmap(model)(x)
        return jnp.mean((x-x_hat)**2)

    # joint loss termn
    @eqx.filter_value_and_grad
    def joint_loss(ae: Autoenc, to: TransportOperator, x0: jnp.ndarray, x1: jnp.ndarray, gamma: float = 1):
        enc, dec = vmap(ae.enc), vmap(ae.dec)
        z0, z1 = enc(x0), enc(x1)
        x0hat, x1hat = dec(z0), dec(z1)
        return jnp.mean((x0-x0hat)**2) + jnp.mean((x1-x1hat)**2) + gamma * E_psi(to, z0, z1)

    @eqx.filter_jit
    def make_step(ae: Autoenc, to: TransportOperator, x0: jnp.ndarray, x1: jnp.ndarray, aeopt_state, toopt_state):
        # loss, aegrad, tograd = joint_loss(ae, to, x0, x1)
        # aeupdates, aeopt_state = ae_optim.update(aegrad, aeopt_state)
        # toupdates, toopt_state = to_optim.update(tograd, toopt_state)
        loss, grad = joint_loss(ae, to, x0, x1)
        aeupdates, aeopt_state = ae_optim.update(grad, aeopt_state)
        toupdates, toopt_state = to_optim.update(grad, toopt_state)
        ae = eqx.apply_updates(ae, aeupdates)
        to = eqx.apply_updates(to, toupdates)
        return loss, ae, to, aeopt_state, toopt_state

    for epoch in range(n_epochs):
        for x0batch, x1batch in loader:
            params = dict(
                ae=autoenc, to=transport, x0=x0batch[None, :], x1=x1batch[None, :],
                aeopt_state=aeopt_state, toopt_state=toopt_state
                )
            loss, autoenc, transport, aeopt_state, toopt_state = make_step(**params)
    return autoenc, transport


def training(ae: Autoenc, to: TransportOperator, dataset: HGMMDataset, epochs: int = 1):
    def jax_collate(samples): return jnp.stack(samples)
    # phase 1
    xloader = DataLoader(dataset, batch_size=128, collate_fn=jax_collate)
    print("Training phase 1")
    ae = train_phase_1(ae, xloader, n_epochs=epochs)
    print("Done with phase 1")
    # phase 2
    # encode and find nearest neighbors
    zdset = vmap(ae.enc)(dataset[:])
    pairwise_dists = jnp.sqrt(((zdset - zdset[:, None])**2).sum(axis=-1))
    idxs = jnp.argwhere(jnp.argsort(pairwise_dists) == 1)[:, 1]
    zpairs = jnp.stack((zdset, zdset[idxs])).transpose(1, 0, 2)
    print("Training phase 2")
    to = train_phase_2_new(to, zpairs, n_epochs=epochs)
    print("Done with phase 2")
    # phase 3
    manifautoenc = ManifoldAutoenc(ae, to)
    xpairs = jnp.stack((dataset.dset, dataset.dset[idxs])).transpose(1, 0, 2)
    print("Training phase 3")
    full_model, losses = train_phase_3_new(manifautoenc, xpairs, n_epochs=epochs)
    print("Done with phase 3")
    return full_model, losses


if __name__ == "__main__":
    key = jrand.PRNGKey(0)
    nobs = 10000
    dim_x = 100
    dim_z = 3  # TODO: make the interface a little bit better
    nops = 3
    init_delta = 1e-3
    time_mult = 100
    aekey, tokey, dsetkey, key = jrand.split(key, num=4)
    # initialize models
    aenc = Autoenc(dim_x=dim_x, dim_z=dim_z, key=aekey)
    transp = TransportOperatorAutoFit(dim=dim_z, nops=nops, key=tokey)
    print(transp)
    # initialize dataset
    dataset_params = dict(
        nobs=nobs, dd=dim_x, ops=get_SO3(), init_delta=init_delta,
        time_mult=time_mult, key=dsetkey)
    points = hgmm(**dataset_params)
    dset = HGMMDataset(points)
    # do the training
    manifold_autoenc, losses = training(aenc, transp, dset, epochs=2)



# %%
# from jax import grad


# def f(x):
#     return x*2

# def g(x):
#     return 2*x**4

# def tot(x):
#     return f(x)+g(x)

# df = grad(tot)
# df(100.)
