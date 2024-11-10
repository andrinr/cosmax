import jax
import jax.numpy as jnp
import optax
from cosmax import cic_ma
from typing import Tuple

def loss(
        pos : jax.Array,
        mass : jax.Array,
        field_truth : jax.Array):
    """
    Loss function for fitting a 3D density field with particles

    Args:
        pos : particle positions
        mass : particle masses
        field_truth : 3D density field

    Returns:
        loss : mean squared error
    """
    
    field_pred = cic_ma(
        pos,
        mass,
        field_truth.shape[0])
    
    return jnp.mean((field_pred - field_truth) ** 2)

def fit_field(
        key : jax.Array,
        N : int,
        field : jax.Array,
        total_mass : float,
        iterations : float = 400,
        learning_rate : float = 0.005,
        ) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Given a 3D density field, fit the particle positions and masses such that
    the density field is well represented by the particles.

    Args:
        key : random key
        N : number of particles in each dimension
        field : 3D density field
        total_mass : total mass of the field
        iterations : number of iterations
        learning_rate : learning rate

    Returns:
        initial particle positions, fitted particle positions, particle masses
    """

    num_particles = N**3

    # equispaced particles in grid
    pos_lag = jnp.array(jnp.meshgrid(
        jnp.linspace(0, 1, N),
        jnp.linspace(0, 1, N),
        jnp.linspace(0, 1, N)))

    pos_lag = jnp.reshape(pos_lag, (3, num_particles))

    pos = pos_lag
    pos += jax.random.uniform(key, (3, num_particles), minval=-0.1, maxval=0.1)

    # pos = jax.random.uniform(key, (3, num_particles))
    mass = jnp.ones(num_particles) * total_mass / num_particles

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(pos)

    grad_f = jax.grad(loss)

    @jax.jit
    def step(pos, opt_state):
        grad = grad_f(pos, mass, field)
        print(grad)
        updates, opt_state = optimizer.update(grad, opt_state)
        pos = optax.apply_updates(pos, updates)
        return pos, opt_state
    
    for i in range(iterations):
        pos, opt_state = step(pos, opt_state)
        if i % 100 == 0:
            print(f"Loss: {loss(pos, mass, field)}")

    return pos_lag, pos, mass