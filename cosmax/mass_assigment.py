import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(2))
def nn_ma(
        pos : jax.Array, 
        weight : jax.Array, 
        grid_size : int) -> jax.Array:

    """
    Nearest Neighbour (NN) mass assignment.
    Position are assumed to be normalized between 0 and 1.
    Periodic boundary conditions are used.
    """

    coords = jnp.linspace(start=0, stop=1, num=grid_size+1)

    grid = jnp.zeros((grid_size, grid_size, grid_size))

    # find position on the grid
    x_idx = jnp.digitize(pos[0] % 1.0, coords, right=False) - 1
    y_idx = jnp.digitize(pos[1] % 1.0, coords, right=False) - 1
    z_idx = jnp.digitize(pos[2] % 1.0, coords, right=False) - 1

    # assign the mass
    grid = grid.at[x_idx, y_idx, z_idx].add(weight)

    return grid

@partial(jax.jit, static_argnums=(2))
def cic_ma(
        pos : jax.Array, 
        weight : jax.Array, 
        grid_size : int) -> jax.Array:

    """
    Periodic cloud in a cell mass (CIC) mass assignment. 
    Position are assumed to be normalized between 0 and 1.
    Periodic boundary conditions are used.
    """
    
    dx = 1 / grid_size
    coords = jnp.linspace(start=0, stop=1, num=grid_size+1)

    field = jnp.zeros((grid_size, grid_size, grid_size))

    # find position on the grid
    x = jnp.digitize(pos[0] % 1.0, coords, right=False) - 1
    y = jnp.digitize(pos[1] % 1.0, coords, right=False) - 1
    z = jnp.digitize(pos[2] % 1.0, coords, right=False) - 1

    # find the weights
    xw = (pos[0] % 1.0 - coords[x]) / dx
    yw = (pos[1] % 1.0 - coords[y]) / dx
    zw = (pos[2] % 1.0 - coords[z]) / dx

    offsets = jnp.array([-1, 1])
    oxs, oys, ozs = jnp.meshgrid(offsets, offsets, offsets)

    # assign the mass
    field = field.at[x, y, z].add(weight * (1 - xw) * (1 - yw) * (1 - zw))
    field = field.at[(x + 1) % grid_size, y, z].add(weight * xw * (1 - yw) * (1 - zw))
    field = field.at[x, (y + 1) % grid_size, z].add(weight * (1 - xw) * yw * (1 - zw))
    field = field.at[(x + 1) % grid_size, (y + 1) % grid_size, z].add(weight * xw * yw * (1 - zw))
    field = field.at[x, y, (z + 1) % grid_size].add(weight * (1 - xw) * (1 - yw) * zw)
    field = field.at[(x + 1) % grid_size, y, (z + 1) % grid_size].add(weight * xw * (1 - yw) * zw)
    field = field.at[x, (y + 1) % grid_size, (z + 1) % grid_size].add(weight * (1 - xw) * yw * zw)
    field = field.at[(x + 1) % grid_size, (y + 1) % grid_size, (z + 1) % grid_size].add(weight * xw * yw * zw)

    return field