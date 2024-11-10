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

    Args:
        pos : position of the particle
        weight : weight of the particle
        grid_size : size of the grid

    Returns:
        The grid with the mass assigned
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