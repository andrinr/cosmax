import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(2))
def cic_ma(
        pos : jax.Array, 
        weight : jax.Array, 
        elements : int,
        size : float = 1.0) -> jax.Array:

    """
    Periodic cloud in a cell mass (CIC) mass assignment. 
    
    Position are assumed to be normalized between 0 and 1.
    Periodic boundary conditions are used.

    Args:
        pos : position of the particle
        weight : weight of the particle
        elements : number of elements in each dimension
        size : size of the box in real space

    Returns:
        The grid with the mass assigned
    
    """
    
    dx = 1 / elements
    coords = jnp.linspace(start=0, stop=size, num=elements+1)

    field = jnp.zeros((elements, elements, elements))

    # find position on the grid
    x = jnp.digitize(pos[0] % size, coords, right=False) - 1
    y = jnp.digitize(pos[1] % size, coords, right=False) - 1
    z = jnp.digitize(pos[2] % size, coords, right=False) - 1

    # find the weights
    xw = (pos[0] % size - coords[x]) / dx
    yw = (pos[1] % size - coords[y]) / dx
    zw = (pos[2] % size - coords[z]) / dx

    # assign the mass
    field = field.at[x, y, z].add(weight * (1 - xw) * (1 - yw) * (1 - zw))
    field = field.at[(x + 1) % elements, y, z].add(weight * xw * (1 - yw) * (1 - zw))
    field = field.at[x, (y + 1) % elements, z].add(weight * (1 - xw) * yw * (1 - zw))
    field = field.at[(x + 1) % elements, (y + 1) % elements, z].add(weight * xw * yw * (1 - zw))
    field = field.at[x, y, (z + 1) % elements].add(weight * (1 - xw) * (1 - yw) * zw)
    field = field.at[(x + 1) % elements, y, (z + 1) % elements].add(weight * xw * (1 - yw) * zw)
    field = field.at[x, (y + 1) % elements, (z + 1) % elements].add(weight * (1 - xw) * yw * zw)
    field = field.at[(x + 1) % elements, (y + 1) % elements, (z + 1) % elements].add(weight * xw * yw * zw)

    return field