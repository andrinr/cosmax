import jax
import jax.numpy as jnp

@jax.jit
def bilinear_interp(
        pos : jax.Array, 
        scalar_field : jax.Array) -> jax.Array:
    """
    Bilinear interpolation of a scalar field.

    Returns values of a scalar field at a given position using bilinear interpolation.
    Periodic boundary conditions are used.

    Args:
        pos : position to interpolate
        scalar_field : scalar field

    Returns:
        The interpolated value
    """
    
    n = scalar_field.shape[0]
    dx = 1.0 / (n)
    coords = jnp.linspace(start=0, stop=1, num=n+1)

    # find position on the grid
    x_idx = jnp.digitize(pos[0] % 1.0, coords, right=False) - 1
    y_idx = jnp.digitize(pos[1] % 1.0, coords, right=False) - 1
    z_idx = jnp.digitize(pos[2] % 1.0, coords, right=False) - 1

    # find the weights
    x_w = (pos[0] % 1.0 - coords[x_idx]) / dx
    y_w = (pos[1] % 1.0 - coords[y_idx]) / dx
    z_w = (pos[2] % 1.0 - coords[z_idx]) / dx

    # perform the interpolation
    interp = scalar_field[x_idx, y_idx, z_idx] * (1 - x_w) * (1 - y_w) * (1 - z_w)
    interp += scalar_field[(x_idx + 1) % n, y_idx, z_idx] * x_w * (1 - y_w) * (1 - z_w)
    interp += scalar_field[x_idx, (y_idx + 1) % n, z_idx] * (1 - x_w) * y_w * (1 - z_w) 
    interp += scalar_field[(x_idx + 1) % n, (y_idx + 1) % n, z_idx] * x_w * y_w * (1 - z_w)
    interp += scalar_field[x_idx, y_idx, (z_idx + 1) % n] * (1 - x_w) * (1 - y_w) * z_w
    interp += scalar_field[(x_idx + 1) % n, y_idx, (z_idx + 1) % n] * x_w * (1 - y_w) * z_w
    interp += scalar_field[x_idx, (y_idx + 1) % n, (z_idx + 1) % n] * (1 - x_w) * y_w * z_w
    interp += scalar_field[(x_idx + 1) % n, (y_idx + 1) % n, (z_idx + 1) % n] * x_w * y_w * z_w

    return interp
