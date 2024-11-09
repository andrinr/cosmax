import jax
import jax.numpy as jnp
from .gradient import central_difference

@jax.jit
def bilinear_interp(
        pos : jax.Array, 
        scalar_field : jax.Array) -> jax.Array:
    """
    Returns values of a scalar field at a given position using bilinear interpolation.
    Periodic boundary conditions are used.
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

def bicubic_interp(
        pos : jax.Array, 
        scalar_field : jax.Array) -> jax.Array:
    """
    Returns values of a scalar field at a given position using bicubic interpolation.
    Periodic boundary conditions are used.
    """
    
    n = scalar_field.shape[0]
    n_particles = pos.shape[0]
    dx = 1.0 / (n)
    coords = jnp.linspace(start=0, stop=1, num=n+1)

    # compute grid derivatives
    grad_x = central_difference(scalar_field, 0, dx)
    grad_y = central_difference(scalar_field, 1, dx)
    grad_z = central_difference(scalar_field, 2, dx)

    grad_xy = central_difference(grad_x, 1, dx)
    grad_xz = central_difference(grad_x, 2, dx)
    grad_yz = central_difference(grad_y, 2, dx)

    grad_xyz = central_difference(grad_xy, 2, dx)

    # find position on the grid
    x = jnp.digitize(pos[0] % 1.0, coords, right=False) - 1
    y = jnp.digitize(pos[1] % 1.0, coords, right=False) - 1
    z = jnp.digitize(pos[2] % 1.0, coords, right=False) - 1

    # find values 
    f = jnp.zeros((n_particles, 2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                f = jax.ops.index_update(f, jax.ops.index[:, i, j, k], 
                                                   scalar_field[(x + i) % n, (y + j) % n, (z + k) % n])

    f_dx = jnp.zeros((n_particles, 2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                f_dx = jax.ops.index_update(f_dx, jax.ops.index[:, i, j, k], 
                                                   grad_x[(x + i) % n, (y + j) % n, (z + k) % n])
                
    f_dy = jnp.zeros((n_particles, 2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                f_dy = jax.ops.index_update(f_dy, jax.ops.index[:, i, j, k], 
                                                   grad_y[(x + i) % n, (y + j) % n, (z + k) % n])
                
    f_dz = jnp.zeros((n_particles, 2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                f_dz = jax.ops.index_update(f_dz, jax.ops.index[:, i, j, k], 
                                                   grad_z[(x + i) % n, (y + j) % n, (z + k) % n])
                
    f_dxy = jnp.zeros((n_particles, 2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                f_dxy = jax.ops.index_update(f_dxy, jax.ops.index[:, i, j, k], 
                                                   grad_xy[(x + i) % n, (y + j) % n, (z + k) % n])
                
    f_dxz = jnp.zeros((n_particles, 2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                f_dxz = jax.ops.index_update(f_dxz, jax.ops.index[:, i, j, k], 
                                                   grad_xz[(x + i) % n, (y + j) % n, (z + k) % n])
                
    f_dyz = jnp.zeros((n_particles, 2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                f_dyz = jax.ops.index_update(f_dyz, jax.ops.index[:, i, j, k], 
                                                   grad_yz[(x + i) % n, (y + j) % n, (z + k) % n])
                
    f_dxyz = jnp.zeros((n_particles, 2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                f_dxyz = jax.ops.index_update(f_dxyz, jax.ops.index[:, i, j, k], 
                                                   grad_xyz[(x + i) % n, (y + j) % n, (z + k) % n])
    
