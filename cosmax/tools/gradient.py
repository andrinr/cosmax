import jax
import jax.numpy as jnp

def central_difference(
        field : jax.Array,
        axis : int, 
        delta : float) -> jax.Array:
    """
    Compute the central difference of a field along a given axis.
    Periodic boundary conditions are used.

    Args:
        field : field to differentiate
        axis : axis along which to differentiate
        delta : grid spacing

    Returns:
        The central difference of the field
    """
    
    field_r = jnp.roll(field, 1, axis=axis)
    field_l = jnp.roll(field, -1, axis=axis)
    
    return (field_r - 2 * field + field_l) / (delta ** 2)

def gradient(
        field : jax.Array,
        delta : float) -> jax.Array:
    """
    Compute the gradient of a field using central differences.
    Periodic boundary conditions are used.

    Args:
        field : field to differentiate
        delta : grid spacing

    Returns:
        The gradient of the field
    """
    
    grad_x = central_difference(field, 0, delta)
    grad_y = central_difference(field, 1, delta)
    grad_z = central_difference(field, 2, delta)

    return jnp.stack([grad_x, grad_y, grad_z], axis=0)