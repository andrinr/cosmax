import jax
from field import gradient

def compute_velocity(potential : jax.Array, dt) -> jax.Array:
    """
    Compute velocity field from density field
    """

    acceleration = -gradient(potential)

    velocity = acceleration * dt

    return velocity