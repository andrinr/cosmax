from cosmax.spectral_ops import Potential
import jax.numpy as jnp
import jax

class Solver:
    """
    Particle Mesh Solver

    Args:
        elements : number of grid points in each dimension
        size : size of the box in real space

    Attributes:
        elements : number of grid points in each dimension
        potential : gravitational potential
    """

    def __init__(self, elements : int, size : float = 1.0):
        self.elements = elements
        self.potential = Potential(elements=elements)

    def __call__(
            self, 
            ic : jax.Array,
            dt : float = 0.01,
            steps : int = 1000):
        
        """
        Solve the N-body problem for a given initial condition.

        Args:
            ic : initial condition
            dt : time step
            steps : number of time steps

        Returns:
            final condition
        """

        
                 



