import jax.numpy as jnp
import jax
from .spectral_op import SpectralOperation

class Potential(SpectralOperation):
    """
    Gravitational potential from a 3D density field.
    
    Args:
        n_grid : number of grid points in each dimension

    """

    def __init__(self, n_grid : int):
        super().__init__(n_grid=n_grid)

    def __call__(
            self, 
            field : jax.Array, 
            G : float = 6.6743 * 10**(-11)):
        
        """
        Compute the potential from a 3D density field
        
        Args:
            field : 3D density field
            G : gravitational

        Returns:
            potential : gravitational potential
        """
        
        potential = jnp.fft.rfftn(
            field,  
            s=(self.n_grid, self.n_grid, self.n_grid), 
            axes=(1, 2, 3))
        
        potential = -4 * jnp.pi * potential  * self.k *G

        potential = jnp.fft.irfftn(
            field,  
            s=(self.n_grid, self.n_grid, self.n_grid), 
            axes=(1, 2, 3))
        
        return potential