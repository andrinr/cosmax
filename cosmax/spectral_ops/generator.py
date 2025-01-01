import jax.numpy as jnp
import jax
from .spectral_op import SpectralOperation

class Generator(SpectralOperation):
    """
    Generates initial conditions given a power spectrum

    Args:
        elements : number of grid points in each dimension
        size : size of the box in real space

    Attributes:
        elements : number of grid points in each dimension
        size : size of the box in real space
        k : wavenumber of each bin
        index_grid : index of the bin for each wavenumber
    """
    n_bins : int
    index_grid : jax.Array

    def __init__(self, elements : int, size : float = 1.0):
        super().__init__(elements=elements, size=size)

        self.bin_edges = jnp.linspace(0, self.k_mag.max(), self.elements + 1, endpoint=True)[1:]

        bins_pad = jnp.pad(self.bin_edges, (1, 0), mode='constant', constant_values=0)
        self.k = (bins_pad[1:] + bins_pad[:-1]) / 2

        self.index_grid = jnp.digitize(
            self.k_mag, 
            self.bin_edges,
            right=False)

    def __call__(self, field : jax.Array, Pk : jax.Array) -> jax.Array:
        """
        Generate a 3D density for a given power spectrum.

        Args:
            field : 3D density field to be perturbed
            Pk : power spectrum

        Returns:
            3D density field

        """

        assert Pk.shape == (self.elements,)

        # Generate the correlation kernel
        Ax = jnp.sqrt(Pk)
        Ax = Ax.at[self.index_grid].get()

        # volume of the box
        V = float(self.size ** 3)
        # volume of each grid cell
        Vx = V / self.elements ** 3

        # Account for the normalization of the random field
        field = field / jnp.sqrt(Vx)

        delta_k = jnp.fft.rfftn(field)

        # Multiply the random field by the correlation kernel
        delta_k = delta_k * Ax

        # Transform back to real space
        delta = jnp.fft.irfftn(delta_k)

        return delta