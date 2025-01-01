import jax.numpy as jnp
import jax
from typing import Tuple
from .spectral_op import SpectralOperation

class PowerSpectrum(SpectralOperation):
    """
    Power spectrum from a 3D density field.

    Args:
        elements : number of grid points in each dimension
        bins : number of bins for the power spectrum
        size : size of the box in real space

    Attributes:
        elements : number of grid points in each dimension
        size : size of the box in real space
        bins : number of bins for the power spectrum
        k : wavenumber of each bin
        index_grid : index of the bin for each wavenumber
        n_modes : number of modes in each bin
        
    """

    bins : int
    index_grid : jax.Array
    n_modes : jax.Array

    def __init__(self, elements : int, bins : int, size : float = 1.0):
        super().__init__(elements=elements, size=size)
        self.bins = bins

        self.bin_edges = jnp.linspace(0, self.k_mag.max(), self.bins + 1, endpoint=True)[1:]

        bins_pad = jnp.pad(self.bin_edges, (1, 0), mode='constant', constant_values=0)
        self.k = (bins_pad[1:] + bins_pad[:-1]) / 2

        self.index_grid = jnp.digitize(
            self.k_mag, 
            self.bin_edges,
            right=False)

        self.n_modes = jnp.zeros(self.bins)
        self.n_modes = self.n_modes.at[self.index_grid].add(1)

    def __call__(self, delta : jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Compute the power spectrum from a 3D density field
        
        Args:
            delta : 3D density field

        Returns:
            wavenumber and power spectrum
        """

        # volume of the box
        V = float(self.size ** 3)
        # volume of each grid cell
        Vx = V / self.elements ** 3

        # get the density field in fourier space
        delta_k = jnp.fft.rfftn(delta, norm="backward")  
        delta_k = Vx * delta_k

        power = jnp.real(delta_k * jnp.conj(delta_k) / V)

        power_ensemble = jnp.zeros(self.bins)
        power_ensemble = power_ensemble.at[self.index_grid].add(power)

        power_ensemble_avg = power_ensemble / self.n_modes
        power_ensemble_avg = jnp.where(jnp.isnan(power_ensemble_avg), 0, power_ensemble_avg)
    
        return self.k, power_ensemble_avg

