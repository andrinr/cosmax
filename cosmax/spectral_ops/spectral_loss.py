import jax.numpy as jnp
import jax
from typing import Tuple
from .spectral_op import SpectralOperation

class SpectralLoss(SpectralOperation):
    """
    Compute the MSE in spectral space for each wavenumber

    Args:
        elements : number of grid points in each dimension
        bins : number of bins for the power spectrum
        size : size of the box in real space

    Attributes:
        elements : number of grid points in each dimension
        size : jax.Array
        bins : number of bins for the power spectrum
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

    def __call__(self, delta_a : jax.Array, delta_b : jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Compute the power spectrum from a 3D density field

        Args:
            pred : 3D density field
            true : 3D density field

        Returns:
            wavenumber and power spectrum

        """

        # volume of the box
        V = float(self.size ** 3)
        # volume of each grid cell
        Vx = V / self.elements ** 3

        # get the density field in fourier space
        delta_k_a = jnp.fft.rfftn(delta_a, norm="backward")  
        delta_k_a = Vx * delta_k_a

        delta_k_b = jnp.fft.rfftn(delta_b, norm="backward")
        delta_k_b = Vx * delta_k_b

        power_a = delta_k_a
        power_b = delta_k_b

        power_loss = jnp.real((power_a - power_b) * jnp.conj(power_a - power_b) / V)

        power_loss_ensemble = jnp.zeros(self.bins)
        power_loss_ensemble = power_loss_ensemble.at[self.index_grid].add(power_loss)

        power_loss_ensemble_avg = power_loss_ensemble / self.n_modes
        power_loss_ensemble_avg = jnp.where(
            jnp.isnan(power_loss_ensemble_avg), 0, power_loss_ensemble_avg)
    
        return self.k, power_loss_ensemble_avg