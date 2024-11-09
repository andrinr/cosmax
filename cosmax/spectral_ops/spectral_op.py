import jax
import jax.numpy as jnp

class SpectralOperation:
    """
    A base class for precomputing the frequencies for square 3D grids.
    """
    k = jax.Array
    frequencies : jax.Array
    n_grid : int
    nyquist : int

    def __init__(self, n_grid : int):
        self.n_grid = n_grid
        self.nyquist = n_grid // 2 + 1
        self.frequencies = jnp.fft.fftfreq(n_grid) * n_grid

        kx, ky, kz = jnp.meshgrid(self.frequencies, self.frequencies, self.frequencies[0:self.nyquist])
        self.k = jnp.sqrt(kx**2 + ky**2 + kz**2)