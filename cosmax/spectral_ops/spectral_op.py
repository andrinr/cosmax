import jax
import jax.numpy as jnp

class SpectralOperation:
    """
    Base class for spectral operations

    Args:
        elements : number of grid points in each dimension
        size : size of the box in real space

    Attributes:
        elements : number of grid points in each dimension
        size : size of the box in real space
        frequencies : frequencies in fourier space
        real_frequencies : real frequencies in fourier space
        k_mag : magnitude of the wavenumber
    """
    k_mag = jax.Array
    frequencies : jax.Array
    elements : int
    size : float
    nyquist : int

    def __init__(self, elements : int, size : float = 1.0):
        self.elements = elements
        self.size = size
        # convert to radians per unit length
        self.frequencies = jnp.fft.fftfreq(elements, d=self.size / self.elements) * 2 * jnp.pi
        self.real_frequencies = jnp.fft.rfftfreq(elements, d=self.size / self.elements) * 2 * jnp.pi

        self.nyquist_index = jnp.ceil(elements / 2).astype(int)

        kx, ky, kz = jnp.meshgrid(self.frequencies, self.frequencies, self.real_frequencies, indexing='ij')
        self.k_mag = jnp.sqrt(kx**2 + ky**2 + kz**2)