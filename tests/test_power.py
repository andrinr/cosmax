import jax.numpy as jnp
import jax
from cosmax import PowerSpectrum
from powerbox import get_power

def gen_data(N):
    key = jax.random.PRNGKey(20)

    rho = jax.random.normal(key, (N, N, N), dtype=jnp.float32) + 1.0

    return rho

def test_power():

    file = '../data/z49.bin'

    # open file
    with open(file, 'rb') as f:
        # read data
        data = f.read()
        # convert to numpy array
        delta = jnp.frombuffer(data, dtype=jnp.float32)
        delta = delta.reshape(256, 256, 256)

    N = 256
    n_bins = 48
    power_spectrum = PowerSpectrum(elements=N, bins=n_bins)

    # delta = gen_data(N)
    
    k, P = power_spectrum(delta)

    assert k.shape == (n_bins,)
    assert P.shape == (n_bins,)
