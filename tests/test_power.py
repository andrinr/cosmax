import jax.numpy as jnp
import jax
from cosmax import PowerSpectrum

def gen_data(N):
    key = jax.random.PRNGKey(20)

    rho = jax.random.normal(key, (N, N, N), dtype=jnp.float32) + 1.0

    return rho

def test_power():
    N = 128
    n_bins = 32
    power_spectrum = PowerSpectrum(n_grid=N, n_bins=n_bins)

    delta = gen_data(N)
    
    k, P_k = power_spectrum(delta)

    assert k.shape == (n_bins,)
    assert P_k.shape == (n_bins,)

    assert jnp.all(P_k >= 0)
