import jax
import jax.numpy as jnp
from typing import Tuple
from .growth_factor import compute_growth_factor

def compute_overdensity_mean(rho : jax.Array) -> Tuple[jax.Array, float]:
    """
    Overdensity (delta) of a density field (rho) as defined in cosmology
    """
    mean = rho.mean()
    return (rho - mean) / mean, mean

def compute_overdensity(rho : jax.Array) -> jax.Array:
    """
    Overdensity (delta) of a density field (rho) as defined in cosmology
    """
    mean = rho.mean()
    return (rho - mean) / mean

def compute_rho(overdensity : jax.Array, mean : float) -> jax.Array:
    """
    Get density (rho) from overdensity (delta)
    """
    return overdensity * mean + mean

def growth_factor_approx(a : float, Omega_M : float, Omega_L : float):
    """
    Approximation of the growth factor in cosmology
    """
    return (5/2 * a * Omega_M) /\
        (Omega_M**(4 / 7) - Omega_L + (1 + Omega_M / 2) * (1 + Omega_L / 70 ))
