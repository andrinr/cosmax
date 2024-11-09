import jax
import jax.numpy as jnp
from typing import Tuple

def compute_overdensity_mean(rho : jax.Array) -> Tuple[jax.Array, float]:
    """
    Overdensity (delta) of a density field (rho) as defined in cosmology

    :param rho: density field
    :return: The overdensity
    :return: The mean density

    """
    mean = rho.mean()
    return (rho - mean) / mean, mean

def compute_overdensity(rho : jax.Array) -> jax.Array:
    """
    Overdensity (delta) of a density field (rho) as defined in cosmology

    :param rho: density field
    :return: The overdensity
    """
    mean = rho.mean()
    return (rho - mean) / mean

def compute_rho(overdensity : jax.Array, mean : float) -> jax.Array:
    """
    Get density (rho) from overdensity (delta)

    :param overdensity: overdensity field
    :param mean: mean density
    :return: The density field
    """
    return overdensity * mean + mean

def growth_factor_approx(a : float, Omega_M : float, Omega_L : float):
    """
    Approximation of the growth factor in cosmology

    :param a: scale factor
    :param Omega_M: matter density parameter
    :param Omega_L: dark energy density parameter
    :return: The growth factor
    """
    return (5/2 * a * Omega_M) /\
        (Omega_M**(4 / 7) - Omega_L + (1 + Omega_M / 2) * (1 + Omega_L / 70 ))
