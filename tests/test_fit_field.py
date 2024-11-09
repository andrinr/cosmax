from field import fit_field, cic_ma
import jax
import jax.numpy as jnp

def test_fit_field():
    key = jax.random.PRNGKey(0)
    key_field, key_opt = jax.random.split(key)
    N = 32
    field = jax.random.uniform(key_field, (N, N, N), minval=0.001, maxval=0.002)
    pos_lag, pos, mass = fit_field(
        key=key_opt,
        N=N,
        field=field, 
        total_mass=jnp.sum(field),
        iterations=1000,
        learning_rate=0.001)
    
    field_pred = cic_ma(pos, mass, field.shape[0])

    assert jnp.mean((field - field_pred) ** 2) < 0.0001
    assert pos.shape == (3, N**3)
    assert mass.shape == (N**3,)


    assert False