import jax.numpy as jnp
import jax
from cosmax import cic_ma, nn_ma

def test_nn():
    pos = jnp.array([
        [0],
        [0.6125],
        [0]
    ])
    weight = jnp.array([1.0])
    total = jnp.sum(weight)

    field = nn_ma(pos, weight, 4)

    print(f"field: {field}")

    assert field[0, 2, 0] == 1.0

    assert jnp.abs(jnp.sum(field) - total) < 1e-5

def test_cic():
    N = 100
    pos = jax.random.uniform(jax.random.PRNGKey(0), (3, N))
    weight = jax.random.uniform(jax.random.PRNGKey(1), (N,))

    total = jnp.sum(weight)

    field = cic_ma(pos, weight, 4)
    
    assert jnp.abs(jnp.sum(field) - total) < 1e-5
    
