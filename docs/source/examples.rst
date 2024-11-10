Examples 
========

* Finding the lagrangian and eulerian positions of an unstrucutred particle distribution, given a 3D grid of the density field.

.. code-block:: python
    
    def inv_cic_ma():
        key = jax.random.PRNGKey(0)
        key_field, key_opt = jax.random.split(key)
        N = 32
        field = jax.random.uniform(key_field, (N, N, N), minval=0.001, maxval=0.002)
        pos_lag, pos, mass = fit_field(
            key=key_opt,
            N=N,
            field=field, 
            total_mass=jnp.sum(field),
            iterations=100,
            learning_rate=0.001)

        return pos_lag, pos, mass

* Defining a power spectrum loss function for a neural network that predicts the density field.

.. code-block:: python

    def power_spectrum_loss(prediction : jax.Array, truth : jax.Array):
        power_spectrum = PowerSpectrum(
            64, 20)
        p_pred, k = power_spectrum(prediction)
        p_true, k = power_spectrum(truth)

        return mse(jnp.log(p_pred), jnp.log(p_true))
