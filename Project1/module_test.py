# /bin/env python3.8
 
import pytest

@pytest.mark.parametrize("num_outputs", [1, 16, 128])
def test_dim_basisy(num_outputs):
    import tensorflow as tf

    from linear import Linear
    from linear import Basis_Expansion
    from linear import Model
    

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10

    linear = Linear(num_inputs, num_outputs)
    basis = Basis_Expansion(num_inputs,num_outputs)
    J = Model(num_inputs,num_outputs)

    a = rng.normal(shape=[num_inputs,1])
    z = basis(a)
    print(tf.shape(z))

    tf.assert_equal(tf.shape(z)[-1], num_outputs)

@pytest.mark.parametrize("num_outputs", [1, 16, 128])
def test_dimensionality(num_outputs):
    import tensorflow as tf 

    from linear import Model 

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10 

    J = Model(num_inputs,num_outputs)
    a = rng.normal(shape=[num_inputs,1])
    J_a = J(a)
    tf.assert_equal(tf.shape(J_a), num_outputs)

@pytest.mark.parametrize("bias", [True, False])
def test_trainable(bias):
    import tensorflow as tf

    from linear import Model 

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1

    J = Model(num_inputs,num_outputs)

    a = rng.normal(shape=[num_inputs,1])

    with tf.GradientTape() as tape:
        z = J(a)
        loss = tf.math.reduce_mean(z**2)

    grads = tape.gradient(loss, J.trainable_variables)

    for grad, var in zip(grads, J.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)

    assert len(grads) == len(J.trainable_variables)

    if bias:
        assert len(grads) == 4
    else:
        assert len(grads) == 3