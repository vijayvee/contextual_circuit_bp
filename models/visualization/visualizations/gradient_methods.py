import tensorflow as tf


def gradient_image(x, layer):
    """Wrapper for simonyan 2014 gradient image."""
    return vis.gradient_image(x, layer)


def gauss_noise(x, mu=0, std=0.15):
    shape = [int(d) for d in x.get_shape()]
    return tf.random_normal(shape=shape, mean=mu, stddev=std, dtype=tf.float32)


def np_gauss_noise(x, mu, std=0.15):
    return np.random.normal(mu, std, x_value.shape)


def stochastic_gradient_image(x, layer, num_iterations=50):
    x_shape = [int(d) for d in x.get_shape()[:-2]]
    perms = tf.zeros(x_shape)
    for idx in num_iterations:
        noise = gauss_noise(x)
        perms += gradient_image(x + noise, layer)
    return perms / num_iterations

