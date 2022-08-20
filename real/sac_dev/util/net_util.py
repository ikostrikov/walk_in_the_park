import tensorflow as tf

from .ensemble_fc import FC


def build_fc_net(input_tfs,
                 layers,
                 activation=tf.nn.relu,
                 weight_init=tf.compat.v1.keras.initializers.VarianceScaling(
                     scale=1.0, mode="fan_avg", distribution="uniform"),
                 reuse=False):
    curr_tf = tf.concat(axis=-1, values=input_tfs)
    for i, size in enumerate(layers):
        with tf.compat.v1.variable_scope(str(i), reuse=reuse):
            curr_tf = tf.compat.v1.layers.dense(inputs=curr_tf,
                                                units=size,
                                                kernel_initializer=weight_init,
                                                activation=activation)
    return curr_tf


def build_fc_ensemble_net(
        input_tfs,
        layers,
        ensemble_size=2,
        activation="ReLU",
        weight_init=tf.compat.v1.keras.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"),
        reuse=False):

    # import ipdb; ipdb.set_trace()
    curr_tf = tf.concat(axis=-1, values=input_tfs)
    if len(curr_tf.shape) == 3:
        curr_tf = tf.squeeze(curr_tf, axis=-2)
    # totally incorrect if we use num_samples > 1 but for the sake of running it asap
    # curr_tf = tf.expand_dims(curr_tf, axis=0)
    input_dims = [int(curr_tf.shape[-1])]
    input_dims.extend(layers[:-1])

    # curr_tf = tf.tile(curr_tf, tf.constant([ensemble_size, 1, 1], tf.int32))

    structure = []

    for i, size in enumerate(layers):
        layer_act = activation
        if i == len(layers) - 1:
            layer_act = None
        layer = FC(size,
                   input_dim=input_dims[i],
                   activation=layer_act,
                   weight_decay=None,
                   ensemble_size=ensemble_size)

        structure.append(layer)

    for i, layer in enumerate(structure):
        with tf.compat.v1.variable_scope("Layer%i" % i):
            layer.construct_vars()

    for layer in structure:
        curr_tf = layer.compute_output_tensor(curr_tf)

    return curr_tf
