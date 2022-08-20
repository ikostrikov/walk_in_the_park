import real.sac_dev.util.net_util as net_util
import tensorflow as tf

NAME = "fc_2layers_512units"


def build_net(input_tfs, reuse=False):
    layers = [512, 256]
    activation = tf.nn.relu
    h = net_util.build_fc_net(input_tfs=input_tfs,
                              layers=layers,
                              activation=activation,
                              reuse=reuse)
    return h
