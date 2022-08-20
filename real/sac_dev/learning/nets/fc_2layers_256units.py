import real.sac_dev.util.net_util as net_util
import tensorflow as tf

NAME = "fc_2layers_256units"


def build_net(input_tfs, reuse=False):
    layers = [256, 128]
    activation = tf.nn.relu
    h = net_util.build_fc_net(input_tfs=input_tfs,
                              layers=layers,
                              activation=activation,
                              reuse=reuse)
    return h
