import real.sac_dev.learning.nets.fc_2layers_256units as fc_2layers_256units
import real.sac_dev.learning.nets.fc_2layers_512units as fc_2layers_512units


def build_net(net_name, input_tfs, reuse=False):
    net = None

    if (net_name == fc_2layers_256units.NAME):
        net = fc_2layers_256units.build_net(input_tfs, reuse)
    elif (net_name == fc_2layers_512units.NAME):
        net = fc_2layers_512units.build_net(input_tfs, reuse)
    else:
        assert False, 'Unsupported net: ' + net_name

    return net
