PIXEL_RANGES = {'Sigmoid': (0, 1), 'Tanh': (-1, 1), 'ReLU': (0, 255)}


def set_requires_grad(nets, requires_grad=False):
    """
    Set requires_grad=False for all the networks to avoid unnecessary computations
    :param nets: a list of networks
    :param requires_grad: whether the networks require gradients or not
    :return:
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
