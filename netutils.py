import numpy as np


def get_param_name(param, idx):
    if idx == 0:
        return param+"_W"
    elif idx == 1:
        return param+"_b"
    else:
        return param+"_%d" % idx


def split_param_name(param_name):
    name = param_name[:-2]
    idx_str = param_name[-1]
    if idx_str == 'W':
        idx = 0
    elif idx_str == 'b':
        idx = 1
    else:
        idx = int(idx_str)
    return name, idx


def extract_net_params(caffe_net, params=None):
    if params is None:
        params = caffe_net.params

    data = {}
    for param in params:
        for i in range(len(caffe_net.params[param])):
            name = get_param_name(param, i)
            data[name] = caffe_net.params[param][i].data

    return data


def set_net_params(caffe_net, params):
    for param_name in params:
        name, i = split_param_name(param_name)
        caffe_net.params[name][i].data[:] = params[param_name]


def pretty_format(param_dict):
    pretty_string = ""
    for param in sorted(param_dict.keys()):
        pretty_string += "="*80 + '\n'
        pretty_string += param + '\n'
        pretty_string += str(np.squeeze(param_dict[param]))
        pretty_string += '\n'

    return pretty_string
