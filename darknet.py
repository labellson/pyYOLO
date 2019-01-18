from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    """
    with open(cfgfile, 'r') as f:
        block = {}
        blocks = []

        for line in f:
            line = line.lstrip().rstrip()  # Remove begin/end whitespaces

            if len(line) < 0 or line[0] == '#':
                continue

            if line[0] == '[':
                if len(block) > 0:  # Re-init the nn block and append to block list
                    blocks.append(block)
                    block = {}

                block["type"] = line[1:-1].rstrip().lstrip()
            else:
                key, value = line.split('=')
                block[key.rstrip()] = value.lstrip()

        blocks.append(block)
        return blocks


def convolutional(idx, block, in_channels):
    assert block['type'] == 'convolutional', 'Convolutional type required'
    module = nn.Sequential()

    # Get the convolutional block parameters from cfg dict
    batch_normalize = int(block.get('batch_normalize', 0))
    bias = False if batch_normalize else True
    filters = int(block['filters'])
    kernel_size = int(block['size'])
    stride = int(block['stride'])
    activation = block['activation']
    pad = (kernel_size - 1) // 2 if block['activation'] else 0

    # Add convolutional layer
    conv = nn.Conv2d(in_channels, filters, kernel_size, stride, pad, bias=bias)
    module.add_module('conv_{}'.format(idx), conv)

    if batch_normalize:
        bn = nn.BatchNorm2d(filter)
        module.add_module('batch_norm_{}'.format(idx), bn)

    # Check the activation possible values leaky relu and linear
    if activation == 'leaky':
        function = nn.LeakyReLU(0.1, inplace=True)
        module.add_module('leaky_{}'.format(idx), function)

    return module, filters


def create_modules(blocks):
    net_info = blocks[0]  # First block contains net hyperparameters
    module_list = nn.ModuleList()
    prev_filters = 3  # RGB
    output_filters = []

    for idx, block in enumerate(blocks[1:]):
        # Create the block, create the module for the block, return module with
        # the previous filters. Then append to the module_list and
        # output_filters
        if block['type'] == 'convolutional':
            module, prev_filters = convolutional(idx, block, prev_filters)

        # Append the module and it output feature map depth size
        module_list.append(module)
        output_filters.append(prev_filters)
