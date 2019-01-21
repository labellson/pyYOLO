from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from util import predict_transform
import numpy as np
import cv2


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


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

            if len(line) <= 0 or line[0] == '#':
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
        bn = nn.BatchNorm2d(filters)
        module.add_module('batch_norm_{}'.format(idx), bn)

    # Check the activation possible values leaky relu and linear
    if activation == 'leaky':
        function = nn.LeakyReLU(0.1, inplace=True)
        module.add_module('leaky_{}'.format(idx), function)

    return module, filters


def upsample(idx, block):
    module = nn.Sequential()
    upsample = nn.Upsample(scale_factor=2, mode='bilinear')
    module.add_module('upsample_{}'.format(idx), upsample)
    return module


def route(idx, block, output_filters):
    module = nn.Sequential()
    route_idx = block['layers'].split(',')

    start = int(route_idx[0])
    end = int(route_idx[1]) if len(route_idx) > 1 else 0

    route = EmptyLayer()
    module.add_module('route_{}'.format(idx), route)

    # Compute the output feature map size
    if start < 0:
        start += idx
    if end < 0:
        end += idx
    if end > 0:
        filters = output_filters[start] + output_filters[end]
    else:
        filters = output_filters[start]

    # Add start and end to the route dict
    block['start'] = start
    block['end'] = end

    return module, filters


def shortcut(idx, block):
    module = nn.Sequential()
    shortcut = EmptyLayer()
    module.add_module('shortcut_{}'.format(idx), shortcut)
    return module


def yolo(idx, block):
    module = nn.Sequential()
    mask = block['mask'].split(',')
    mask = [int(x) for x in mask]

    anchors = [int(a) for a in block['anchors'].split(',')]
    anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
    anchors = [anchors[i] for i in mask]

    detection = DetectionLayer(anchors)
    module.add_module('Detection_{}'.format(idx), detection)
    return module


def create_modules(blocks):
    net_info = blocks[0]  # First block contains net hyperparameters
    module_list = nn.ModuleList()
    prev_filters = 3  # RGB
    output_filters = []

    for idx, block in enumerate(blocks[1:]):
        # Create the block, create the module for the block, return module with
        # the previous filters. Then append to the module_list and
        # output_filters
        block_type = block['type']
        if block_type == 'convolutional':
            module, prev_filters = convolutional(idx, block, prev_filters)

        elif block_type == 'upsample':
            module = upsample(idx, block)

        elif block_type == 'route':
            module, prev_filters = route(idx, block, output_filters)

        elif block_type == 'shortcut':
            module = shortcut(idx, block)

        elif block_type == 'yolo':
            module = yolo(idx, block)

        # Append the module and it output feature map depth size
        module_list.append(module)
        output_filters.append(prev_filters)

    return (net_info, module_list)


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, cuda=True):
        modules = self.blocks[1:]
        outputs = {}

        write = False
        for idx, module in enumerate(modules):
            module_type = module['type']

            if module_type in ('convolutional', 'upsample'):
                x = self.module_list[idx](x)

            elif module_type == 'route':
                if module['end'] == 0:
                    x = outputs[module['start']]
                else:
                    x = torch.cat((outputs[module['start']],
                                   outputs[module['end']]), 1)

            elif module_type == 'shortcut':
                from_ = int(module['from'])
                x = outputs[idx - 1] + outputs[idx + from_]

            elif module_type == 'yolo':
                anchors = self.module_list[idx][0].anchors
                in_dim = int(self.net_info['height'])
                num_classes = int(module['classes'])

                # Transform
                x = predict_transform(x.data, in_dim, anchors, num_classes,
                                      cuda)
                if not write:
                    detections = x
                    write = True

                else:
                    detections = torch.cat((detections, x), 1)

            outputs[idx] = x

        return detections


if __name__ == '__main__':
    blocks = parse_cfg('./yolov3.cfg')
    modules = create_modules(blocks)
    print(modules)

    # Test the net forward pass
    img = cv2.imread('dog-cycle-car.png')
    size = int(blocks[0]['height'])
    img = cv2.resize(img, (size, size)) # Input dimension
    img = img[:, :, ::-1].transpose((2,0,1))
    img = img[np.newaxis] / 255.0
    img = Variable(torch.from_numpy(img).float())

    model = Darknet('./yolov3.cfg')
    pred = model(img, False)
    print(pred)
