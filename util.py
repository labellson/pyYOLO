from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2


def predict_transform(prediction, in_dim, anchors, num_classes, cuda=True):
    batch_size = prediction.size(0)
    stride = in_dim // prediction.size(2)
    grid_size = in_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors,
                                 grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size,
                                 grid_size * grid_size * num_anchors,
                                 bbox_attrs)

    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    # Apply sigmoid to centre_x, centre_y and object confidence
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add the center offsets
    grid = np.arange(grid_size)
    xv, yv = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(xv).view(-1, 1)
    y_offset = torch.FloatTensor(yv).view(-1, 1)

    if cuda:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    prediction[:, :, :2] += x_y_offset

    # Log space transform for height and with of bbox
    anchors = torch.FloatTensor(anchors)

    if cuda:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    # Apply sigmoid to class scores
    prediction[:, :, 5:] = torch.sigmoid(prediction[:, :, 5:])

    # Resize to the input image size
    prediction[:, :, :4] *= stride
    return prediction
