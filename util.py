from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2


def predict_transform(prediction, in_dim, anchors, num_classes, cuda=True):
    """
    NBatches: Number of input batches
    BBoxes: Total amount of bounding boxes given the following pixel order ->
    [(0,0), (0,0), (0,0), (0,1), ..., (n, n)]
    Attributes: tx + ty + w + h + num_classes

    :return: (NBatches, BBoxes, Attributes) -> torch.Tensor
    """
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


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    
    
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
 
    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou


def write_results(prediction, confidence, num_classes, nms_conf=0.4, cuda=True):
    # Remove all the predictions below confidence
    prediction = prediction[prediction[:, :, 4] > confidence]

    if len(prediction.shape) < 3:
        prediction = prediction.unsqueeze(0)

    # Convert the prediction to (top-left x, top-left y, bottom-right x,
    # bottom-right y, ...)
    tx, ty, w, h = prediction[:, :, 0].clone(), prediction[:, :, 1].clone(), \
                   prediction[:, :, 2].clone(), prediction[:, :, 3].clone()
    prediction[:, :, 0] = tx - w / 2
    prediction[:, :, 1] = ty - h / 2
    prediction[:, :, 2] = tx + w / 2
    prediction[:, :, 3] = ty + h / 2

    write = False
    for batch_idx in range(prediction.size(0)):
        img_pred = prediction[batch_idx]

        # Compute the belonging class
        confidence, class_ = torch.max(img_pred[:, 5:], 1)
        confidence = confidence.float().unsqueeze(1)
        class_ = class_.float().unsqueeze(1)
        img_pred = torch.cat((img_pred[:, :5], confidence, class_), 1)

        # Get classes detected in the image
        img_classes = torch.unique(img_pred[:, -1].cpu(), sorted=True)

        if cuda:
            img_classes = img_classes.cuda()

        # Perform NMS classwise
        for cls in img_classes:
            # Get the detections of one particular class
            img_pred_class = img_pred[img_pred[:, -1] == cls]

            # Sort detections given the confidence
            _, sort_conf_idx = torch.sort(img_pred_class[:, 4],
                                          descending=True)
            img_pred_class = img_pred_class[sort_conf_idx]

            for idx in range(img_pred_class.size(0)):
                # Get IoU of all boxes after
                if idx + 1 >= img_pred_class.size(0):
                    break

                ious = bbox_iou(img_pred_class[idx].unsqueeze(0),
                                img_pred_class[idx + 1:])

                # Remove detections with IoU > nms_threshold
                iou_idx = ious < nms_conf
                img_pred_class = torch.cat((img_pred_class[:idx + 1],
                                            img_pred_class[idx + 1:][iou_idx]))

            batch_idx_tensor = img_pred_class.new(img_pred_class.size(0),
                                                  1).fill_(batch_idx)

            out = (batch_idx_tensor, img_pred_class)
            if not write:
                output = torch.cat(out, 1)
                write = True

            else:
                output = torch.cat((output, torch.cat(out, 1)))

    try:
        return output

    except Exception:
        return 0


def letterbox_image(img, dim):
    """
    Resize image with unchanged aspect ratio using padding and filled with
    (128, 128, 128)
    """
    img_h, img_w, _ = img.shape
    w, h = dim
    scale_factor = w / max(img_w, img_h)

    new_w = int(img_w * scale_factor)
    new_h = int(img_h * scale_factor)

    resized_image = cv2.resize(img, (new_w, new_h),
                               interpolation=cv2.INTER_CUBIC)
    canvas = np.full((h, w, 3), 128)
    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h,
           (w - new_w) // 2:(w - new_w) // 2 + new_w] = resized_image
    return canvas


def prep_image(img, dim):
    """
    Convert images to the input type of the neural networ
    
    :return: torch.autograd.Variable
    """
    img = letterbox_image(img, (dim, dim))
    img = img[:, :, ::-1].copy().transpose((2, 0, 1))
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img
