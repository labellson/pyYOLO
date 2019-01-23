from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import prep_image, write_results
from argparse import ArgumentParser
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random


def arg_parse():
    """
    Parse arguements to the detect module
    """
    parser = ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--images", dest='images',
                        help="Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)

    parser.add_argument("--det", dest='det',
                        help="Image / Directory to store detections to",
                        default="det", type=str)

    parser.add_argument("--bs", dest="bs", help="Batch size", default=1,
                        type=int)
    parser.add_argument("--confidence", dest="confidence",
                        help="Object Confidence to filter predictions",
                        default=0.5, type=float)
    parser.add_argument("--nms_thresh", dest="nms_thresh",
                        help="NMS Threshhold", default=0.4, type=float)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso',
                        help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=int)

    return parser.parse_args()


def load_classes(namesfile):
    with open(namesfile, 'r') as f:
        return f.read().splitlines()


if __name__ == '__main__':
    args = arg_parse()

    images = args.images
    batch_size = args.bs
    confidence = args.confidence
    nms_thresh = args.nms_thresh

    cuda = torch.cuda.is_available()

    classes = load_classes('./data/coco.names')
    with open('./data/pallete', 'rb') as f:
        colors = pkl.load(f)

    # Load the neural network
    print('Loading the network...')
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print('Network succesfully loaded')

    in_dim = args.reso
    model.net_info['height'] = in_dim
    assert in_dim % 32 == 0, 'Argument resolution --reso must be divisible by 32'
    assert in_dim > 32, 'Argument resolution --reso must be greater than 32'

    if cuda:
        model.cuda()

    # Set the model in evaluation mode
    model.eval()

    # Load the images
    t_read_dir = time.time()
    try:
        if osp.isdir(images):
            imlist = [osp.join(osp.realpath('.'), images, img) for img in
                      os.listdir(images) if osp.isfile(osp.join(images, img))]
        else:
            imlist = [osp.join(osp.realpath('.'), images)]

    except Exception.FileNotFoundError:
        print('No file or directory with the name: {}'.format(images))
        exit()

    if not osp.exists(args.det):
        os.mkdir(args.det)

    t_load_batch = time.time()
    loaded_img = [cv2.imread(im) for im in imlist]

    # PyTorch Variables for images
    im_batches = list(map(prep_image, loaded_img,
                      [in_dim for _ in range(len(imlist))]))

    # List containing dimension of original images
    im_dim_list = [(im.shape[1], im.shape[0]) for im in loaded_img]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

    if cuda:
        im_dim_list = im_dim_list.cuda()

    # Create the batches
    leftover = 0
    if len(im_dim_list) % batch_size:
        leftover = 1

    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover
        im_batches = [torch.cat((im_batches[i * batch_size:min((i + 1) * batch_size,
                       len(im_batches))])) for i in range(num_batches)]

    # Predict
    write = False
    output = []
    t_detection = time.time()
    for idx, batch in enumerate(im_batches):
        t_loop_start = time.time()

        if cuda:
            batch = batch.cuda()

        pred = model(Variable(batch, volatile=True), cuda)
        pred = write_results(pred, confidence, len(classes),
                             nms_conf=nms_thresh, cuda=cuda)
        t_loop_end = time.time()

        if type(pred) == int:
            for im_num, image in enumerate(imlist[i * batch_size:
                                                  min((i + 1) * batch_size,
                                                      len(imlist))]):
                im_id = idx * batch_size + im_num
                print('{0:20s} predicted in {1:6.3f} seconds'.format(image.split('/')[-1],
                                                                     (t_loop_end - t_loop_start) / batch_size))
                print('{0:20s} {1:s}'.format('Objects Detected:', ''))
                print("----------------------------------------------------------")
            continue

        pred[:, 0] += idx * batch_size  # Transform idx batch to idx images

        if not write:
            output = pred
            write = True
        else:
            output = torch.cat((output, pred))

        for im_num, image in enumerate(imlist[idx * batch_size:
                                              min((idx + 1) * batch_size,
                                                  len(imlist))]):
            im_id = idx * batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            #objs = [classes[int(x[-1])] for x in pred if x[0] == im_num]
            print('{0:20s} predicted in {1:6.3f} seconds'.format(image.split('/')[-1],
                                                                 (t_loop_end - t_loop_start) / batch_size))
            print('{0:20s} {1:s}'.format('Objects Detected:', ' '.join(objs)))
            print("----------------------------------------------------------")

        if cuda:
            torch.cuda.synchronize()

    if not len(output) > 0:
        print('No detections were made')
        exit()

    # Convert the coordinates to img coord without gray space padding
    im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())
    scaling_factor = torch.min(in_dim / im_dim_list, 1)[0].view(-1, 1)

    output[:, [1, 3]] -= (in_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
    output[:, [2, 4]] -= (in_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2
    output[:, 1:5] /= scaling_factor

    t_output_processing = time.time()

    # Restrict bboxes with vertices outside the image
    for idx in range(output.shape[0]):
        output[idx, [1, 3]] = torch.clamp(output[idx, [1, 3]], min=0.0,
                                        max=im_dim_list[idx, 0])
        output[idx, [2, 4]] = torch.clamp(output[idx, [2, 4]], min=0.0,
                                        max=im_dim_list[idx, 1])

    def write(x, results):
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        img = results[int(x[0])]
        cls = int(x[-1])
        color = random.choice(colors)
        label = "{0}".format(classes[cls])
        cv2.rectangle(img, c1, c2,color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2,color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
        return img

    t_draw = time.time()
    list(map(lambda x: write(x, loaded_img), output))

    det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det,x.split("/")[-1]))
    list(map(cv2.imwrite, det_names, loaded_img))

    t_end = time.time()

    print("SUMMARY")
    print("----------------------------------------------------------")
    print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
    print()
    print("{:25s}: {:2.3f}".format("Reading addresses", t_load_batch - t_read_dir))
    print("{:25s}: {:2.3f}".format("Loading batch", t_detection- t_load_batch))
    print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", t_output_processing - t_detection))
    print("{:25s}: {:2.3f}".format("Output Processing", t_draw - t_output_processing))
    print("{:25s}: {:2.3f}".format("Drawing Boxes", t_end - t_draw))
    print("{:25s}: {:2.3f}".format("Average time_per_img", (t_end - t_load_batch)/len(imlist)))
    print("----------------------------------------------------------")
