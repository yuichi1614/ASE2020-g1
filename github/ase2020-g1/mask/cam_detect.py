from __future__ import division

from models import Darknet
from utils.utils import load_classes,non_max_suppression_output

import argparse

import time
import torch
import numpy as np
from torch.autograd import Variable
from datetime import datetime

import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/yolov3_mask.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_35.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/mask_dataset.names", help="Detection lables' path")
    parser.add_argument("--conf_thres", type=float, default=0.9, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--frame_size", type=int, default=416, help="size of each image dimension")

    opt = parser.parse_args()
    print(opt)
    
    '''Depending on situation our group meet, we only can use CPU.And so you should change the image_size as input, like 64/96/128/160'''
    
    if(torch.cuda.is_available()):
        print("Running on GPU")
    else:
        print("Running on CPU")

    # checking for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.frame_size).to(device)

    # loading weights
    if opt.weights_path.endswith(".weights"):
        model.load_darknet_weights(opt.weights_path)  # Load weights
    else:
        model.load_state_dict(torch.load(opt.weights_path,map_location='cpu'))  # here is configuration of cpu, you should add this code.

    # Set in evaluation mode
    model.eval()

    # Extracts class labels from file
    classes = load_classes(opt.class_path)

    # ckecking for GPU for Tensor
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # camara capture
cap = cv2.VideoCapture(0) #0 open your camara, if we have USB-camara, so we need to use 1!
    assert cap.isOpened(), 'Cannot capture source'

    print("\Real-Time: mask detection:")

    # Video feed dimensions
    _, frame = cap.read()
    v_height, v_width = frame.shape[:2]

    # For a black image
    x = y = v_height if v_height > v_width else v_width

    # Putting original image into black image
    start_new_i_height = int((y - v_height) / 2)
    start_new_i_width = int((x - v_width) / 2)

    # For accommodate results in original frame
    mul_constant = x / opt.frame_size

    # for text in output
    t_size = cv2.getTextSize(" ", cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]

    frames = fps = 0
    start = time.time()

    while _:

        # frame extraction
        _, org_frame = cap.read()
       
        # Black image
        frame = np.zeros((x, y, 3), np.uint8)

        frame[start_new_i_height: (start_new_i_height + v_height) ,start_new_i_width: (start_new_i_width + v_width) ] = org_frame

        #resizing to [416x 416]
        frame = cv2.resize(frame, (opt.frame_size, opt.frame_size))
        # [BGR -> RGB]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # [[0...255] -> [0...1]]
        frame = np.asarray(frame) / 255
        # [[3, 416, 416] -> [416, 416, 3]]
        frame = np.transpose(frame, [2, 0, 1])
        # [[416, 416, 3] => [416, 416, 3, 1]]
        frame = np.expand_dims(frame, axis=0)
        # [np_array -> tensor]
        frame = torch.Tensor(frame)

        # plt.imshow(frame[0].permute(1,2,0))
        # plt.show() if you want to show picture, you can use this code when detection start/

        # [tensor -> variable]
        frame = Variable(frame.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(frame)
        detections = non_max_suppression_output(detections, opt.conf_thres, opt.nms_thres)

        # For each detection in detections
        """These codes: I choose red box and green box to figure out mask or not. """
        detection = detections[0]
        if detection is not None:

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:

                # Accommodate bounding box in original frame
                x1 = int(x1 * mul_constant - start_new_i_width)
                y1 = int(y1 * mul_constant - start_new_i_height)
                x2 = int(x2 * mul_constant - start_new_i_width)
                y2 = int(y2 * mul_constant - start_new_i_height)

                # Bounding box making and setting Bounding box title
                if (int(cls_pred) == 0):
                    # WITH_MASK
                    cv2.rectangle(org_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)#(0,255,0) G,B,R
                else:
                    #WITHOUT_MASK
                    cv2.rectangle(org_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)#(0,0,255) G,B,R

                cv2.putText(org_frame, classes[int(cls_pred)]+ ": %.2f" % conf, (x1, y1 + t_size[1]),
                            cv2.FONT_HERSHEY_PLAIN, 1,
                            [225, 255, 255], 2)

        # CURRENT TIME SHOWING
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")

        # FPS PRINTING
        cv2.rectangle(org_frame, (0, 0), (175, 20), (0, 0, 0), -1)
        cv2.putText(org_frame, current_time + " FPS : %3.2f" % (fps), (0, t_size[1] + 2),
                    cv2.FONT_HERSHEY_PLAIN, 1,
                    [255, 255, 255], 1)

        frames += 1
        fps = frames / (time.time() - start)

        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN-5, cv2.WINDOW_FULLSCREEN)

        cv2.imshow('frame', org_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
