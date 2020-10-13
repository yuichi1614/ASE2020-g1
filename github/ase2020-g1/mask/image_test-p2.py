# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import with_statement
from __future__ import absolute_import
from models import Darknet
from utils.utils import load_classes,non_max_suppression_output, non_max_suppression
import argparse
import os
import torch
import numpy as np
from torch.autograd import Variable
import cv2
import datetime
import schedule
import threading
import time
import pickle
import time
import paramiko  
from scp import SCPClient
from io import open



u"""
def upload_img(img_name, remote_path="/eden",file_path="testing/output/images/"):
    host = "172.21.39.222"  # 服务器ip地址
    port = 22  # 端口号
    username = "tensor"  # ssh 用户名
    password = "tensor"  # 密码

    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    ssh_client.connect(host, port, username, password)
    scpclient = SCPClient(ssh_client.get_transport(), socket_timeout=15.0)
    local_path = file_path + img_name
    try:
        scpclient.put(local_path, remote_path)
    except FileNotFoundError as e:
        print(e)
        print("系统找不到指定文件" + local_path)
    else:
        print("文件上传成功")
    ssh_client.close()"""

u"""
from __future__ import with_statement
from __future__ import absolute_import
"""

u"""
idef ssh_scp_put(ip, port, user, password, local_file, remote_file):

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip, port, user, password)

    sftp = ssh.open_sftp()
    sftp.put(local_file, remote_file)
"""
def mask_catch(input,output):

        parser = argparse.ArgumentParser()
        parser.add_argument(u"--input_file_path", type=unicode, default=input, help=u"path to images directory")
        parser.add_argument(u"--output_path", type=unicode, default=output, help=u"output image directory")
        parser.add_argument(u"--model_def", type=unicode, default=u"data/yolov3_mask.cfg", help=u"path to model definition file")
        parser.add_argument(u"--weights_path", type=unicode, default=u"checkpoints/yolov3_ckpt_499.pth", help=u"path to weights file")
        parser.add_argument(u"--class_path", type=unicode, default=u"data/mask_dataset.names", help=u"path to class label file")
        parser.add_argument(u"--conf_thres", type=float, default=0.8, help=u"object confidence threshold")
        parser.add_argument(u"--nms_thres", type=float, default=0.3, help=u"iou thresshold for non-maximum suppression")
        parser.add_argument(u"--frame_size", type=int, default=416, help=u"size of each image dimension")

        opt = parser.parse_args()
        # Output directory
        os.makedirs(opt.output_path, exist_ok=True)

        # checking for GPU
        device = torch.device(u"cuda" if torch.cuda.is_available() else u"cpu")

        # Set up model
        model = Darknet(opt.model_def, img_size=opt.frame_size).to(device)

        # loading weights
        if opt.weights_path.endswith(u".weights"):
            model.load_darknet_weights(opt.weights_path)  # Load weights
        else:
            model.load_state_dict(torch.load(opt.weights_path))  # Load checkpoints

        # Set in evaluation mode
        model.eval()

        # Extracts class labels from file
        classes = load_classes(opt.class_path)

        # ckecking for GPU for Tensor
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        print u"\nPerforming object detection:"

        # for text in output
        t_size = cv2.getTextSize(u" ", cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]

        for imagename in os.listdir(opt.input_file_path):

            print u"\n"+imagename+u"_______"
            image_path = os.path.join(opt.input_file_path, imagename)
            print image_path
            # frame extraction
            org_img = cv2.imread(image_path)

            # Original image width and height
            i_height, i_width = org_img.shape[:2]

            # resizing => [BGR -> RGB] => [[0...255] -> [0...1]] => [[3, 416, 416] -> [416, 416, 3]]
            #                       => [[416, 416, 3] => [416, 416, 3, 1]] => [np_array -> tensor] => [tensor -> variable]

            # resizing to [416 x 416]

            # Create a black image
            x = y = i_height if i_height > i_width else i_width

            # Black image
            img = np.zeros((x, y, 3), np.uint8)

            # Putting original image into black image
            start_new_i_height = int((y - i_height) / 2)
            start_new_i_width = int((x - i_width) / 2)

            img[start_new_i_height: (start_new_i_height + i_height) ,start_new_i_width: (start_new_i_width + i_width) ] = org_img

            #resizing to [416x 416]
            img = cv2.resize(img, (opt.frame_size, opt.frame_size))

            # [BGR -> RGB]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # [[0...255] -> [0...1]]
            img = np.asarray(img) / 255
            # [[3, 416, 416] -> [416, 416, 3]]
            img = np.transpose(img, [2, 0, 1])
            # [[416, 416, 3] => [416, 416, 3, 1]]
            img = np.expand_dims(img, axis=0)
            # [np_array -> tensor]
            img = torch.Tensor(img)

            # plt.imshow(img[0].permute(1, 2, 0))
            # plt.show()

            # [tensor -> variable]
            img = Variable(img.type(Tensor))

            # Get detections
            with torch.no_grad():
                detections = model(img)

            detections = non_max_suppression_output(detections, opt.conf_thres, opt.nms_thres)

            # print(detections)

            # For accommodate results in original frame
            mul_constant = x / opt.frame_size

            #We should set a variable for the number of nomask people. i is the variable
            i=0

            # For each detection in detections
            for detection in detections:
                if detection is not None:

                    print u"{0} Detection found".format(len(detection))
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:

                        # Accommodate bounding box in original frame
                        x1 = int(x1 * mul_constant - start_new_i_width)
                        y1 = int(y1 * mul_constant - start_new_i_height)
                        x2 = int(x2 * mul_constant - start_new_i_width)
                        y2 = int(y2 * mul_constant - start_new_i_height)

                        # Bounding box making and setting Bounding box title
                        if (int(cls_pred) == 0):

                            # WITH_MASK
                            cv2.rectangle(org_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        else:
                            #WITHOUT_MASK
                            i+=1
                            cv2.rectangle(org_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                        cv2.putText(org_img, classes[int(cls_pred)]+u": %.2f" %conf, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                                    [225, 255, 255], 2)

            u"""------------Ready to save!-----------------"""
            import time
            now = time.strftime(u"%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))

            #num is the number of people
            num=len(detection)
            #na=now + '-' + 'NUM:%d'%num +'-'+ 'Nom:%d'%i+'-'+'.jpg'


            u"""------------txt_save-----------------"""


        

            u"""------------image_save-----------------"""
            na=u'result.jpg'
            out_filepath = os.path.join(opt.output_path,na)
            cv2.imwrite(out_filepath,org_img)#org_img is final result with frames

            #naa = now + '-' + 'NUM:%d' % num + '-' + 'Nom:%d' % i
            #ssh_scp_put('172.21.39.222',22,'tensor','tensor',out_filepath,'/home/tensor/eden/%s.jpg'%naa)
            #upload_img(na)
            #os.remove(out_filepath)

            signal=1       #we first set signal only 1

            if i==0:
                signal=0

            print u"Signal is ",signal
            print u"Finish to save!!!"

            msg=now + u'-' + u'NUM:%d'%num +u'-'+ u'Nomask:%d'%i+u'-'
            nam=u'info.txt'
            full_path = os.path.join(opt.output_path,nam)
            print u"----------------"
            file = open(full_path, u'w')
            file.write(msg)



        cv2.destroyAllWindows()
        return signal

#schedule.every(2).seconds.do(mask_catch,opt)
u"""
while True:
    schedule.run_pending()
    time.sleep(1)
"""


