Hi, 

Codes of training and models come from opening source code:https://github.com/eriklindernoren/PyTorch-YOLOv3 (Train/py, Models.py ) 

Checkpoints: I had trained 500 times for detection. If you want to try the mask-detection , you can use the file in checkpoints.

Utils: Including the files of dataset and image processing. 

Weight: Here are yolov3 trained weights and darknet53 trained weights. It can be downloaded from official website.

Data: The file is label of mask: only 2 categories.

Models.py the structure of mask-detection 

Train/py you can use your dataset to train it. But you should prepare your dataset with your category.

cam_detect.py Real-time camera mask detection, after training, please change the path that you have save the weights.

-Eden
