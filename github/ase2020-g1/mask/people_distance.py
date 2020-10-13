# importing the required packages.
#from ctypes import *
import math
import random
#import os
#import cv2
import numpy as np
#import time
#import darknet
#from itertools import combinations

def check_distance(pointX):
    """
    :param:
    pointX = (x1,x2,y1,y2)
    :return:
    min_distance : Minimum distance (最小の距離)
    """
    point_x = np.array(pointX[1])-np.array(pointX[0])
    point_y = np.array(pointX[3])-np.array(pointX[2])
    #print('pointX', point_x)
    #print('pointY', point_y)
    midpointX, midpointY = point_x/2, point_y/2 # The central point of each person (各人物の中心点)
    #midpointX, midpointY = (np.array(pointX[1])-np.array(pointX[0]))/2, (np.array(pointX[3])-np.array(pointX[2]))/2 # The central point of each person (各人物の中心点)
    #print(midpointX, midpointY)
    #areaXY = point_x * point_y
    #print('areaXY', areaXY)
    #lineXY = np.sqrt(areaXY)
    #print('lineXY',lineXY)

    min_distance = 0
    distance = [] # Initialize the place to store the distance (距離を格納する所を初期化)
    for i in range(midpointX.shape[0]-1):
       for j in range(midpointX.shape[0]):
           if i == j or i > j:
               continue
           #print(j)
           dx, dy = midpointX[i]-midpointX[j], midpointY[i]-midpointY[j] # Differential coordinates for finding the distance between each person (各人物間の距離を求めるための座標差分)
           #print(dx, dy)
           #tmp = get_2distance(dx, dy) # Calculates the Euclidean distance (ユークリッド距離の計算)
           #D = (lineXY[i]-lineXY[j]) * 519.2640429 - 519.2640429
           #D = (lineXY[i]-lineXY[j]) * 207.7056172 - 207.7056172
           #print('D', D)
           dst = math.sqrt(dx**2 + dy**2) # Calculates the Euclidean distance (ユークリッド距離の計算)
           #dst = math.sqrt(dx**2 + dy**2 + D**2)
           distance.append(dst)
    #print('distance =\n', distance)
    min_distance = distance[np.argmin(distance)] # Get the minimum distance (最小の距離を取得する)
    return min_distance

##############テスト用#####
# pointx1 = []
# pointx2 = []
# pointy1 = []
# pointy2 = []
# for h in range(3):
#     pointx1.append(np.random.randint(1, 10))
#     pointx2.append(np.random.randint(10, 20))
#     pointy1.append(np.random.randint(1, 10))
#     pointy2.append(np.random.randint(10, 20))
# pointY = (pointx1, pointx2, pointy1, pointy2)
# print(pointY)
# min_distance = check_distance(pointY)
# print(min_distance)
#####################


# def get_2distance(p1, p2):  # Calculate Euclidean Distance between two points (2点間のユークリッド距離の計算)
#     """
#     :param:
#     p1, p2 = two points for calculating Euclidean Distance (ユークリッド距離の対象点)
#     :return:
#     dst = Euclidean Distance between two 2d points (2次元点間のユークリッド距離)
#     """
#     dst = math.sqrt(p1**2 + p2**2)
#     return dst




# def combinations_count(n, r):
#     return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))

# def convertBack(x, y, w, h):  # Converts center coordinates to rectangle coordinates (中心座標を矩形座標に変換)
#     """
#     :param:
#     x, y = midpoint of bbox (中心座標)
#     w, h = width, height of the bbox (ボックスの大きさ)
#     :return:
#     xmin, ymin, xmax, ymax
#     """
#     xmin = int(round(x - (w / 2)))
#     xmax = int(round(x + (w / 2)))
#     ymin = int(round(y - (h / 2)))
#     ymax = int(round(y + (h / 2)))
#     return xmin, ymin, xmax, ymax

# def check_distance(detections, img):
#     """
#     :param:
#     detections = total detections in one frame (1フレーム内の総検出数)
#     img = image from detect_image method of darknet (ダークネットの検出メソッドからの画像)
#     :return:
#     check_list = 0 or 1    if OK(LED-light_GREEN):0, if NO(LED-light_RED):1
#     """
#     #================================================================
#     # 3.1 Purpose : Filter out Persons class from detections and get
#     #           bounding box centroid for each person detection. (検出されたクラスをフィルタリングし，各人物のバウンディングボックスの重心を取得する)
#     #================================================================
#     if len(detections) > 0:                         # At least 1 detection in the image and check detection presence in a frame (画像内で少なくとも1つの検出を行い，フレーム内の検出の有無確認)
#         centroid_dict = dict()                      # Function creates a dictionary and calls it centroid_dict (辞書の関数を作成)
#         objectId = 0                                # We inialize a variable called ObjectId and set it to 0 (変数IDを作成し，初期値0とする)
#         for detection in detections:                # In this if statement, we filter all the detections for persons only (人物のみにフィルタリングを行う)
#             # Check for the only person name tag (各人物にタグをつける)
#             name_tag = str(detection[0].decode())   # Coco file has string of all the names (検出されたクラス内から人物だけを取得する)
#             if name_tag == 'person':
#                 x, y, w, h = detection[2][0],\
#                             detection[2][1],\
#                             detection[2][2],\
#                             detection[2][3]         # Store the center points of the detections (検出したものの中心座標，ボックスの大きさを格納する)
#                 xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))   # Convert from center coordinates to rectangular coordinates, We use floats to ensure the precision of the BBox (中心座標を矩形座標に変換)
#                 # Append center point of bbox for persons detected. (検出された人のボックスの中心点を追加する)
#                 centroid_dict[objectId] = (int(x), int(y), xmin, ymin, xmax, ymax) # Create dictionary of tuple with 'objectId' as the index center points and bbox (各人物の中心座標やボックスサイズの情報をcentroid_dictの辞書に格納する)
#                 objectId += 1 #Increment the index for each detection (IDをインクリメント)
#     #=================================================================#
#     #=================================================================
#     # 3.2 Purpose : Determine which person bbox are close to each other (各人物間の距離が近いかどうかを判別する)
#     #=================================================================
#         check_list = 0 # OK(LED-light_GREEN):0, NO(LED-light_RED):1
#         for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2): # Get all the combinations of close detections, #List of multiple items - id1 1, points 2, 1,3 (接近検出の組み合わせを全て取得する)
#             dx, dy = p1[0] - p2[0], p1[1] - p2[1]   # Check the difference between centroid x: 0, y :1 (各人物間の距離を求めるための座標差分)
#             distance = get_2distance(dx, dy)             # Calculates the Euclidean distance (ユークリッド距離の計算)
#             if distance < 75.0:                     # Set our social distance threshold - If they meet this condition then.. (社会的距離の閾値の設定)
#                 Check_list = 1
#                 break
#     #=================================================================#

#     return check_list

# def cvDrawBoxes(detections, img):
#     """
#     :param:
#     detections = total detections in one frame (1フレーム内の総検出数)
#     img = image from detect_image method of darknet (ダークネットの検出メソッドからの画像)
#     :return:
#     img with bbox
#     """
#     #================================================================
#     # 3.1 Purpose : Filter out Persons class from detections and get
#     #           bounding box centroid for each person detection. (検出されたクラスをフィルタリングし，各人物のバウンディングボックスの重心を取得する)
#     #================================================================
#     if len(detections) > 0:                         # At least 1 detection in the image and check detection presence in a frame (画像内で少なくとも1つの検出を行い，フレーム内の検出の有無確認)
#         centroid_dict = dict()                      # Function creates a dictionary and calls it centroid_dict (辞書の関数を作成)
#         objectId = 0                                # We inialize a variable called ObjectId and set it to 0 (変数IDを作成し，初期値0とする)
#         for detection in detections:                # In this if statement, we filter all the detections for persons only (人物のみにフィルタリングを行う)
#             # Check for the only person name tag (各人物にタグをつける)
#             name_tag = str(detection[0].decode())   # Coco file has string of all the names (検出されたクラス内から人物だけを取得する)
#             if name_tag == 'person':
#                 x, y, w, h = detection[2][0],\
#                             detection[2][1],\
#                             detection[2][2],\
#                             detection[2][3]         # Store the center points of the detections (検出したものの中心座標，ボックスの大きさを格納する)
#                 xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))   # Convert from center coordinates to rectangular coordinates, We use floats to ensure the precision of the BBox (中心座標を矩形座標に変換)
#                 # Append center point of bbox for persons detected. (検出された人のボックスの中心点を追加する)
#                 centroid_dict[objectId] = (int(x), int(y), xmin, ymin, xmax, ymax) # Create dictionary of tuple with 'objectId' as the index center points and bbox (各人物の中心座標やボックスサイズの情報をcentroid_dictの辞書に格納する)
#                 objectId += 1 #Increment the index for each detection (IDをインクリメント)
#     #=================================================================#
#     #=================================================================
#     # 3.2 Purpose : Determine which person bbox are close to each other (各人物間の距離が近いかどうかを判別する)
#     #=================================================================
#         red_zone_list = [] # List containing which Object id is in under threshold distance condition. (距離が一定値より近いIDリストの作成)
#         red_line_list = []
#         for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2): # Get all the combinations of close detections, #List of multiple items - id1 1, points 2, 1,3 (接近検出の組み合わせを全て取得する)
#             dx, dy = p1[0] - p2[0], p1[1] - p2[1]   # Check the difference between centroid x: 0, y :1 (各人物間の距離を求めるための座標差分)
#             distance = get_2distance(dx, dy)             # Calculates the Euclidean distance (ユークリッド距離の計算)
#             if distance < 75.0:                     # Set our social distance threshold - If they meet this condition then.. (社会的距離の閾値の設定)
#                 if id1 not in red_zone_list:
#                     red_zone_list.append(id1)       #  Add Id to a List (IDをリストに追加)
#                     red_line_list.append(p1[0:2])   #  Add points to the list (線を引くための中心点をリストに追加)
#                 if id2 not in red_zone_list:
#                     red_zone_list.append(id2)       # Same for the second id
#                     red_line_list.append(p2[0:2])
#         for idx, box in centroid_dict.items():  # dict (1(key):red(value), 2 blue)  idx - key  box - value
#             if idx in red_zone_list:   # if id is in red zone list (IDがレッドゾーンにある場合)
#                 cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (255, 0, 0), 2) # Create Red bounding boxes  #starting point, ending point size of 2 (ボックスの色を赤にする)
#             else:
#                 cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2) # Create Green bounding boxes (ボックスの色を緑にする)
#     #=================================================================#

#     #=================================================================
#     # 3.3 Purpose : Display Risk Analytics and Show Risk Indicators (リスク分析の表示とリスク指標の表示)
#     #=================================================================
#     #=================================================================#
#         text = "People at Risk: %s" % str(len(red_zone_list))           # Count People at Risk (リスクのある人物を数える)
#         location = (10,25)                                              # Set the location of the displayed text (表示されるテキストの位置の設定)
#         cv2.putText(img, text, location, cv2.FONT_HERSHEY_SIMPLEX, 1, (246,86,86), 2, cv2.LINE_AA)  # Display Text (表示テキスト)

#         for check in range(0, len(red_line_list)-1):                    # Draw line between nearby bboxes iterate through redlist items (近くにあるボックス間に線を引き，レッドリストの項目を反復処理する)
#             start_point = red_line_list[check]
#             end_point = red_line_list[check+1]
#             check_line_x = abs(end_point[0] - start_point[0])           # Calculate the line coordinates for x (線のx座標の計算)
#             check_line_y = abs(end_point[1] - start_point[1])           # Calculate the line coordinates for y (線のy座標の計算)
#             if (check_line_x < 75) and (check_line_y < 25):             # If both are We check that the lines are below our threshold distance. (両方の場合は，ラインが閾値以下であることを確認)
#                 cv2.line(img, start_point, end_point, (255, 0, 0), 2)   # Only above the threshold lines are displayed. (閾値以上の線だけが表示される)
#     #=================================================================#
#     return img

# netMain = None
# metaMain = None
# altNames = None


# def YOLO():
#     """
#     Perform Object detection
#     """
#     global metaMain, netMain, altNames
#     configPath = "./cfg/yolov4.cfg"
#     weightPath = "./yolov4.weights"
#     metaPath = "./cfg/coco.data"
#     if not os.path.exists(configPath):
#         raise ValueError("Invalid config path `" +
#                          os.path.abspath(configPath)+"`")
#     if not os.path.exists(weightPath):
#         raise ValueError("Invalid weight path `" +
#                          os.path.abspath(weightPath)+"`")
#     if not os.path.exists(metaPath):
#         raise ValueError("Invalid data file path `" +
#                          os.path.abspath(metaPath)+"`")
#     if netMain is None:
#         netMain = darknet.load_net_custom(configPath.encode(
#             "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
#     if metaMain is None:
#         metaMain = darknet.load_meta(metaPath.encode("ascii"))
#     if altNames is None:
#         try:
#             with open(metaPath) as metaFH:
#                 metaContents = metaFH.read()
#                 import re
#                 match = re.search("names *= *(.*)$", metaContents,
#                                   re.IGNORECASE | re.MULTILINE)
#                 if match:
#                     result = match.group(1)
#                 else:
#                     result = None
#                 try:
#                     if os.path.exists(result):
#                         with open(result) as namesFH:
#                             namesList = namesFH.read().strip().split("\n")
#                             altNames = [x.strip() for x in namesList]
#                 except TypeError:
#                     pass
#         except Exception:
#             pass
#     #cap = cv2.VideoCapture(0)
#     cap = cv2.VideoCapture("./Input/test5.mp4")
#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))
#     new_height, new_width = frame_height // 2, frame_width // 2
#     # print("Video Reolution: ",(width, height))

#     out = cv2.VideoWriter(
#             "./Demo/test5_output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
#             (new_width, new_height))
#     # print("Starting the YOLO loop...")

#     # Create an image we reuse for each detect
#     darknet_image = darknet.make_image(new_width, new_height, 3)
#     while True:
#         prev_time = time.time()
#         ret, frame_read = cap.read()
#         # Check if frame present :: 'ret' returns True if frame present, otherwise break the loop.
#         if not ret:
#             break

#         frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
#         frame_resized = cv2.resize(frame_rgb,
#                                    (new_width, new_height),
#                                    interpolation=cv2.INTER_LINEAR)

#         darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

#         detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
#         image = cvDrawBoxes(detections, frame_resized)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         print(1/(time.time()-prev_time))
#         cv2.imshow('Demo', image)
#         cv2.waitKey(3)
#         out.write(image)

#     cap.release()
#     out.release()
#     print(":::Video Write Completed")

# if __name__ == "__main__":
#     YOLO()