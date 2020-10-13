import json
import os
import re

from django.http import HttpResponse
from django.shortcuts import render


# Create your views here.
from DI.settings import MEDIAS_ROOT


def find_NewFile(path):
    # 获取文件夹中的所有文件
    lists = os.listdir(path)
    if not lists or len(lists) <= 0:
        return ""
    # 对获取的文件根据修改时间进行排序
    lists.sort(key=lambda x: os.path.getmtime(path + '/' + x))
    if ".DS_Store" == lists[-1]:
        lists.remove(lists[-1])
        if not lists or len(lists) <= 0:
            return ""
        return lists[-1]
    return lists[-1]


def findNewFile():
    image_name = 'txt.jpg'

    date_time = ""
    num = ""
    no_mask = ""
    distance = ""

    try:
        with open(MEDIAS_ROOT + "info.txt", "r") as f:  # 打开文件
            data = f.read()  # 读取文件
            searchObj = re.search(r'(.*)-NUM:(.*)-Nomask:(.*)-Distance:(.*)-', data, re.M | re.I)
            date_time = searchObj.group(1)
            num = searchObj.group(2)
            no_mask = searchObj.group(3)
            distance = searchObj.group(4)
    except:
        pass

    try:
        no_mask_num = int(no_mask)
        distance_num = float(distance)
        mask = int(num) - no_mask_num
    except:
        no_mask_num = 0
        distance_num = 0.0
        mask = 0

    situation = "Safe"
    if distance_num < 2.5 or no_mask_num > 0:
        situation = "Dangerous"

    date_time = date_time.replace("_", ':').replace("-", "+", 2).replace("-", " ").replace("+", "-", 2)
    data = {"f": image_name, "t": date_time, "num": num,
            "mask": mask, "no_mask": no_mask, "distance": distance,
            "situation": situation}
    return data


def allPage(request):
    info_dict = findNewFile()
    return render(request, 'all.html', {'data': info_dict})


def refresh(request):
    key = request.POST.get("key", '')
    print(key)
    import requests
    # url = "http://"
    # res = requests.get(url)
    # print(res.text)
    data = findNewFile()
    return HttpResponse(json.dumps(data))
