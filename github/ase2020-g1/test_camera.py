# -*- coding:utf-8 -*-
import time
import RPi.GPIO as GPIO
import os
import subprocess
import showStat as show
import LEDproc as led
sensor_pin = 26

def takeSendPic():
    path = os.path.dirname(os.path.abspath(__file__))
    cmd = '{}/takeSendPic.sh'.format(path)
    subprocess.call(cmd, shell=True)
    res = show.main("../Pictures/pic.png")
    return res

if __name__ == '__main__':
    led.setup_LED()
    try:
	takeSendPic()
    except KeyboardInterrupt:
	led.destroy()
