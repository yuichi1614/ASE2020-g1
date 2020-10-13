# -*- coding:utf-8 -*-
import time
import RPi.GPIO as GPIO
import os
import subprocess
import showStat as show
import LEDproc as led
sensor_pin = 26

ONE = 300
TWO = 120
MANY = 30

GPIO.setmode(GPIO.BCM)
GPIO.setup(sensor_pin, GPIO.IN)

A = 0

def main():
    res = 0
    take_time = time.time()

    while True:
        if( GPIO.input(sensor_pin) == 0 ):
            print ("OFF")
        else:
            print("ON")
            res = takeSendPic()
            teke_time = time.time()
            continue
	
        time.sleep(0.5)
        
        now_time = time.time()
        if res == 1:
            if now_time - take_time >= ONE:
                res = takeSendPic()
                take_time = time.time()
        elif res == 2:
            if now_time - take_time >= TWO:
                res = takeSendPic()
                take_time = time.time()
        elif res == 3:
            if now_time - take_time >= MANY:
                res = takeSendPic()
                take_time = time.time()

            
		'''
        #def call_shell():
        path = os.path.dirname(os.path.abspath(__file__))
		#cmd = '{}/takeSendPic.sh {}'.format(path, 10)
        cmd = '{}/takeSendPic.sh'.format(path)
        subprocess.call(cmd, shell=True)
        res = show.main("../Pictures/pic.png")

        if res == 0:
	    time.sleep(ONE)
        elif res == 1:
            time.sleep(ONE)
        elif res == 2:
            time.sleep(TWO)
        elif res == 3:
            time.sleep(MANY)
        '''

def takeSendPic():
    path = os.path.dirname(os.path.abspath(__file__))
    cmd = '{}/takeSendPic.sh'.format(path)
    subprocess.call(cmd, shell=True)
    res = show.main("../Pictures/pic.png")
    return res

if __name__ == '__main__':
    led.setup_LED()
    try:
	main()
    except KeyboardInterrupt:
	led.destroy()
"""
    if A == 0:
	
    elif A == 1:
	time.sleep(600)
    elif A == 2:
        time.sleep(300)
    elif A >= 3:
        time.sleep(60)
"""
