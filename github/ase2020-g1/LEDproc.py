##########################################################################
# Filename    : dht11.py
# Description : test for SunFoudner DHT11 humiture & temperature module
# Author      : Alan
# Website     : www.osoyoo.com
# Update      : 2017/07/06
##########################################################################
import RPi.GPIO as GPIO
import time

#DHT11 connect to BCM_GPIO14
DHTPIN = 14
REDPIN = 22
GREENPIN = 12
YELLOWPIN = 20 

GPIO.setmode(GPIO.BCM)

MAX_UNCHANGE_COUNT = 100

STATE_INIT_PULL_DOWN = 1
STATE_INIT_PULL_UP = 2
STATE_DATA_FIRST_PULL_DOWN = 3
STATE_DATA_PULL_UP = 4
STATE_DATA_PULL_DOWN = 5

# LED light process
def setup_LED():
    GPIO.setwarnings(False)
    #set the gpio modes to BCM numbering
    GPIO.setmode(GPIO.BCM)
    #set LEDPIN's mode to output,and initial level to LOW(0V)
    GPIO.setup(REDPIN,GPIO.OUT,initial=GPIO.LOW)
    GPIO.setup(GREENPIN,GPIO.OUT,initial=GPIO.LOW)
    GPIO.setup(YELLOWPIN,GPIO.OUT,initial=GPIO.LOW)

def LED_on(pin):
    GPIO.output(pin,GPIO.HIGH)

def LED_off():
	GPIO.output(REDPIN, GPIO.LOW)
	GPIO.output(GREENPIN, GPIO.LOW)
	GPIO.output(YELLOWPIN, GPIO.LOW)

def destroy():
    #turn off LED
	LED_off()

    #release resource
	GPIO.cleanup()
