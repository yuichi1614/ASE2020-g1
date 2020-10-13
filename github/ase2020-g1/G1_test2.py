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
BLUEPIN = 5
YELLOWPIN = 4


GPIO.setmode(GPIO.BCM)

MAX_UNCHANGE_COUNT = 100

STATE_INIT_PULL_DOWN = 1
STATE_INIT_PULL_UP = 2
STATE_DATA_FIRST_PULL_DOWN = 3
STATE_DATA_PULL_UP = 4
STATE_DATA_PULL_DOWN = 5

def read_dht11_dat():
    GPIO.setup(DHTPIN, GPIO.OUT)
    GPIO.output(DHTPIN, GPIO.HIGH)
    time.sleep(0.05)
    GPIO.output(DHTPIN, GPIO.LOW)
    time.sleep(0.02)
    GPIO.setup(DHTPIN, GPIO.IN, GPIO.PUD_UP)

    unchanged_count = 0
    last = -1
    data = []
    while True:
        current = GPIO.input(DHTPIN)
        data.append(current)
        if last != current:
            unchanged_count = 0
            last = current
        else:
            unchanged_count += 1
            if unchanged_count > MAX_UNCHANGE_COUNT:
                break

    state = STATE_INIT_PULL_DOWN

    lengths = []
    current_length = 0

    for current in data:
        current_length += 1

        if state == STATE_INIT_PULL_DOWN:
            if current == GPIO.LOW:
                state = STATE_INIT_PULL_UP
            else:
                continue
        if state == STATE_INIT_PULL_UP:
            if current == GPIO.HIGH:
                state = STATE_DATA_FIRST_PULL_DOWN
            else:
                continue
        if state == STATE_DATA_FIRST_PULL_DOWN:
            if current == GPIO.LOW:
                state = STATE_DATA_PULL_UP
            else:
                continue
        if state == STATE_DATA_PULL_UP:
            if current == GPIO.HIGH:
                current_length = 0
                state = STATE_DATA_PULL_DOWN
            else:
                continue
        if state == STATE_DATA_PULL_DOWN:
            if current == GPIO.LOW:
                lengths.append(current_length)
                state = STATE_DATA_PULL_UP
            else:
                continue
    if len(lengths) != 40:
        print "Data not good, skip"
        return False

    shortest_pull_up = min(lengths)
    longest_pull_up = max(lengths)
    halfway = (longest_pull_up + shortest_pull_up) / 2
    bits = []
    the_bytes = []
    byte = 0

    for length in lengths:
        bit = 0
        if length > halfway:
            bit = 1
        bits.append(bit)
    print "bits: %s, length: %d" % (bits, len(bits))
    for i in range(0, len(bits)):
        byte = byte << 1
        if (bits[i]):
            byte = byte | 1
        else:
            byte = byte | 0
        if ((i + 1) % 8 == 0):
            the_bytes.append(byte)
            byte = 0
    print the_bytes
    checksum = (the_bytes[0] + the_bytes[1] + the_bytes[2] + the_bytes[3]) & 0xFF
    if the_bytes[4] != checksum:
        print "Data not good, skip"
        return False

    return the_bytes[0], the_bytes[2]

def main():
    print "Raspberry Pi wiringPi DHT11 Temperature test program\n"
    while True:
        result = read_dht11_dat()
        if result:
            humidity, temperature = result
            print "humidity: %s %%,  Temperature: %s C" % (humidity, temperature)
            thi = calc_THI(temperature, humidity)
            print("thi = " + str(thi))

            if (thi <= 75.0):
                LED_onoff(BLUEPIN)
            elif (thi < 85.0):
                LED_onoff(YELLOWPIN)
            else:
                LED_onoff(REDPIN)

#def destroy():
#    GPIO.cleanup()

# calculate thi
def calc_THI(temp, humi):
    """
    thi1 = 0.81 * temp
    thi2 = 0.01 * humi
    thi3 = 0.99 * temp
    thi4 = thi3 - 14.3
    thi5 = thi2 * thi4 + 46.3
    thi = thi1 + thi4
    """
    thi = 0.81 * temp + 0.01 * humi * (0.99 * temp - 14.3) + 46.3
    return thi

# LED light process
def setup_LED():
    GPIO.setwarnings(False)
    #set the gpio modes to BCM numbering
    GPIO.setmode(GPIO.BCM)
    #set LEDPIN's mode to output,and initial level to LOW(0V)
    GPIO.setup(REDPIN,GPIO.OUT,initial=GPIO.LOW)
    GPIO.setup(BLUEPIN,GPIO.OUT,initial=GPIO.LOW)
    GPIO.setup(YELLOWPIN,GPIO.OUT,initial=GPIO.LOW)

def LED_onoff(pin):
    GPIO.output(pin,GPIO.HIGH)
    time.sleep(1)
    GPIO.output(pin,GPIO.LOW)

def destroy():
    #turn off LED
    GPIO.output(REDPIN,GPIO.LOW)
    GPIO.output(BLUEPIN,GPIO.LOW)
    GPIO.output(YELLOWPIN,GPIO.LOW)

    #release resource
    GPIO.cleanup()

if __name__ == '__main__':
    setup_LED()
    try:
        main()
    except KeyboardInterrupt:
        destroy()
