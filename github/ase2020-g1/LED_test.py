import LEDproc as led
import time

led.setup_LED()
cnt = 0
while True:
	if cnt % 3 == 0:
		print("turn on GREEN!!")
		led.LED_on(led.GREENPIN)
	elif cnt % 3 == 1:
		print("turn on RED!!")
		led.LED_on(led.REDPIN)
	else:
		print("turn on YELLOW!!")
		led.LED_on(led.YELLOWPIN)
	time.sleep(0.1)
	cnt += 1
	led.LED_off()
