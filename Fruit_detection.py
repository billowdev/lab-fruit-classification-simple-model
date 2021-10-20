import tensorflow as tf
from keras.models import load_model
import numpy as np
import cv2

model=load_model("./AppleBananaTomato")

cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()
	width = int(cap.get(3))
	height = int(cap.get(4))

	############################  Model  #####################################

	new_img = cv2.resize(frame, (frame.shape[1] // 1, frame.shape[0] // 1))
	resized = cv2.resize(new_img, (224, 224)) 
	img_array = tf.keras.preprocessing.image.img_to_array(resized) 
	img_array = tf.expand_dims(img_array, 0) #ขยายมิติภาพฟิตกับโมดล
	predictions = model.predict(img_array) # ทำนายบน ROI (Region of Interest)
	score = tf.nn.softmax(predictions[0]) # ผลลัพธ์
	label = np.argmax(score) # หาค่าสูงสุด

	if label == 0:
		new_img = cv2.putText(new_img, "Apple", (55, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) # แสดงข้อความ "mask"
		# print("\n\n Apple")

	elif label == 1:
		new_img = cv2.putText(new_img, "Banana", (55, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) # แสดงข้อความ "mask"
		# print("\n\n Banana")

	elif label == 2:
		new_img = cv2.putText(new_img, "Tomato", (55, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) # แสดงข้อความ "mask"
		# print("\n\n Tomato")

	else:
		new_img = cv2.putText(new_img, "Not Found", (55, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) # แสดงข้อความ "mask"


	############################ Color Detection ############################

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# Red Mask
	lower_red = np.array([160,50,20])
	# upper_red = np.array([180, 255, 255])
	upper_red = np.array([179,255,255])
	
	lower_raw = np.array([0,50,20])
	upper_raw = np.array([80,255,255])

	mask_red = cv2.inRange(hsv, lower_red, upper_red)
	mask_raw = cv2.inRange(hsv, lower_red, upper_red)

	red_mask = mask_red 

	# Yellow Mask
	lower_yellow = np.array([22, 93, 0])
	upper_yellow = np.array([45, 255, 255])

	yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)


	redcnts = cv2.findContours(red_mask.copy(),
                              cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)[-2]

	yellowcnts = cv2.findContours(yellow_mask.copy(),
                              cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)[-2]

	if len(redcnts)>0:
		red_area = max(redcnts, key=cv2.contourArea)
		(xg,yg,wg,hg) = cv2.boundingRect(red_area)
		cv2.rectangle(frame,(xg,yg),(xg+wg, yg+hg),(255,0,0),2)

	if len(yellowcnts)>0:
		yellow_area = max(yellowcnts, key=cv2.contourArea)
		(xg,yg,wg,hg) = cv2.boundingRect(yellow_area)
		cv2.rectangle(frame,(xg,yg),(xg+wg, yg+hg),(255,0,0),2)

		# print("\n\n Not Found")

	cv2.imshow('frame', new_img)
	cv2.imshow('mask', red_mask )

	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()