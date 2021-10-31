import tensorflow as tf
from keras.models import load_model
import numpy as np
import cv2

abt_model = load_model("Model\custom_abtmodel")

cap = cv2.VideoCapture(0)

def abt_predict(image):
	"""
	ฟังก์ชันสำหรับการทำนายภาพผลไม้ แอปเปื้ล กล้วย และมะเขือเทศ
	Prediction Function 
	args: image (<class 'numpy.ndarray'>)
	"""
	resized = cv2.resize(image, (224, 224))
	img_array = tf.keras.preprocessing.image.img_to_array(resized)
	img_array = tf.expand_dims(img_array, 0)
	pred = abt_model.predict(img_array)
	score = pred[0]

	print("prediction score: Apple {:.2f} %".format(score[0]*100))
	print("prediction score: Banana {:.2f} %".format(score[1]*100))
	print("prediction score: Tomato {:.2f} %".format(score[2]*100))
	print("----------")

	return score


while True:
	ret, frame = cap.read()
	width = int(cap.get(3))
	height = int(cap.get(4))
	new_img = cv2.resize(frame, (frame.shape[1] // 1, frame.shape[0] // 1))
	score = abt_predict(new_img)
	result = np.argmax(score)

	if result == 0:
		new_img = cv2.putText(new_img, " Apple {:.2f} % \n Banana {:.2f} % \n Tomato {:.2f} %".format(score[0]*100, score[1]*100, score[2]*100) , (55, 65), cv2.FONT_HERSHEY_SIMPLEX,
	                          0.8, (0, 255, 0), 2)

	if result == 1:
		new_img = cv2.putText(new_img, " Apple {:.2f} % \n Banana {:.2f} % \n Tomato {:.2f} %".format(score[0]*100, score[1]*100, score[2]*100) , (55, 65), cv2.FONT_HERSHEY_SIMPLEX,
	                          0.8, (0, 255, 0), 2)
	if result == 2:
		new_img = cv2.putText(new_img, " Apple {:.2f} % \n Banana {:.2f} % \n Tomato {:.2f} %".format(score[0]*100, score[1]*100, score[2]*100) , (55, 65), cv2.FONT_HERSHEY_SIMPLEX,
	                          0.8, (0, 255, 0), 2)

	cv2.imshow('frame', new_img)
	# cv2.imshow('mask', red_mask )

	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
