import tensorflow as tf
from keras.models import load_model
import numpy as np
import cv2
from keras import Model
from keras.layers import Dense

# Custom model
abt_model = load_model('Model/apple_banana_tomato_model')

abt_output = Dense(3, activation='softmax')
abt_output = abt_output(abt_model.layers[-2].output)

abt_input = abt_model.input
abt_model = Model(inputs=abt_input, outputs=abt_output)

for layer in abt_model.layers[:-1]:
  layer.trainable = False

# Compile Model
abt_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

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

	cv2.putText(new_img, " Apple {:.2f} %  Banana {:.2f} %  Tomato {:.2f} %".format(score[0]*100, score[1]*100, score[2]*100) , (20, 65), cv2.FONT_HERSHEY_SIMPLEX,
	                          0.6, (0, 255, 0), 2)

	cv2.imshow('frame', new_img)
	# cv2.imshow('mask', red_mask )

	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
