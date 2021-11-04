from ctypes import windll
from tkinter import *
from tkinter import filedialog
import tkinter as tk
import os
import cv2
from PIL import Image, ImageTk # pip install python3-pil.imagetk
import tensorflow as tf
from keras.models import load_model
import numpy as np
from contextlib import suppress

from keras import Model
from keras.layers import Dense

windll.shcore.SetProcessDpiAwareness(1)

# Custom model
abt_model = load_model("./Model/apple_banana_tomato_model")

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

gui = Tk()
gui.title("Image Browser")
gui.geometry("1080x650")

# input_imgage = None

def show_image():
	global fln
	fln = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Image File", filetypes=(("JPG File", "*.jpg"), ("PNG File", "*.png"), ("All Files", "*.")))
	img = Image.open(fln)
	img.thumbnail((400, 400)) # dimention image resize
	img = ImageTk.PhotoImage(img)
	lbl.configure(image=img)
	lbl.image = img


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
	global score
	score = pred[0]
	# print(abt_model.predict(img_array/255.0, batch_size=32, verbose=0))

	label = np.argmax(pred[0])
	if label == 0:
		t0 = "ผลลัพธ์การทำนาย คือ ... Apple"
	if label == 1:
		t0 = "ผลลัพธ์การทำนาย คือ ... Banana"
	else:
		t0 = "ผลลัพธ์การทำนาย คือ ... Tomato"
	
	t1 = "prediction score: Apple {:.2f} %".format(score[0]*100)
	t2 = "prediction score: Banana {:.2f} %".format(score[1]*100)
	t3 = "prediction score: Tomato {:.2f} %".format(score[2]*100)

	alltext = f"{t0} \n {t1} \n {t2} \n {t3} \n -------------------"

	return alltext

def prediction():
	input_image = cv2.imread(fln)
	res = abt_predict(input_image)
	res_text.insert(tk.END, res)


def clear_text():
	lbl.configure(image="")
	res_text.delete('1.0', END)

# Create text widget and specify size.
res_text = Text(gui, height = 10, width = 52)
res_text.pack(side=BOTTOM)

frm = Frame(gui)
frm.pack(side=BOTTOM, padx=15, pady=15)

lbl = Label(gui)
lbl.pack()

# Browse Image
btn = Button(frm, text="Browse Image", command=show_image)
btn.pack(side=tk.LEFT, padx=10)

# Prediction Button
btn_prediction = Button(frm, text="Prediction", command=prediction)
btn_prediction.pack(side=tk.LEFT, padx=10)

# Clear Button
btn_clear = Button(frm, text="clear", command=clear_text)
btn_clear.pack(side=tk.LEFT, padx=10)

# Exit Button
btn_exit = Button(frm, text="Exit", command=lambda: exit())
btn_exit.pack(side=tk.LEFT, padx=10)


gui.mainloop()