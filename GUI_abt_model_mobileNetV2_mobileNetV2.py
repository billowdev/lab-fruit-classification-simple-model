from ctypes import windll # ใช้ในการช่วยให้ GUI ไม่เบลอ
from tkinter import * # ส่วนของ ตัวที่ใช้ออกแบบ GUI
from tkinter import filedialog # ส่วนที่ใช้ในการ Browse file ภาพ
import tkinter as tk # ส่วนของ ตัวที่ใช้ออกแบบ GUI

from tkinter import ttk

import os # ใช้ในการเข้าถึงpath ไฟล์
import cv2 # ใช้ในการอ่านภาพ
from PIL import Image, ImageTk #  ใช้ในการแสดงผลภาพใน GUI
import tensorflow as tf # ใช้สำหรับการเตรียมข้อมูลภาพ
from keras.models import load_model # โหลดโมเดล
import numpy as np # ใช้ในการจัดการข้อมูล array

from keras import Model # ส่วนของ keras เพื่อใช้ในการ custom โมเดล
from keras.layers import Dense # ส่วนของ keras เพื่อใช้ในการ custom โมเดล

windll.shcore.SetProcessDpiAwareness(1) # ทำให้ GUI ไม่เบลอ

# Custom model
abt_model = load_model("./Model/apple_banana_tomato_model") # โหลดโมเดลมา
 
abt_output = Dense(3, activation='softmax') # ทำการ set เอาต์พุต เป็น 3 เลเยอร์
abt_output = abt_output(abt_model.layers[-2].output)  

abt_input = abt_model.input # โมเดลอินพุต
abt_model = Model(inputs=abt_input, outputs=abt_output) # ใส่ ค่า input output ไปในโมเดล

# ไม่ให้เทรน เลเยอร์อื่นๆ ค่าที่เป็นของ mobilenet_v2
for layer in abt_model.layers[:-1]:
  layer.trainable = False

# Compile Model
abt_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

gui = Tk() # เรียกใช้ tkinter ในการสร้าง gui
gui.title("Apple Banana Tomato Prediction") # ชื่อโปรแกรม
gui.geometry("860x650") # ขนาดโปรแกรม


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
	score = pred[0]
	# print(abt_model.predict(img_array/255.0, batch_size=32, verbose=0))

	label = np.argmax(pred[0])
	if label == 0:
		t0 = "ผลลัพธ์การทำนาย คือ ... Apple"
	if label == 1:
		t0 = "ผลลัพธ์การทำนาย คือ ... Banana"
	if label == 3:
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

frm = ttk.Frame(gui)
frm.pack(side=BOTTOM, padx=15, pady=15)

lbl = ttk.Label(gui)
lbl.pack()

# Browse Image
btn = ttk.Button(frm, text="Browse Image", command=show_image)
btn.pack(side=tk.LEFT, padx=10)

# Prediction Button
btn_prediction = ttk.Button(frm, text="Prediction", command=prediction)
btn_prediction.pack(side=tk.LEFT, padx=10)

# Clear Button
btn_clear = ttk.Button(frm, text="clear", command=clear_text)
btn_clear.pack(side=tk.LEFT, padx=10)

# Exit Button
btn_exit = ttk.Button(frm, text="Exit", command=lambda: exit())
btn_exit.pack(side=tk.LEFT, padx=10)

gui.mainloop()
