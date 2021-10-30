import tensorflow as tf
from keras.models import load_model
import numpy as np
import cv2

model = load_model("Model/abtmodel")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    new_img = cv2.resize(frame, (frame.shape[1] // 1, frame.shape[0] // 1))
    resized = cv2.resize(new_img, (224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(resized)
    img_array = tf.expand_dims(img_array, 0)  # ขยายมิติภาพฟิตกับโมดล
    predictions = model.predict(img_array)  # ทำนายบน ROI (Region of Interest)
    maxPred = model.getTotalClasses()
    print(maxPred)
    score = tf.nn.softmax(predictions[0])  # ผลลัพธ์
    label = np.argmax(score)  # หาค่าสูงสุด

    if label == 0:
        new_img = cv2.putText(new_img, "Apple", (55, 65), cv2.FONT_HERSHEY_SIMPLEX,
                              0.8, (0, 255, 0), 2)  # แสดงข้อความ "mask"
        # print("\n\n Apple")

    elif label == 1:
        new_img = cv2.putText(new_img, "Banana", (55, 75),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)  # แสดงข้อความ "mask"
        # print("\n\n Banana")

    elif label == 2:
        new_img = cv2.putText(new_img, "Tomato", (55, 85),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)  # แสดงข้อความ "mask"
        # print("\n\n Tomato")

    else:
        pass
        # new_img = cv2.putText(new_img, "Not Found", (55, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) # แสดงข้อความ "mask"

    cv2.imshow('frame', new_img)
    # cv2.imshow('mask', red_mask )

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
