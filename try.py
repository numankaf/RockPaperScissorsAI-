import tensorflow as tf
from keras.models import load_model
import cv2
import numpy as np
from random import choice

model = load_model("rock-paper-scissors-model.h5")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        continue
    frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_AREA)
    cv2.rectangle(frame, (100, 100), (450, 600), (255, 255, 255), 2)

    #cv2.rectangle(frame, (800, 100), (1200, 500), (255, 255, 255), 2)

    roi = frame[100:600, 100:450]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 200))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    c = model.predict(images)

    user_move_name=""
    if c[0, 0] == 1:
        user_move_name='Paper'
    elif c[0, 1] == 1:
        user_move_name='Rock'
    elif c[0, 2] == 1:
        user_move_name='Scissor'
    else:
        user_move_name="none"

    font = cv2.FONT_HERSHEY_TRIPLEX
    cv2.putText(frame, "Your Move: " + user_move_name,
                (50, 50), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Rock Paper Scissors", frame)
    k = cv2.waitKey(10)

    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
