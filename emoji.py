import tkinter as tk
from ttk import *
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
import cv2
import tensorflow as tf
from keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

model = tf.keras.models.Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape =(48,48,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01) ))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(7, activation='softmax'))

model.load_weights('model_weights.h5')

cv2.ocl.setUseOpenCL(False)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

emoji_dist={0:"./emojis/angry.png",1:"./emojis/disgusted.png",2:"./emojis/fearful.png",3:"./emojis/happy.png",4:"./emojis/neutral.png",5:"./emojis/sad.png",6:"./emojis/surpriced.png"}

#global last_frame  # creating global variable
last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
#global cap
show_text = [0]
cap = cv2.VideoCapture(0)


def show_vid():  # creating a function

    if not cap.isOpened():  # checks for the opening of camera
        print("cant open the camera")
    flag, frame = cap.read()
    # frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (400, 300))

    bounding_box = cv2.CascadeClassifier(
        'D:/PycharmProject/pythonProject/EmojiCreator/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        prediction = model.predict(cropped_img)

        maxindex = int(np.argmax(prediction))
        # cv2.putText(frame1, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        show_text[0] = maxindex

    if flag is None:
        print("Major error!")
    elif flag:
        global last_frame
        last_frame = frame.copy()

    pic = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)  # we can change the display color of the frame gray,black&white here
    img = Image.fromarray(pic)
    imgtk = ImageTk.PhotoImage(image=img)
    live_label1.imgtk = imgtk
    live_label1.configure(image=imgtk)
    live_label1.after(20, show_vid)


def show_vid2():
    frame2 = cv2.imread(emoji_dist[show_text[0]])
    pic2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(frame2)
    img2 = img2.resize((300, 300))
    imgtk2 = ImageTk.PhotoImage(image=img2)
    live_label2.imgtk2 = imgtk2
    live_label3.configure(text=emotion_dict[show_text[0]], font=('arial', 45, 'bold'))

    live_label2.configure(image=imgtk2)
    live_label2.after(1, show_vid2)


if __name__ == '__main__':
    root = tk.Tk()  # assigning root variable for Tkinter as tk
    root.state("zoomed")
    root.resizable(0, 0)
    live_label1 = tk.Label(master=root, text="My Label")
    live_label1.grid(column=0, rowspan=4, padx=5, pady=5)
    live_label1.place(x=100, y=500)

    live_label2 = tk.Label(master=root, bd=10, bg='black')
    live_label2.grid(column=0, rowspan=4, padx=5, pady=5)
    live_label2.place(x=550, y=500)

    live_label3 = tk.Label(master=root, bd=10, fg="#CDCDCD", bg='black')
    live_label3.pack()
    live_label3.place(x=550, y=400)

    root.title("Sign Language Processor")  # you can give any title
    show_vid()
    show_vid2()
    root.mainloop()  # keeps the application in an infinite loop so it works continuosly
    cap.release()