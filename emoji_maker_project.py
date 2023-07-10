import tkinter as tk
import winsound
from tkinter import filedialog
from tkinter import messagebox
from tkinter.filedialog import askopenfile
from ttk import *
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
import cv2
import tensorflow as tf
import time
from keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from skimage import color
from skimage import io
import pygame
from PyQt5.QtMultimedia import *
from PyQt5.QtCore import QUrl


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

ret = True
frame = None
cam_on = False
cap = None
show_text = [0]
mainWindow = Tk()
mainWindow.state("zoomed")

mainFrame = Frame(mainWindow, highlightbackground='blue', highlightthickness=2, height=350, width=840)
mainFrame.pack(padx=1, pady=10)
mainFrame.place(x=10, y=10)
mainFrame['bg'] = 'black'

live_capture_frame = Frame(mainWindow, highlightbackground='blue', highlightthickness=2, height=350, width=680)
live_capture_frame.place(x=850, y=10)
live_capture_frame['bg'] = 'black'

#-------------------------------------------Live Emotion Capture---------------------------------------------------
last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
#global cap
show_text = [0]
cap = cv2.VideoCapture(0)


def show_vid():  # creating a function

    if not cap.isOpened():  # checks for the opening of camera
        print("cant open the camera")
    flag, frame = cap.read()
    # frame = cv2.flip(frame, 1)
    #frame = cv2.resize(frame, (500, 500))
    #frame = cv2.resize(frame, (350, 342))
    frame = cv2.resize(frame, (347, 295))

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
    img2 = img2.resize((310, 300))
    imgtk2 = ImageTk.PhotoImage(image=img2)
    live_label2.imgtk2 = imgtk2
    live_label3.configure(text=emotion_dict[show_text[0]], font=('arial', 20, 'bold'))

    live_label2.configure(image=imgtk2)
    live_label2.after(1, show_vid2)
#------------------------------------------------------------------------------------------------------------------


'''def live_cap():
    global cam_on, cap
    cam_on = True
    cap = cv2.VideoCapture(0)
    flag, frame = cap.read()
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
        show_text[0] = maxindex

    if flag is None:
        print("Major error!")
    elif flag:
        global last_frame
        last_frame = frame.copy()

    pic = cv2.cvtColor(last_frame,
                       cv2.COLOR_BGR2RGB)  # we can change the display color of the frame gray,black&white here
    img = Image.fromarray(pic)
    imgtk = ImageTk.PhotoImage(image=img)
    vid_lbl.imgtk = imgtk
    vid_lbl.configure(image=imgtk)
    # vid_lbl.after(1, live_cap)

    frame2 = cv2.imread(emoji_dist[show_text[0]])
    pic2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(frame2)
    img2 = img2.resize((300, 300))
    imgtk2 = ImageTk.PhotoImage(image=img2)
    pred_img_label.imgtk2 = imgtk2
    pred_img_label.configure(image=imgtk2)
    pred_img_label.after(1, live_cap)'''


def show_frame():
    if cam_on:
        global ret, frame
        ret, frame = cap.read()
        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # img = Image.fromarray(cv2image).resize((810,640))
            img = Image.fromarray(cv2image).resize((400, 300))
            imgtk = ImageTk.PhotoImage(image=img)
            vid_lbl.imgtk = imgtk
            vid_lbl.configure(image=imgtk)

        vid_lbl.after(10, show_frame)


def start_vid():
    global cam_on, cap
    stop_vid()
    cam_on = True
    cap = cv2.VideoCapture(0)
    show_frame()


def stop_vid():
    global cam_on
    cam_on = False

    if cap:
        cap.release()


def snapshot():
    ret, frame = cap.read()
    if ret:
        image = "IMG" + time.strftime("%H-%M-%S-%d-%m") + ".jpg"
        cv2.imwrite('D:/PycharmProject/pythonProject/EmojiCreator/captured_images/' + image, frame)   # Saves original image
        #cv2.imwrite('D:/PycharmProject/pythonProject/EmojiCreator/captured_images/'+image, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        messagebox.showinfo("Welcome to GFG.", "Hi I'm your message")


def upload_file():
    global img
    # f_types = [('Jpg Files', '*.jpg'), ('Jpeg Files','*.jpeg'), ('PNG Files','*.png')]
    f_types = [('Jpg Files', '*.jpg')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    img = Image.open(filename)
    #img_resized = img.resize((400, 300))  # new width & height
    img_resized = img.resize((150, 150))
    img = ImageTk.PhotoImage(img_resized)
    upload_image_label.imgtk = img
    upload_image_label.configure(image=img)

    img_to_be_pred = cv2.imread(filename)
    imgGray = color.rgb2gray(img_to_be_pred)
    image_from_array = Image.fromarray(imgGray, 'L')
    resize_image = image_from_array.resize((48, 48))
    expanded_input = np.expand_dims(resize_image, axis=0)
    input_data = np.array(expanded_input)
    input_data = input_data / 255
    pred = model.predict(input_data)
    result = pred.argmax()
    # pred_img_label.config(text=emotion_dict[result])

    frame2 = cv2.imread(emoji_dist[result])
    pic2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(pic2)
    img2_resized = img2.resize((150, 150))
    imgtk2 = ImageTk.PhotoImage(image=img2_resized)
    pred_img_label.imgtk2 = imgtk2
    pred_img_label.configure(image=imgtk2)


    pred = pred.flatten()

    li_emotion = []
    for i in pred.argsort()[-2:][::-1]:
        li_emotion.append(str(emotion_dict[i]) + " " + str(int(np.ceil(pred[i] * 100))) + " " + str("%"))

    pred_vale_1.configure(text=li_emotion[0])
    pred_vale_2.configure(text=li_emotion[1])


def close():
    mainWindow.destroy()


vid_lbl = Label(mainFrame, text='My Label')
vid_lbl.grid(row=0, column=0)
vid_lbl.place(x=100, y=10)
# vid_lbl.place(relx = 0.5, rely = 0.5, anchor = 'se')

snapshot = Button(mainFrame, text="Snapshot", width=30, bg='goldenrod2', activebackground='red', command=snapshot)
snapshot.place(x=190, y=310)

TurnCameraOn = Button(mainFrame, text="Open Camera", bg="blue", fg="white", command=start_vid, pady=20)
TurnCameraOn.place(x=10, y=10)

#TurnCameraOff = Button(mainFrame, text="stop Video", bg="blue", fg="white", command=stop_vid)
#TurnCameraOff.place(x=10, y=60)

# predict_button = Button(mainWindow, text="Predict", bg = "blue", fg="white")
# predict_button.place(x=700, y=10)

upload_image_label = Label(mainFrame, text='My Upload Label')
upload_image_label.grid(row=0, column=0)
upload_image_label.place(x=520, y=180)

upload_image = Button(mainFrame, text='Upload File and predict', bg="blue", fg="white", wraplength=80,
                      command=upload_file)
upload_image.place(x=10, y=110)

#live_predict = Button(mainFrame, text='Live predict', bg="blue", fg="white", wraplength=80, command=live_cap)
#live_predict.place(x=10, y=180)

exit = Button(mainFrame, text='Exit', bg="blue", fg="white", wraplength=80, command=close)
exit.place(x=10, y=230)

pred_img_label = Label(mainFrame, text='Pred image output')
pred_img_label.place(x=520, y=10)

pred_vale_1 = Label(mainFrame, text='First Prediction', anchor='n', padx=11, pady=64, font=('Times', 15), width=11)
pred_vale_1.place(x=680, y=10)

pred_vale_2 = Label(mainFrame, text='Second Prediction', padx=11, pady=64, font=('Times', 15), width=11)
pred_vale_2.place(x=680, y=180)


live_label1 = tk.Label(master=live_capture_frame, text="My Label")
live_label1.grid(column=0, rowspan=4, padx=5, pady=5)
live_label1.place(x=5, y=40)

live_label2 = tk.Label(master=live_capture_frame, bd=5, bg='black')
live_label2.grid(column=0, rowspan=4, padx=5, pady=5)
live_label2.place(x=355, y=35)

live_label3 = tk.Label(master=live_capture_frame, bd=2, fg="#CDCDCD", bg='black')
live_label3.pack()
live_label3.place(x=450, y=0)

live_detection_label = tk.Label(master=live_capture_frame, text="LIVE DETECTION", font=('arial', 20, 'bold'), bd=2, fg="#CDCDCD", bg='black')
live_detection_label.place(x=50, y=0)

show_vid()
show_vid2()

mainWindow.mainloop()
