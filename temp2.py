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
from keras.utils import img_to_array


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

whatsapp_emoji = {0:"./whatsapp_emoji/anger.png", 1:"./whatsapp_emoji/Disgusted.png", 2:"./whatsapp_emoji/Fearful.png", 3:"./whatsapp_emoji/Happy.png", 4:"./whatsapp_emoji/Neutral.png", 5:"./whatsapp_emoji/Sad.png", 6:"./whatsapp_emoji/Surprised.png"}

ret = True
frame = None
cam_on = False
cap = None
show_text = [0]
mainWindow = Tk()
mainWindow.state("zoomed")
#mainWindow.geometry('1536x700+0+0')
mainWindow.resizable(False, False)

mainFrame = Frame(mainWindow, highlightbackground='blue', highlightthickness=2, height=350, width=840)
mainFrame.pack(padx=1, pady=10)
mainFrame.place(x=10, y=75)
mainFrame['bg'] = 'black'

live_capture_frame = Frame(mainWindow, highlightbackground='blue', highlightthickness=2, height=350, width=680)
live_capture_frame.place(x=850, y=75)
live_capture_frame['bg'] = 'black'

artistic_frame = Frame(mainWindow, highlightbackground='#6200EE', highlightthickness=2, height=340, width=1520)
artistic_frame.place(x=10, y=495)
artistic_frame['bg'] = 'black'

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
        messagebox.showinfo("Message Box.", "Image Captured Successfully")


def upload_file():
    global img, predictions
    filename = filedialog.askopenfilename(initialdir="/", title="Select an Image", filetypes=(("Image files", "*.jpg *.jpeg *.png"),))
    img = Image.open(filename)
    #img_resized = img.resize((400, 300))  # new width & height
    img_resized = img.resize((150, 150))
    img = ImageTk.PhotoImage(img_resized)
    upload_image_label.imgtk = img
    upload_image_label.configure(image=img)

    img_to_be_pred = cv2.imread(filename)
    grey_image = cv2.cvtColor(img_to_be_pred, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(
        'D:/PycharmProject/pythonProject/EmojiCreator/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(grey_image, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img_to_be_pred, (x, y), (x + w, y + h), (255, 0, 0))
        roi_gray = grey_image[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        image_pixels = img_to_array(roi_gray)
        image_pixels = np.expand_dims(image_pixels, axis=0)
        image_pixels /= 255
        predictions = model.predict(image_pixels)

    max_index = np.argmax(predictions[0])
    frame2 = cv2.imread(emoji_dist[max_index])
    pic2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(pic2)
    img2_resized = img2.resize((150, 150))
    imgtk2 = ImageTk.PhotoImage(image=img2_resized)
    pred_img_label.imgtk2 = imgtk2
    pred_img_label.configure(image=imgtk2)



    pred = predictions.flatten()

    emotion_numbers = []

    li_emotion = []
    for i in pred.argsort()[-2:][::-1]:
        li_emotion.append(str(emotion_dict[i]) + " " + str(round((pred[i] * 100), 1)) + " " + str("%"))
        emotion_numbers.append(i)

    pred_val1_img = cv2.imread(whatsapp_emoji[emotion_numbers[0]])
    pic3 = cv2.cvtColor(pred_val1_img, cv2.COLOR_BGR2RGB)
    img3 = Image.fromarray(pic3)
    img3_resized = img3.resize((145, 125))
    imgtk3 = ImageTk.PhotoImage(image=img3_resized)
    pred_vale_1.imgtk3 = imgtk3
    pred_vale_1.configure(image=imgtk3, text=li_emotion[0], compound='top', bg='white')

    pred_val2_img = cv2.imread(whatsapp_emoji[emotion_numbers[1]])
    pic4 = cv2.cvtColor(pred_val2_img, cv2.COLOR_BGR2RGB)
    img4 = Image.fromarray(pic4)
    img4_resized = img4.resize((145, 125))
    imgtk4 = ImageTk.PhotoImage(image=img4_resized)
    pred_vale_2.imgtk4 = imgtk4
    pred_vale_2.configure(image=imgtk4, text=li_emotion[1], compound='top', bg='white')



    #pred_vale_1.configure(text=li_emotion[0])
    #pred_vale_2.configure(text=li_emotion[1])

    print(emotion_numbers)

def close():
    mainWindow.destroy()


#------------------------------Function for Artistic frame-----------------------------------------

def upload_file_cartoonify():
    width = 260
    height = 280
    filename= filedialog.askopenfilename(initialdir="/", title="Select an Image", filetypes=(("Image files", "*.jpg *.jpeg *.png"),))
    image = Image.open(filename)
    resized_image = image.resize((width, height))
    img = ImageTk.PhotoImage(resized_image)
    cartoon_label1.img = img
    cartoon_label1.configure(image=img, text='ORIGINAL IMAGE', font=('Times', 18), compound='top', bg='black', fg='white', width=260, height=300)

    Img = cv2.imread(filename)
    Img = cv2.resize(Img, (300, 300))
    GrayImg = cv2.cvtColor(src=Img, code=cv2.COLOR_BGR2GRAY)
    SmoothImg = cv2.medianBlur(src=GrayImg, ksize=5)
    Edges = cv2.adaptiveThreshold(src=SmoothImg, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=7, C=6)
    ColorImg = cv2.bilateralFilter(src=Img, d=9, sigmaColor=220, sigmaSpace=200)
    CartoonImg = cv2.bitwise_and(src1=ColorImg, src2=ColorImg, mask=Edges)
    path = "D:/PycharmProject/pythonProject/EmojiCreator/cartoon_images/cartoon.jpg"
    grey_img_path = "D:/PycharmProject/pythonProject/EmojiCreator/cartoon_images/cartoon_grey.jpg"
    edge_img_path = "D:/PycharmProject/pythonProject/EmojiCreator/cartoon_images/cartoon_edge.jpg"
    cv2.imwrite(path, CartoonImg)
    cv2.imwrite(grey_img_path, GrayImg)
    cv2.imwrite(edge_img_path, Edges)


    grey_image_open = Image.open(grey_img_path)
    grey_image_open = grey_image_open.resize((width, height))
    grey_img = ImageTk.PhotoImage(grey_image_open)
    cartoon_label2.grey_img = grey_img
    cartoon_label2.configure(image=grey_img, text='GREYSCALE IMAGE', font=('Times', 18), compound='top', bg='black', fg='white', width=260, height=300)

    anime_image = Image.open(path)
    resized_image = anime_image.resize((width, height))
    img = ImageTk.PhotoImage(resized_image)
    cartoon_label3.img = img
    cartoon_label3.configure(image=img, text='CARTOON IMAGE', font=('Times', 18), compound='top', bg='black', fg='white', width=260, height=300)

    edge_img_open = Image.open(edge_img_path)
    edge_img_open = edge_img_open.resize((width, height))
    edge_img = ImageTk.PhotoImage(edge_img_open)
    cartoon_label4.edge_img = edge_img
    cartoon_label4.configure(image=edge_img, text='EGDE IMAGE', font=('Times', 18), compound='top', bg='black', fg='white', width=260, height=300)

    pencil_image = cv2.imread(filename)
    GrayImg_pencil = cv2.cvtColor(src=pencil_image, code=cv2.COLOR_BGR2GRAY)
    InvertImg_pencil = cv2.bitwise_not(src=GrayImg_pencil)
    SmoothImg_pencil = cv2.medianBlur(src=InvertImg_pencil, ksize=27)
    IvtSmoothImg = cv2.bitwise_not(SmoothImg_pencil)
    SketchImgPencil = cv2.divide(GrayImg_pencil, IvtSmoothImg, scale=250)
    pencil_img_path = "D:/PycharmProject/pythonProject/EmojiCreator/cartoon_images/pencil_img.jpg"
    cv2.imwrite(pencil_img_path, SketchImgPencil)

    pencil_image_show = Image.open(pencil_img_path)
    pencil_image_resize = pencil_image_show.resize((width, height))
    pen_img = ImageTk.PhotoImage(pencil_image_resize)
    cartoon_label5.pen_img = pen_img
    cartoon_label5.configure(image=pen_img, text='PENCIL ART', font=('Times', 18), compound='top', bg='black', fg='white', width=260, height=300)

#--------------------------------------------------------------------------------------------------



# topic = Label(mainWindow, text="Image gen")
# topic.grid(row=0, column=0)
# topic.place(x=10, y=10)

title1 = Label(mainWindow, text='FACIAL EMOTION RECOGNITION USING CNN & TKINTER', font=('arial', 30, 'bold'), anchor=tk.CENTER, bd=2, fg="white", width=63, borderwidth=3, background="#FF1D58")
title1.place(x=10, y=15)

vid_lbl = Label(mainFrame, text='My Label')
vid_lbl.grid(row=0, column=0)
vid_lbl.place(x=100, y=10)
# vid_lbl.place(relx = 0.5, rely = 0.5, anchor = 'se')

snapshot = Button(mainFrame, text="Snapshot", width=30, bg='goldenrod2', activebackground='red', command=snapshot)
snapshot.place(x=190, y=310)

TurnCameraOn = Button(mainFrame, text="Open Camera", bg="blue", fg="white", command=start_vid, pady=20, height=2)
TurnCameraOn.place(x=10, y=10)

#TurnCameraOff = Button(mainFrame, text="stop Video", bg="blue", fg="white", command=stop_vid)
#TurnCameraOff.place(x=10, y=60)

# predict_button = Button(mainWindow, text="Predict", bg = "blue", fg="white")
# predict_button.place(x=700, y=10)

upload_image_label = Label(mainFrame, text='My Upload Label')
upload_image_label.grid(row=0, column=0)
upload_image_label.place(x=520, y=180)

upload_image = Button(mainFrame, text='Upload File and predict', bg="blue", fg="white", wraplength=80, padx=7, pady=20, height=2,
                      command=upload_file)
upload_image.place(x=10, y=110)

#live_predict = Button(mainFrame, text='Live predict', bg="blue", fg="white", wraplength=80, command=live_cap)
#live_predict.place(x=10, y=180)

exit = Button(mainFrame, text='Exit', bg="blue", fg="white", wraplength=80, padx=27, pady=20, height=2, command=close)
exit.place(x=10, y=210)

pred_img_label = Label(mainFrame, text='Pred image output')
pred_img_label.place(x=520, y=10)

#pred_vale_1 = Label(mainFrame, text='First Prediction', anchor='n', padx=11, pady=64, font=('Times', 15), width=11)
pred_vale_1 = Label(mainFrame, text='First Prediction', font=('Times', 15))
pred_vale_1.place(x=680, y=10)

#pred_vale_2 = Label(mainFrame, text='Second Prediction', padx=11, pady=64, font=('Times', 15), width=11)
pred_vale_2 = Label(mainFrame, text='Second Prediction', font=('Times', 15))
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

title2 = Label(mainWindow, text='IMAGE CARTOONIFIER USING COMPUTER VISION', font=('arial', 30, 'bold'), anchor=tk.CENTER, bd=2, fg="white", width=63, borderwidth=3, background="#6200EE")
title2.place(x=10, y=440)

upload_image = Button(artistic_frame, text='Upload File', bg="blue", fg="white", wraplength=80, command=upload_file_cartoonify)
upload_image.place(x=10, y=11)

cartoon_label1 = Label(artistic_frame, text="My Image1", width=37, height=21, background="#B35A20")
cartoon_label1.place(x=90, y=10)

cartoon_label2 = Label(artistic_frame, text="My Image2", width=37, height=21, background="#E8891D")
cartoon_label2.place(x=370, y=10)

cartoon_label3 = Label(artistic_frame, text="My Image3", width=37, height=21, background="#BFD5C9")
cartoon_label3.place(x=650, y=10)

cartoon_label4 = Label(artistic_frame, text="My Image4", width=37, height=21, background="#05A3A4")
cartoon_label4.place(x=930, y=10)

cartoon_label5 = Label(artistic_frame, text="My Image5", width=37, height=21, background="#006373")
cartoon_label5.place(x=1212, y=10)

show_vid()
show_vid2()

mainWindow.mainloop()
