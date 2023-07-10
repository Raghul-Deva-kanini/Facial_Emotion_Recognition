import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import utils
import os
import cv2

from livelossplot.inputs.tf_keras import PlotLossesCallback
import tensorflow as tf
from keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
#from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

from IPython.display import SVG, Image
# from livelossplot import PlotLossesTensorFlowKeras
print("Tensorflow version:", tf.__version__)

plt.figure(figsize=(8, 8))
count = 1
for i in os.listdir("D:/PycharmProject/pythonProject/EmojiCreator/train/"):
    for j in os.listdir("D:/PycharmProject/pythonProject/EmojiCreator/train/"+i):
        plt.subplot(1, 7, count)
        count +=1
        img = load_img("D:/PycharmProject/pythonProject/EmojiCreator/train/"+i+"/"+j)
        plt.title(i)
        plt.imshow(img)
        plt.axis('off')
        break

# Checking number of images in each each training expressions
for i in os.listdir("D:/PycharmProject/pythonProject/EmojiCreator/train/"):
    print("Number of images in "+ i+ " : "+ str(len(os.listdir('D:/PycharmProject/pythonProject/EmojiCreator/train/'+i))))

# Checking number of images in each each test expressions
for i in os.listdir("D:/PycharmProject/pythonProject/EmojiCreator/test/"):
    print("Number of images in "+ i+ " : "+ str(len(os.listdir('D:/PycharmProject/pythonProject/EmojiCreator/test/'+i))))


# Data Augmentation

datagen_train = ImageDataGenerator(rescale=1./255,
                                   zoom_range=0.3,
                                   horizontal_flip=True)

train_generator = datagen_train.flow_from_directory('D:/PycharmProject/pythonProject/EmojiCreator/train/',
                                                batch_size = 64,
                                                target_size=(48, 48),
                                                shuffle=True,
                                                color_mode='grayscale',
                                                class_mode='categorical')

datagen_test = ImageDataGenerator(rescale=1./255,
                                   zoom_range=0.3,
                                   horizontal_flip=True)

test_generator = datagen_test.flow_from_directory('D:/PycharmProject/pythonProject/EmojiCreator/test/',
                                                batch_size = 64,
                                                target_size=(48, 48),
                                                shuffle=True,
                                                color_mode='grayscale',
                                                class_mode='categorical')


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

model.compile(optimizer=Adam(learning_rate=0.0005, decay=1e-6),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
print(model.summary())


# Use this only for creating model
'''epochs = 50
steps_per_epoch = train_generator.n/train_generator.batch_size
testing_steps = test_generator.n/test_generator.batch_size

checkpoint = ModelCheckpoint("model_weights.h5", monitor="val_accuracy", save_weights_only=True, mode='max', verbose=1)

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor=0.1, patience = 2, min_lr=0.00001, model='auto')

callbacks = [PlotLossesCallback(), checkpoint, reduce_lr]

history = model.fit(
    x=train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=testing_steps,
    callbacks=callbacks
)'''