import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'

import numpy as np
import cv2 as cv
import tensorflow as tf

gpus= tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow import keras




categories=['with_mask','without_mask']
data=[]
for category in categories:
  file_path=os.path.join('data',category)
  label=categories.index(category)

  for file in os.listdir(file_path):
    img_path=os.path.join(file_path,file)
    img=cv.imread(img_path)
    img=cv.resize(img,(224,224))
    data.append([img,label])

print(len(data))

import random
random.shuffle(data)

X=[]
y=[]

for features,label in data:
  X.append(features)
  y.append(label)

X=np.array(X)
y=np.array(y)

X = X/255

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33)


from tensorflow.keras.applications.vgg19 import VGG19

vgg=VGG19()

from tensorflow.keras.models import Sequential
model = Sequential()

for layer in vgg.layers[:-1]:
    model.add(layer)

for layer in model.layers:
    layer.trainable=False


from tensorflow.keras.layers import Dense
model.add(Dense(1,activation='sigmoid'))

opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer=opt)

model.fit(X_train,y_train,epochs=6,validation_data=(X_test,y_test),batch_size=58)



haar = cv.CascadeClassifier('haarcascade_frontalface_default.xml')


def detect_face(img):
    cords=haar.detectMultiScale(img)

    return cords



def draw_label(img,text,pos,bg_color):

    text_size = cv.getTextSize(text, cv.FONT_HERSHEY_COMPLEX,1,cv.FILLED)
    end_x = pos[0] + text_size[0][0] + 2
    end_y = pos[1] + text_size[0][1] - 2
    cv.rectangle(img,pos,(end_x,end_y),bg_color,cv.FILLED)
    cv.putText(img,text,pos, cv.FONT_HERSHEY_COMPLEX,1,(0,0,0),1,cv.LINE_AA)



def facemask_detect(img):

    y_pred=model.predict_classes(img.reshape(1,224,224,3))
    return y_pred[0][0]

cap=cv.VideoCapture(0)

while True:
    ret,frame=cap.read()

    img= cv.resize(frame,(224,224))

    cords = detect_face(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))


    y_pred= facemask_detect(img)



    if y_pred == 0:
        draw_label(frame, "Mask ON", (30, 30), (0, 255, 0))
        for x, y, w, h in cords:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        draw_label(frame, "Mask OFF", (30, 30), (0, 0, 255))
        for x, y, w, h in cords:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv.imshow('window',frame)
    if cv.waitKey(1) & 0xFF == ord('x'):
        break

cv.destroyAllWindows()



