import keras
import sklearn
from sklearn.model_selection import cross_validate
import sklearn.metrics
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.optimizers import Adam,SGD
from keras.layers import Dense, Dropout, Activation, MaxPool2D, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from keras import initializers

import cv2
import numpy as np
import glob

NegativeImages= [cv2.imread(file,0) for file in glob.glob("Covid19Dataset/CT_NonCOVID/*png")]
PositiveImages = [cv2.imread(file,0) for file in glob.glob("Covid19Dataset/positive/*.png")]

#combonong positive & negative
Combined=NegativeImages+PositiveImages

#resizing the images
dataset_x=[]
for img in Combined:
    img=cv2.resize(img, (128,128))
    dataset_x.append(np.reshape(img,[128,128,1]))  

#Labels
oneHotEncode=np.eye(2)
nLabels=[oneHotEncode[0].astype(int) for i in range(len(NegativeImages)) ]
pLabels=[oneHotEncode[1].astype(int) for i in range(len(PositiveImages)) ]
dataset_y=nLabels+pLabels

dataset_x = np.asarray(dataset_x)
dataset_y = np.asarray(dataset_y)

#shuffle dataset
p = np.random.permutation(len(dataset_x))
dataset_x = dataset_x[p]
dataset_y = dataset_y[p]

#getting the test and train dataset
X_test = dataset_x[:int(len(dataset_x)*0.3)]
Y_test = dataset_y[:int(len(dataset_x)*0.3)]

X_train = dataset_x[int(len(dataset_x)*0.3):]
Y_train = dataset_y[int(len(dataset_x)*0.3):]

#CNN network
def customCNN(Xtrain, Ytrain, Xtest, Ytest,optimiser='Adam'):
    batch_size = 128
    num_classes = 15
    epochs = 12
    img_rows, img_cols = X_train.shape[1],X_train.shape[2]
    input_shape = (img_rows, img_cols, 1)
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same',activation ='relu', input_shape = input_shape))

    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))




    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))                  
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))



    model.add(Flatten())
    model.add(Dense(1024, kernel_initializer=initializers.he_normal(seed=None),activation='relu'))  
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation = "softmax"))

    model.compile(optimizer = optimiser , loss = "binary_crossentropy", metrics=["acc"])
    history=model.fit(Xtrain, Ytrain,epochs=epochs)
    score = model.evaluate(Xtest, Ytest, verbose=1)

    print('\nAccuracy:', score[1])

    return model.summary()

#with Adam Optimization
print("--------------------------")
print("Optimizer used=Adam")
summary=customCNN(X_train, Y_train, X_test, Y_test,optimiser='Adam')

#with SGD Optimization
print("--------------------------")
print("Optimizer used=SGD")
summary=customCNN(X_train, Y_train, X_test, Y_test,optimiser='SGD')

#with RMSProp Optimization
print("--------------------------")
print("Optimizer used=RMSProp")
summary=customCNN(X_train, Y_train, X_test, Y_test,optimiser='RMSProp')



    
    