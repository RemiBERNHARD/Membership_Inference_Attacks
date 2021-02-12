import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= sys.argv[1]

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Activation, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import numpy as np
import random
random.seed(123)
import sys

#Load data set
X_train = np.load("SVHN_data/X_train.npy")
X_test = np.load("SVHN_data/X_test.npy")
y_train = np.load("SVHN_data/y_train.npy")
y_test = np.load("SVHN_data/y_test.npy")

y_train[y_train==10]=0
y_test[y_test==10]=0

X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
 
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


indices_train = np.load("indices/indices_train.npy")
indices_test = np.load("indices/indices_test.npy")

X_train = X_train[indices_train]
Y_train = Y_train[indices_train]

X_test = X_test[indices_test]
Y_test = Y_test[indices_test]


generator=ImageDataGenerator(rotation_range=10,
                               width_shift_range=5./32,
                               height_shift_range=5./32)
generator.fit(X_train)

inputs = Input(shape=(32,32,3))
l = Conv2D(64, kernel_size=(3,3), strides=(1,1),padding="valid", use_bias=False)(inputs)
l = BatchNormalization()(l)
l = Activation("relu")(l)
l = Conv2D(64, kernel_size=(3,3), strides=(1,1),padding="same", use_bias=False)(l)
l = MaxPooling2D(pool_size=(2,2), strides=(2,2))(l)
l = BatchNormalization()(l)
l = Activation("relu")(l)
l = Conv2D(128, kernel_size=(3,3), strides=(1,1),padding="same", use_bias=False)(l)
l = BatchNormalization()(l)
l = Activation("relu")(l) 
l = Conv2D(128, kernel_size=(3,3), strides=(1,1),padding="same", use_bias=False)(l)
l = MaxPooling2D(pool_size=(2,2), strides=(2,2))(l)
l = BatchNormalization()(l)
l = Activation("relu")(l)
l = Conv2D(256, kernel_size=(3,3), strides=(1,1),padding="same", use_bias=False)(l)
l = BatchNormalization()(l)
l = Activation("relu")(l)
l = Conv2D(256, kernel_size=(3,3), strides=(1,1),padding="same", use_bias=False)(l)
l = MaxPooling2D(pool_size=(2,2), strides=(2,2))(l)
l = BatchNormalization()(l)
l = Activation("relu")(l)
l = Dropout(0.4)(l)
l = Flatten()(l)
l = Dense(1024, use_bias=False)(l)
l = BatchNormalization()(l)
l = Activation("relu")(l)
l = Dropout(0.4)(l)
l = Dense(1024, use_bias=False)(l)
l = BatchNormalization()(l)
l = Activation("relu")(l)
l = Dense(10, activation="softmax", use_bias=False)(l)
predictions = l

model = Model(inputs=inputs, outputs=predictions)
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
generator = generator.flow(X_train, Y_train, batch_size=28)
                           
for step in np.arange(50000):
    x_batch, y_batch = next(generator)
    model.train_on_batch(x_batch, y_batch)
    
    if (step % 1000 == 0):
        print("step number: " + str(step))
        
        acc_train = model.evaluate(X_train, Y_train, verbose=0)[1]
        print("Accuracy auto-hetero on train set: " + str(acc_train))
        
        acc_test = model.evaluate(X_test, Y_test, verbose=0)[1]
        print("Accuracy auto-hetero on test set: " + str(acc_test))
        
    if (step % 1000 == 0):
        model.save_weights("weights/svhn_" + str(step) + "_acc_train:" + str(acc_train) + "_acc_test: " + str(acc_test) + ".hdf5")




#model.load_weights("weights/svhn_10000_acc_train:0.94232_acc_test: 0.934.hdf5")
#
#print(model.evaluate(X_train, Y_train, verbose=0))
#print(model.evaluate(X_test, Y_test, verbose=0))
#
#model.save("models/SVHN_target_3.h5")
