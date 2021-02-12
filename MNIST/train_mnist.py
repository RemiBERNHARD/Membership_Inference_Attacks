import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= sys.argv[1]

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import random
random.seed(123)
import sys
import keras.backend as K

#Load data set
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)



indices_train = np.load("indices/indices_train.npy")
indices_test = np.load("indices/indices_test.npy")

X_train = X_train[indices_train]
y_train = y_train[indices_train]
Y_train = Y_train[indices_train]

X_test = X_test[indices_test]
y_test = y_test[indices_test]
Y_test = Y_test[indices_test]

generator=ImageDataGenerator()
generator.fit(X_train)

inputs = Input(shape=(28,28,1))
l = Conv2D(32, kernel_size=(5, 5), strides=(1,1), activation='relu', padding="same", use_bias=False)(inputs)
l = MaxPooling2D(pool_size=(2,2), strides=(2,2))(l)
l = Conv2D(64, kernel_size=(5, 5), strides=(1,1), activation='relu', padding="same", use_bias=False)(l)
l = MaxPooling2D(pool_size=(2,2), strides=(2,2))(l)
l = Flatten()(l)
l = Dense(1024, activation='relu')(l)
predictions = Dense(10, activation='softmax')(l)      
        
model = Model(inputs=inputs, outputs=predictions)

def lr_schedule(epoch):
    lr = 0.001
    if epoch > 50:
        lr = 0.0001
    if epoch > 50:
        lr = 0.0005
    if epoch > 30:
        lr = 0.0008        
    print('Learning rate: ', lr)
    return lr    

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr_schedule(0)), metrics=['accuracy'])
	generator = generator.flow(X_train, Y_train, batch_size=28)
                           
for step in np.arange(50000):
    x_batch, y_batch = next(generator)
    K.set_value(model.optimizer.lr, lr_schedule(step))
    model.train_on_batch(x_batch, y_batch)
    
    if (step % 1000 == 0):
        print("step number: " + str(step))
        
        acc_train = model.evaluate(X_train, Y_train, verbose=0)[1]
        print("Accuracy auto-hetero on train set: " + str(acc_train))
        
        acc_test = model.evaluate(X_test, Y_test, verbose=0)[1]
        print("Accuracy auto-hetero on test set: " + str(acc_test))
        
    if (step % 1000 == 0):
        model.save_weights("weights/mnist_" + str(step) + "_acc_train:" + str(acc_train) + "_acc_test: " + str(acc_test) + ".hdf5")

#model.load_weights("weights/mnist_0009_1.000000_0.962600.hdf5")
#
#print(model.evaluate(X_train, Y_train, verbose=0))
#print(model.evaluate(X_test, Y_test, verbose=0))
#
#model.save("models/MNIST_target.h5")




