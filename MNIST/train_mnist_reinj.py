import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= sys.argv[1]

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import load_model
import keras.backend as K
import numpy as np
import random
random.seed(123)
import sys


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


#Load auto_hetero model
model_auto_hetero = load_model("models/MNIST_auto_hetero.h5")


#Create Target classifier
inputs = Input(shape=(28,28,1))
l = Conv2D(32, kernel_size=(5, 5), strides=(1,1), activation='relu', padding="same", use_bias=False)(inputs)
l = MaxPooling2D(pool_size=(2,2), strides=(2,2))(l)
l = Conv2D(64, kernel_size=(5, 5), strides=(1,1), activation='relu', padding="same", use_bias=False)(l)
l = MaxPooling2D(pool_size=(2,2), strides=(2,2))(l)
l = Flatten()(l)
l = Dense(1024, activation='relu')(l)
predictions = Dense(10, activation='softmax')(l)

model = Model(inputs=inputs, outputs=predictions)
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])




##Train target model with auto-hetero generated samples and labels
batch_size_for_reinjection = 128
number_of_reinjections = 5

def lr_schedule(step):
    lr = 0.001
    if step > 10000:
        lr = 0.0005
    if step > 15000:
        lr = 0.0008
    if step > 20000:
        lr = 0.00001        
    return lr    
  
for step in range(25000): 
    
    #Generate synthetic inputs    
    synthetic_features = np.array([], dtype=np.float32).reshape(0,784)
    synthetic_labels= np.array([], dtype=np.float32).reshape(0,10)
    
    pseudo_examples_before_reinjection = np.random.randn(batch_size_for_reinjection,784)
    for R in range (number_of_reinjections):
            
            pseudo_examples_after_reinjection = model_auto_hetero.predict_on_batch(pseudo_examples_before_reinjection)
            
            if R>0: 
                synthetic_features = np.concatenate((synthetic_features, pseudo_examples_before_reinjection))
                synthetic_labels = np.concatenate((synthetic_labels,pseudo_examples_after_reinjection[1]))
            pseudo_examples_before_reinjection = pseudo_examples_after_reinjection[0]

    #Train target model
    K.set_value(model.optimizer.lr, lr_schedule(step))
    
    synthetic_features = np.reshape(synthetic_features, (synthetic_features.shape[0],28,28,1))
    
    model.train_on_batch(synthetic_features, synthetic_labels) 
    
    if (step % 100 == 0):
        print("step number: " + str(step))
        print("learning_rate: " + str(K.eval(model.optimizer.lr)))
        acc_train = model.evaluate(X_train, Y_train, verbose=0)[1]
        acc_test = model.evaluate(X_test, Y_test, verbose=0)[1]
        print("Model, accuracy on train set: " + str(acc_train))
        print("Model, accuracy on test set: " + str(acc_test))
        
    if (step % 500 == 0):
        model.save_weights("weights/mnist_target_step:" + str(step) + "_acctrain:" + str(acc_train) + "_acctest:" + str(acc_test) + ".hdf5")
    
    
#model.load_weights("weights/mnist_target_step:18500_acctrain:0.942_acctest:0.9136.hdf5")    
#    
#model.save("models/2MNIST_target_reinj.h5")    
#    
#acc_train = model.evaluate(X_train, Y_train, verbose=0)[1]
#acc_test = model.evaluate(X_test, Y_test, verbose=0)[1]
#print("Model, accuracy on train set: " + str(acc_train))
#print("Model, accuracy on test set: " + str(acc_test))



