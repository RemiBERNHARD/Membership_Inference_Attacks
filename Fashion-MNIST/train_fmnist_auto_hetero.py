import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= sys.argv[1]

from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import binary_crossentropy
import keras.backend as K
import numpy as np
import random
random.seed(123)
import sys


#Load data set
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
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


#Create auto-hetero model
inputs = Input(shape=(784,))
l = Dense(units=600, activation="relu")(inputs)
l = Dropout(0.5)(l)
l = Dense(units=400, activation="relu")(l)
l = Dropout(0.5)(l)
encoded = Dense(100, activation="relu")(l)
l = Dropout(0.5)(l)
l = Dense(units=400, activation="relu")(encoded)
l = Dense(units=600, activation="relu")(l)
l = Dropout(0.5)(l)

decoded = Dense(units=784, activation="sigmoid", name="auto")(l)
pred = Dense(units=10, activation="sigmoid", name="hetero")(l)

model_auto_hetero = Model(inputs, outputs=[decoded, pred])
model_auto_hetero.compile(optimizer=Adam(lr=0.001), loss=["binary_crossentropy", "binary_crossentropy"])

#inputs= Input(shape=(784,))
#l = Dense(400, activation="relu", use_bias=True, kernel_initializer="glorot_normal")(inputs)
#l = Dense(400, activation="relu", use_bias=True, kernel_initializer="glorot_normal")(l)
#l  = Dense(400, activation="relu", use_bias=True, kernel_initializer="glorot_normal")(l)
#
#output_2  = Dense(10, activation="softmax", use_bias=True, kernel_initializer="glorot_normal", name='class')(l)
#output_a  = Dense(100, activation="sigmoid", use_bias=True, kernel_initializer="glorot_normal", name='class_a')(l)
#
#aux = Concatenate()([output_2, output_a])
#l  = Dense(400, activation="relu", use_bias=True, kernel_initializer="glorot_normal")(aux)
#output_1  = Dense(784, activation="sigmoid", use_bias=True, kernel_initializer="glorot_normal", name='auto')(l)
#
#model_auto_hetero = Model(inputs=inputs, outputs=[output_1, output_2, output_a])
#model_auto_hetero.compile(optimizer='adam', loss={'auto': 'binary_crossentropy', 'class': 'categorical_crossentropy', 'class_a': 'binary_crossentropy'})

#Train model auto-hetero
def lr_schedule(step):
    lr = 0.001
    if step > 10000:
        lr = 0.0005
    elif step > 15000:
        lr = 0.0008
    elif step > 20000:
        lr = 0.00001        
    return lr    

generator=ImageDataGenerator(rotation_range=10,
                             width_shift_range=5./32,
                             height_shift_range=5./32)
#generator=ImageDataGenerator()
generator.fit(X_train)
batch_size = 128
generator = generator.flow(X_train, Y_train, batch_size=batch_size)
for step in np.arange(25000):

    x_batch, y_batch = next(generator)
    x_batch = np.reshape(x_batch, (x_batch.shape[0], 784))
        
    K.set_value(model_auto_hetero.optimizer.lr, lr_schedule(step))
    model_auto_hetero.train_on_batch(x_batch, {"auto": x_batch, "hetero": y_batch})
    
    if (step % 100 == 0):
        print("step number: " + str(step))
        print("learning_rate: " + str(K.eval(model_auto_hetero.optimizer.lr)))
        
        pred_train = model_auto_hetero.predict(np.reshape(X_train, (len(X_train), 784)))
        label_train = np.argmax(pred_train[1], axis=1)
        acc_train = np.mean(np.equal(label_train, y_train))
        print("Accuracy auto-hetero on train set: " + str(acc_train))
        
        pred_test = model_auto_hetero.predict(np.reshape(X_test, (len(X_test), 784)))
        label_test = np.argmax(pred_test[1], axis=1)
        acc_test = np.mean(np.equal(label_test, y_test))
        print("Accuracy auto-hetero on test set: " + str(acc_test))

        bce_train = np.mean(K.eval(binary_crossentropy(K.constant(np.reshape(X_train, (len(X_train), 784))), K.constant(pred_train[0]))))
        print("Bce loss on train set:" + str(bce_train))
        bce_test = np.mean(K.eval(binary_crossentropy(K.constant(np.reshape(X_test, (len(X_test), 784))), K.constant(pred_test[0]))))
        print("Bce loss on test set:" + str(bce_test))
         
    if (step % 500 == 0):
        model_auto_hetero.save_weights("weights/fmnist_autohetero_shadow_step:" + str(step) + "_acc_test:" + str(acc_test) + "_bcetest:" + str(bce_test) + ".hdf5")


#model_auto_hetero.load_weights("weights/fmnist_autohetero_shadow_step:24000_acc_test:0.9018_bcetest:0.27164555.hdf5")
#model_auto_hetero.save("models/FMNIST_auto_hetero.h5")
