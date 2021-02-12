import numpy as np
from keras.utils import np_utils
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


#Get indices for target model
train_len = int(sys.argv[1])
print("Number of examples for training set:" + str(train_len))
rand_choice_train = np.random.choice(np.arange(len(X_train)),train_len)
np.save("indices/indices_train", rand_choice_train)

test_len = int(sys.argv[2])
print("Number of examples for testing set:" + str(test_len))
rand_choice_test = np.random.choice(np.arange(len(X_test)),test_len)
np.save("indices/indices_test", rand_choice_test)


#Get indices for shadow model
indices_train = np.load("indices/indices_train.npy")
rand_choice_train = np.random.choice(np.setdiff1d(np.arange(len(X_train)),indices_train),train_len)
np.save("indices/indices_train_shadow", rand_choice_train)

indices_test = np.load("indices/indices_test.npy")
rand_choice_test = np.random.choice(np.setdiff1d(np.arange(len(X_test)),indices_test),test_len)
np.save("indices/indices_test_shadow", rand_choice_test)








