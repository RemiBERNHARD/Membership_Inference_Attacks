from __future__ import division
import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= sys.argv[1]

import numpy as np
from attacks_base import attack_label, attack_ce, attack_lossce, attack_conf, attack_mentr, attack_shokri
from attacks_threshold_choice import attack_ce_threshold, attack_lossce_threshold, attack_conf_threshold, attack_mentr_threshold
from utils_mia import metrics
from keras.utils import np_utils
from keras.models import load_model


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


#################
#################
model_target = sys.argv[2]
print("Model target: " + model_target)

model_shadow = sys.argv[3]
print("Model shadow: " + model_shadow)

model = load_model("models/SVHN_" + model_target + ".h5")

model_subst = load_model("models/SVHN_" + model_shadow + ".h5")




#################
#################
indices_train = np.load("indices/indices_train.npy")
indices_test = np.load("indices/indices_test.npy")

indices_train_shadow = np.load("indices/indices_train_shadow.npy")
indices_test_shadow = np.load("indices/indices_test_shadow.npy")

X_train_target = X_train[indices_train]
y_train_target = y_train[indices_train]
X_test_target = X_test[indices_test]
y_test_target = y_test[indices_test]

#Shadow data : data from the shadow data set available to the adversary
X_train_shadow = X_train[indices_train_shadow]
y_train_shadow = y_train[indices_train_shadow]
X_test_shadow = X_test[indices_test_shadow]
y_test_shadow = y_test[indices_test_shadow]



Y_train_target = np_utils.to_categorical(y_train_target, 10)
Y_test_target = np_utils.to_categorical(y_test_target, 10)
Y_train_shadow = np_utils.to_categorical(y_train_shadow, 10)
Y_test_shadow = np_utils.to_categorical(y_test_shadow, 10)


print("Target model, accuracy on train set: " + str(model.evaluate(X_train_target, Y_train_target, verbose=0)[1]))
print("Target model, accuracy on test set: " + str(model.evaluate(X_test_target, Y_test_target, verbose=0)[1]))

print("Shadow model, accuracy on train set: " + str(model_subst.evaluate(X_train_shadow, Y_train_shadow, verbose=0)[1]))
print("Shadow model, accuracy on test set: " + str(model_subst.evaluate(X_test_shadow, Y_test_shadow, verbose=0)[1]))







l = 1000
#Target data : data from the training and testing set of the target model
X_target = np.concatenate((X_train_target[:l], X_test_target[:l]))
y_target = np.concatenate((y_train_target[:l], y_test_target[:l]))
memb_true_target = np.concatenate((np.ones(l), np.zeros(l)))

#Shadow data : data from the shadow data set available to the adversary
X_shadow = np.concatenate((X_train_shadow[:l], X_test_shadow[:l]))
y_shadow = np.concatenate((y_train_shadow[:l], y_test_shadow[:l]))
memb_true_shadow = np.concatenate((np.ones(l), np.zeros(l)))


#################
#####Attack label
print("Attack label")
#No threshold for this attack

#Apply attack on target model
memb_pred = attack_label(model, X_target, y_target)
metrics_base = metrics(memb_pred, memb_true_target)
print("Accuracy on target model: " + str(metrics_base))


#################
#####Attack ce
print("Attack ce")
#Choose threshold on substitute model with shadow data
threshold_adv, acc_adv =  attack_ce_threshold(model_subst, X_shadow, memb_true_shadow, 0)
#print("threshold adv: " + str(threshold_adv))
print("Accuracy on shadow model: " + str(acc_adv))

#Apply attack on target model
memb_pred = attack_ce(model, X_target, threshold_adv)
metrics_res = metrics(memb_pred, memb_true_target)
print("Accuracy on target model: " + str(metrics_res))
print("Accuracy difference: " + str(metrics_res[0] - metrics_base[0]))

#################
#####Attack lossce
print("Attack lossce")
#Choose threshold on substitute model with shadow data
threshold_adv, acc_adv =  attack_lossce_threshold(model_subst, X_shadow, y_shadow, memb_true_shadow, 0)
#print("threshold adv: " + str(threshold_adv))
print("Accuracy on shadow model: " + str(acc_adv))

#Apply attack on target model
memb_pred = attack_lossce(model, X_target, y_target, threshold_adv)
metrics_res = metrics(memb_pred, memb_true_target)
print("Accuracy on target model: " + str(metrics_res))
print("Accuracy difference: " + str(metrics_res[0] - metrics_base[0]))

#################
######Attack conf
#print("Attack conf")
##Choose threshold on substitute model with shadow data
#threshold_adv, acc_adv =  attack_conf_threshold(model_subst, X_shadow, y_shadow, memb_true_shadow)
#print("threshold adv: " + str(threshold_adv))
#print("Accuracy on shadow model: " + str(acc_adv))
#
##Apply attack on target model
#memb_pred = attack_conf(model, X_target, y_target, threshold_adv)
#metrics_res = metrics(memb_pred, memb_true_target)
#print("Accuracy on target model: " + str(metrics_res))
#print("Accuracy difference: " + str(metrics_res[0] - metrics_base[0]))


#################
#####Attack mentr
print("Attack mentr")
#Choose threshold on substitute model with shadow data
threshold_adv, acc_adv =  attack_mentr_threshold(model_subst, X_shadow, y_shadow, memb_true_shadow, 0)
#print("threshold adv: " + str(threshold_adv))
print("Accuracy on shadow model: " + str(acc_adv))

#Apply attack on target model
memb_pred = attack_mentr(model, X_target, y_target, threshold_adv)
metrics_res = metrics(memb_pred, memb_true_target)
print("Accuracy on target model: " + str(metrics_res))
print("Accuracy difference: " + str(metrics_res[0] - metrics_base[0]))


#################
#####Attack shokri
print("Attack shokri")

X_train_target = X_train[indices_train][:l]
y_train_target = y_train[indices_train][:l]
X_test_target = X_test[indices_test][:l]
y_test_target = y_test[indices_test][:l]

#Shadow data : data from the shadow data set available to the adversary
X_train_shadow = X_train[indices_train_shadow][:l]
y_train_shadow = y_train[indices_train_shadow][:l]
X_test_shadow = X_test[indices_test_shadow][:l]
y_test_shadow = y_test[indices_test_shadow][:l]



nb_classes = 10
prop_class = np.zeros((nb_classes))
acc_class = np.zeros((nb_classes,3))
for c in range(0, nb_classes):
    #print("class :" + str(c))
    #prepare shadow data with only examples from class c
    X_train_shadow_c = X_train_shadow[y_train_shadow == c]
    y_train_shadow_c = y_train_shadow[y_train_shadow == c]
    X_test_shadow_c = X_test_shadow[y_test_shadow == c]
    y_test_shadow_c = y_test_shadow[y_test_shadow == c]
    
    X_shadow = np.concatenate((X_train_shadow_c, X_test_shadow_c))
    y_shadow = np.concatenate((y_train_shadow_c, y_test_shadow_c))
    memb_true_shadow = np.concatenate((np.ones(len(X_train_shadow_c)), np.zeros(len(X_test_shadow_c))))
    
    #Train inference model on shadow data
    memb_pred_shadow, model_f =  attack_shokri(model_subst, X_shadow, memb_true_shadow)
    acc_adv = metrics(memb_pred_shadow, memb_true_shadow)
    #print("Accuracy on shadow model: " + str(acc_adv))
    
    #prepare target data with only examples from class c
    X_train_target_c = X_train_target[y_train_target == c]
    y_train_target_c = y_train_target[y_train_target == c]
    X_test_target_c = X_test_target[y_test_target == c]
    y_test_target_c = y_test_target[y_test_target == c]
    
    X_target = np.concatenate((X_train_target_c, X_test_target_c))
    y_target = np.concatenate((y_train_target_c, y_test_target_c))
    memb_true_target = np.concatenate((np.ones(len(X_train_target_c)), np.zeros(len(X_test_target_c))))

    conf_pred = model.predict(X_target)
    Y_member = np_utils.to_categorical(memb_true_target[memb_true_target==1],2)
    Y_nonmember = np_utils.to_categorical(memb_true_target[memb_true_target==0],2)
    X_data = conf_pred
    Y_data = np.concatenate((Y_member, Y_nonmember))

    #perform attack on target model
    memb_pred = memb_pred = np.argmax(model_f.predict(X_data), axis=1)
    acc_target = metrics(memb_pred, memb_true_target)
    #print("Accuracy on target model: " + str(acc_target))    
        
    #print some results
    #print("len shadow: " + str(len(memb_true_shadow)))
    #print("len target: " + str(len(memb_true_target)))
    
    prop_class[c] = len(memb_true_target)/(len(indices_train[:l]) + len(indices_test[:l]))
    acc_class[c,0] = acc_target[0]
    acc_class[c,1] = acc_target[1]
    acc_class[c,2] = acc_target[2]
    
#print(prop_class)    
#print(acc_class)

res_tot = np.sum(prop_class * acc_class[:,0])
res_train = np.sum(prop_class * acc_class[:,1])
res_test = np.sum(prop_class * acc_class[:,2])

print("Accuracy on target model: (" + str(res_tot) + ", " + str(res_train) + ", " + str(res_test) + ")")
print("Accuracy difference: " + str(res_tot - metrics_base[0]))



