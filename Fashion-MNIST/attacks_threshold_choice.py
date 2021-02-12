import numpy as np
from attacks_base import c_ent, attack_ce, c_ent_loss, attack_lossce, attack_conf, mentr, attack_mentr
from utils_mia import metrics
from keras.utils import np_utils



#########################
def attack_ce_threshold(model_subst, X_vec, memb_true, met_ind):
    conf_pred = model_subst.predict(X_vec)
    ce_pred = np.apply_along_axis(c_ent, 1, conf_pred)
    acc_res = np.zeros(len(ce_pred))    
    i = 0
    for ce_val in ce_pred:
        memb_pred = attack_ce(model_subst, X_vec, ce_val)
        acc_res[i] = metrics(memb_pred, memb_true)[met_ind]
        i = i + 1
    threshold = ce_pred[np.argmax(acc_res)]
    return threshold, np.max(acc_res)



#########################
def attack_lossce_threshold(model_subst, X_vec, y_vec, memb_true, met_ind):
    Y_vec = np_utils.to_categorical(y_vec,10)
    conf_pred = model_subst.predict(X_vec)
    lossce_pred = np.zeros(conf_pred.shape[0])
    for i in range(0, conf_pred.shape[0]):
        lossce_pred[i] = c_ent_loss(conf_pred[i], Y_vec[i])                     
    acc_res = np.zeros(len(lossce_pred))        
    i = 0
    for lossce_val in lossce_pred:
        memb_pred = attack_lossce(model_subst, X_vec, y_vec, lossce_val)
        acc_res[i] = metrics(memb_pred, memb_true)[met_ind]
        i = i + 1   
    threshold = lossce_pred[np.argmax(acc_res)]
    return threshold, np.max(acc_res)



#########################
def attack_conf_threshold(model_subst, X_vec, y_vec, memb_true, met_ind):
    Y_vec = np_utils.to_categorical(y_vec,10)
    conf_pred = model_subst.predict(X_vec)
    conf_label_pred = conf_pred[np.where(Y_vec==1)]
    acc_res = np.zeros(len(conf_label_pred))    
    i = 0
    for conf_label_val in conf_label_pred:
        memb_pred = attack_conf(model_subst, X_vec, y_vec, conf_label_val)
        acc_res[i] = metrics(memb_pred, memb_true)[met_ind]
        i = i + 1
    threshold = conf_label_pred[np.argmax(acc_res)]
    return threshold, np.max(acc_res)



#########################
def attack_mentr_threshold(model_subst, X_vec, y_vec, memb_true, met_ind):
    Y_vec = np_utils.to_categorical(y_vec,10)
    conf_pred = model_subst.predict(X_vec)
    mentr_pred = np.zeros(conf_pred.shape[0])
    for i in range(0, conf_pred.shape[0]):
        mentr_pred[i] = mentr(conf_pred[i], Y_vec[i])
    acc_res = np.zeros(len(mentr_pred))    
    i = 0    
    for mentr_val in mentr_pred:
        memb_pred = attack_mentr(model_subst, X_vec, y_vec, mentr_val)
        acc_res[i] = metrics(memb_pred, memb_true)[met_ind]
        i = i + 1
    threshold = mentr_pred[np.argmax(acc_res)]
    return threshold, np.max(acc_res)
















