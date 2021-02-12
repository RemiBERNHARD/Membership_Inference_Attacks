import numpy as np
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam

#########################
def attack_label(model, X_vec, y_vec):
    y_pred = np.argmax(model.predict(X_vec), axis=1)    
    memb_pred = np.equal(y_pred, y_vec)
    return(memb_pred)


#########################
def c_ent(vec):
    vec = np.clip(vec, 1e-12, 1. - 1e-12)
    ce = -np.sum(vec*np.log(vec+1e-9))
    return(ce)

def attack_ce(model, X_vec, threshold):
    conf_pred = model.predict(X_vec)
    ce_pred = np.apply_along_axis(c_ent, 1, conf_pred)
    memb_pred = np.less_equal(ce_pred, threshold)
    return(memb_pred)


#########################
def c_ent_loss(vec_pred, vec_true):
    vec_pred = np.clip(vec_pred, 1e-12, 1. - 1e-12)
    ce = -np.sum(vec_true*np.log(vec_pred+1e-9))
    return(ce)

def attack_lossce(model, X_vec, y_vec, threshold):
    Y_vec = np_utils.to_categorical(y_vec,10)
    conf_pred = model.predict(X_vec)
    lossce_pred = np.zeros(conf_pred.shape[0])
    for i in range(0, conf_pred.shape[0]):
        lossce_pred[i] = c_ent_loss(conf_pred[i], Y_vec[i])    
    memb_pred = np.less_equal(lossce_pred, threshold)
    return(memb_pred)


#########################
def attack_conf(model, X_vec, y_vec, threshold):
    Y_vec = np_utils.to_categorical(y_vec,10)
    conf_pred = model.predict(X_vec)
    conf_label_pred = conf_pred[np.where(Y_vec==1)]
    memb_pred = np.greater_equal(conf_label_pred, threshold)
    return(memb_pred)


#########################
def mentr(conf_vec, Y_vec):
    conf_y = conf_vec[np.where(Y_vec==1)][0]
    conf_noty = conf_vec[np.where(Y_vec!=1)]
    mentr_val = -(1 - conf_y)*np.log(conf_y) - c_ent(conf_noty)
    return(mentr_val)

def attack_mentr(model, X_vec, y_vec, threshold):
    Y_vec = np_utils.to_categorical(y_vec,10)
    conf_pred = model.predict(X_vec)
    mentr_pred = np.array([mentr(conf_pred[i], Y_vec[i]) for i in range(0, conf_pred.shape[0])])
    memb_pred = np.less_equal(mentr_pred, threshold)
    return(memb_pred)
    

#########################
def attack_shokri(model, X_vec, memb_true):
    conf_pred = model.predict(X_vec)
    Y_member = np_utils.to_categorical(memb_true[memb_true==1],2)
    Y_nonmember = np_utils.to_categorical(memb_true[memb_true==0],2)    
    X_data = conf_pred
    Y_data = np.concatenate((Y_member, Y_nonmember))

    input_shape = (10,)
    inputs = Input(shape=input_shape)
    l = Dense(64, activation="tanh")(inputs)
    outputs = Dense(2, activation="softmax")(l)
    model_f = Model(inputs, outputs)

    model_f.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
    model_f.fit(X_data, Y_data, epochs=50, batch_size=10, verbose=0)
    
    memb_pred = np.argmax(model_f.predict(X_data), axis=1)
    return(memb_pred, model_f)


  