from __future__ import division
import numpy as np


def metrics(memb_pred, memb_true):
   res = np.sum(np.equal(memb_pred, memb_true))
   percent = np.sum(np.equal(memb_true,1))/len(memb_true)
   res_train = np.sum(np.equal(memb_pred[:int(percent*len(memb_pred))], memb_true[:int(percent*len(memb_true))]))		
   res_test = np.sum(np.equal(memb_pred[int(percent*len(memb_pred)):], memb_true[int(percent*len(memb_true)):]))	
   acc_tot = res/len(memb_true)
   acc_train = res_train/(percent*len(memb_pred))
   acc_test = res_test/(len(memb_true)-percent*len(memb_pred))
   
   return acc_tot, acc_train, acc_test	

